"""Shared Parallel.ai HTTP client — the ONE place the three Parallel workers
(``pipelines/parallel/{enrich,deep_research,search}.py``) talk to the vendor.

Lives in ``core/`` and is shipped to every worker container via
``Image.add_local_python_source("core.parallel_client")`` — the proven fleet idiom
for a shared local module (mirror of ``core.name_norm`` in the crosswalk workers).
The three trigger tasks and three Modal entrypoints stay SEPARATE (the directive's
mandate); only this HTTP client is factored out so it is not triplicated.

Auth: header ``x-api-key`` (``PARALLEL_API_KEY`` from the ``parallel-api`` Modal
secret). Base ``https://api.parallel.ai``. Endpoints + body nesting are taken
VERBATIM from §0 of docs/reference/DIRECTIVE_24_PARALLEL_ENRICHMENT_CAPABILITY.md
(the authoritative, settled contract) — nothing here is re-derived from the web or
from any other vendor (Parallel ≠ Exa ≠ icypeas; their async semantics do not apply).

This module is dependency-light (stdlib + ``requests``) so it imports cleanly into
every Modal worker image and is unit-importable without Modal. It owns:

  • ``request`` — one call with jittered exponential backoff on 429 / 5xx (honours
    ``Retry-After``); 4xx (except 429) is terminal.
  • Task API helpers — single run (``create_task_run`` → ``get_task_result`` blocking
    GET with ``timeout``), and Task Groups (``create_task_group`` →
    ``add_group_runs`` ≤1000/POST → ``get_group`` poll to ``is_active==false`` →
    ``get_group_runs`` results with ``include_output``).
  • Search API helper — ``search`` (``POST /v1/search``, SYNCHRONOUS — no run
    lifecycle, no groups, no basis).
  • ``wrap_json_schema`` / ``text_output`` / ``auto_output`` — the REQUIRED
    ``output_schema`` envelopes. A bare JSON-Schema object is INVALID; enrichment
    must be ``{"type":"json","json_schema":{…}}``; deep research is ``{"type":"text"}``
    or ``{"type":"auto"}``.

No business logic, no Lance, no Postgres — those live in the per-mode workers.
"""

from __future__ import annotations

import math
import os
import random
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

# ── Parallel coordinates (§0; env-overridable only for testing) ──────────────────────────
PARALLEL_BASE = os.environ.get("PARALLEL_API_BASE", "https://api.parallel.ai").rstrip("/")

# §0: ≤1,000 runs per POST to /v1/tasks/groups/{id}/runs.
GROUP_RUNS_PER_POST = 1000
# §0: GET /result?timeout=600 — server holds the connection up to this many seconds.
RESULT_TIMEOUT_S = 600
# §0 Basis: per-list-element basis requires this beta header (else basis is one entry
# for the whole array). Enrichment sends it so <field>__confidence can match array fields.
FIELD_BASIS_BETA = "field-basis-2025-11-25"

# §0 status lifecycle. Active = {queued, running, cancelling}; terminal = {completed,
# failed, cancelled}. (action_required is surfaced verbatim; the worker decides.)
ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

_DEFAULT_ATTEMPTS = 6
_HTTP_TIMEOUT = (30, 180)  # (connect, read) — read tolerates a long blocking /result GET


class ParallelError(RuntimeError):
    """A non-retryable Parallel API failure (4xx other than 429, or attempts exhausted).
    Carries the HTTP status + truncated body so the worker can land it on the ledger."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Auth + content headers. ``PARALLEL_API_KEY`` comes from the ``parallel-api`` Modal
    secret (minted from Doppler ``core-x/prd``). Raises if absent — a missing key must
    fail loud BEFORE any spend."""
    key = os.environ.get("PARALLEL_API_KEY")
    if not key:
        raise ParallelError(
            "PARALLEL_API_KEY not set in the parallel-api Modal secret "
            "(mint it from Doppler core-x/prd before deploy)."
        )
    headers = {
        "x-api-key": key, "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": "core-x (https://github.com/bencrane/core-x)",
    }
    if extra:
        headers.update(extra)
    return headers


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    """Honor numeric or HTTP-date Retry-After; ignore unusable server values."""
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                deadline = parsedate_to_datetime(retry_after)
                # Obsolete asctime HTTP dates have no explicit timezone, but are UTC.
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                return max(0.0, deadline.timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
        else:
            if math.isfinite(delay) and delay >= 0:
                return delay
    return min(30.0, (2 ** attempt) + random.random())


def request(
    method: str,
    path: str,
    *,
    json_body: Any = None,
    params: dict | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = _DEFAULT_ATTEMPTS,
    read_timeout: float | None = None,
) -> dict[str, Any]:
    """One Parallel API call with backoff. Returns the parsed JSON body (``{}`` on an
    empty 2xx). 429 / 5xx → jittered exponential backoff honouring ``Retry-After``; any
    other 4xx is terminal (raises ``ParallelError`` immediately — never burns retries on
    a bad request). A 408 on ``GET /result`` is the documented "still active, re-call"
    signal and is returned to the caller as ``{"_status": 408}`` rather than raised, so
    the blocking-result loop can decide to re-poll."""
    url = f"{PARALLEL_BASE}{path}"
    timeout = (_HTTP_TIMEOUT[0], read_timeout if read_timeout is not None else _HTTP_TIMEOUT[1])
    last = None
    for i in range(attempts):
        try:
            resp = requests.request(
                method, url, headers=_headers(headers),
                json=json_body, params=params, timeout=timeout,
            )
            sc = resp.status_code
            if sc < 300:
                return resp.json() if resp.content else {}
            # 408 on the blocking /result GET = "still active" — hand it back, do not raise.
            if sc == 408:
                return {"_status": 408}
            last = f"{sc} {resp.text[:300]}"
            if sc != 429 and sc < 500:  # terminal client error
                raise ParallelError(f"Parallel {method} {path} → {last}", status=sc, body=resp.text[:500])
            retry_after = resp.headers.get("Retry-After")
            sleep = _retry_delay(retry_after, i)
        except ParallelError:
            raise
        except requests.RequestException as exc:
            last = str(exc)
            sleep = min(30.0, (2 ** i) + random.random())
        if i < attempts - 1:
            time.sleep(sleep)
    raise ParallelError(f"Parallel {method} {path} failed after {attempts} attempts: {last}")


# ── output_schema envelopes (§0 — REQUIRED wrapping; a bare object is invalid) ────────────
def wrap_json_schema(json_schema: dict) -> dict:
    """Enrichment output_schema: ``{"type":"json","json_schema": <JSON-Schema subset>}``.
    The caller passes the pure JSON-Schema object (``{"type":"object","properties":…}``);
    this wraps it as §0 requires. A bare JSON-Schema object is NOT valid wire form."""
    return {"type": "json", "json_schema": json_schema}


def text_output(description: str | None = None) -> dict:
    """Deep-research output_schema for a long-form report: ``{"type":"text"}`` (with an
    optional ``description`` steering the report)."""
    out: dict[str, Any] = {"type": "text"}
    if description:
        out["description"] = description
    return out


def auto_output() -> dict:
    """Deep-research output_schema ``{"type":"auto"}`` — let Parallel choose the shape."""
    return {"type": "auto"}


# ── Task API: single run (smoke path) ────────────────────────────────────────────────────
def create_task_run(
    *,
    input_: Any,
    processor: str,
    task_spec: dict | None = None,
    metadata: dict | None = None,
    source_policy: dict | None = None,
    beta_field_basis: bool = False,
) -> dict[str, Any]:
    """``POST /v1/tasks/runs`` (§0) → 202 ``TaskRun {run_id, status:"queued", …}``.

    ``input_`` is sent every run (string OR JSON object). ``processor`` is a free string
    (the tier name). ``task_spec`` (optional; omit → auto) carries the WRAPPED
    ``output_schema``. ``beta_field_basis=True`` adds the ``parallel-beta`` header so
    per-list-element basis is returned."""
    body: dict[str, Any] = {"input": input_, "processor": processor}
    if task_spec is not None:
        body["task_spec"] = task_spec
    if metadata:
        body["metadata"] = metadata
    if source_policy:
        body["source_policy"] = source_policy
    extra = {"parallel-beta": FIELD_BASIS_BETA} if beta_field_basis else None
    return request("POST", "/v1/tasks/runs", json_body=body, headers=extra)


def get_task_result(run_id: str, *, timeout: int = RESULT_TIMEOUT_S,
                    beta_field_basis: bool = False) -> dict[str, Any]:
    """``GET /v1/tasks/runs/{run_id}/result?timeout=…`` (§0) — a BLOCKING GET: the server
    holds the connection up to ``timeout`` seconds, returns 200 ``TaskRunResult
    {run, output}`` when complete. On 408 (still active) this returns ``{"_status": 408}``
    so the caller's bounded loop re-calls. NOT a client busy-loop."""
    extra = {"parallel-beta": FIELD_BASIS_BETA} if beta_field_basis else None
    # read_timeout slightly exceeds the server hold so the socket doesn't close early.
    return request(
        "GET", f"/v1/tasks/runs/{run_id}/result",
        params={"timeout": int(timeout)}, headers=extra,
        read_timeout=int(timeout) + 30,
    )


# ── Task API: Task Groups (batch) ────────────────────────────────────────────────────────
def create_task_group(metadata: dict | None = None) -> dict[str, Any]:
    """``POST /v1/tasks/groups`` (§0) → ``{task_group_id, …}``."""
    return request("POST", "/v1/tasks/groups", json_body={"metadata": metadata or {}})


def add_group_runs(group_id: str, inputs: list[dict], *,
                   default_task_spec: dict | None = None,
                   beta_field_basis: bool = False) -> dict[str, Any]:
    """``POST /v1/tasks/groups/{id}/runs`` (§0) — ``inputs`` is a list of ``TaskRunInput``
    (≤1000; the caller chunks). ``default_task_spec`` sets ONE output_schema for the whole
    batch (per-run overrides allowed but unused here). Each input is
    ``{"input": {...}, "processor": "<tier>"}``."""
    if len(inputs) > GROUP_RUNS_PER_POST:
        raise ParallelError(
            f"add_group_runs got {len(inputs)} inputs > {GROUP_RUNS_PER_POST}/POST cap; chunk first."
        )
    body: dict[str, Any] = {"inputs": inputs}
    if default_task_spec is not None:
        body["default_task_spec"] = default_task_spec
    extra = {"parallel-beta": FIELD_BASIS_BETA} if beta_field_basis else None
    return request("POST", f"/v1/tasks/groups/{group_id}/runs", json_body=body, headers=extra)


def get_group(group_id: str) -> dict[str, Any]:
    """``GET /v1/tasks/groups/{id}`` (§0). Terminal = ``is_active == false``; then read
    ``status.task_run_status_counts.{completed,failed}``."""
    return request("GET", f"/v1/tasks/groups/{group_id}")


def get_group_runs(group_id: str, *, include_output: bool = True,
                   beta_field_basis: bool = False) -> list[dict[str, Any]]:
    """``GET /v1/tasks/groups/{id}/runs?include_output=true`` (§0). The endpoint is an SSE
    stream (one event per run); ``requests`` reads it as a body. This parses the response
    into a flat list of run records regardless of whether the server answers with a JSON
    array, a ``{"runs":[…]}`` / ``{"data":[…]}`` envelope, or newline-delimited SSE
    ``data:`` frames — so the worker gets a uniform ``[{run, output}, …]`` shape."""
    import json as _json

    url = f"{PARALLEL_BASE}/v1/tasks/groups/{group_id}/runs"
    extra = {"parallel-beta": FIELD_BASIS_BETA} if beta_field_basis else None
    params = {"include_output": "true" if include_output else "false"}
    resp = requests.get(url, headers=_headers(extra), params=params, timeout=(_HTTP_TIMEOUT[0], 600))
    if resp.status_code >= 300:
        raise ParallelError(
            f"Parallel GET /tasks/groups/{group_id}/runs → {resp.status_code} {resp.text[:300]}",
            status=resp.status_code, body=resp.text[:500],
        )
    text = resp.text or ""
    # 1) Try a single JSON document (array or enveloped).
    try:
        doc = _json.loads(text)
        if isinstance(doc, list):
            return [r for r in doc if isinstance(r, dict)]
        if isinstance(doc, dict):
            for key in ("runs", "data", "results"):
                if isinstance(doc.get(key), list):
                    return [r for r in doc[key] if isinstance(r, dict)]
            return [doc]
    except ValueError:
        pass
    # 2) Fall back to SSE / NDJSON: collect every parseable JSON object from data: lines.
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("event:", "id:", ":")):
            continue
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if not line:
            continue
        try:
            obj = _json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


# ── Search API (SYNCHRONOUS — no run lifecycle, no groups, no basis) ──────────────────────
def search(*, objective: str | None = None, search_queries: list[str] | None = None,
           mode: str = "advanced", source_policy: dict | None = None) -> dict[str, Any]:
    """``POST /v1/search`` (§0 Web Search) — synchronous; returns ranked excerpts inline.
    Body: ``{objective, search_queries (≤5), mode:"advanced", source_policy?}``. No poll,
    no group, no basis. ``objective`` and/or ``search_queries`` per Parallel's Search API.
    (Result sizing is not exposed here: Parallel nests it under ``advanced_settings``, not a
    top-level ``max_results`` — a bare top-level field would be dead. Add it under
    ``advanced_settings`` if/when a caller needs it.)"""
    body: dict[str, Any] = {"mode": mode}
    if objective:
        body["objective"] = objective
    if search_queries:
        body["search_queries"] = search_queries[:5]
    if source_policy:
        body["source_policy"] = source_policy
    return request("POST", "/v1/search", json_body=body)
