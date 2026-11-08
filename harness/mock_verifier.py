"""Userspace verifier abstraction proof.

The mock verifier listens on a UNIX socket (default ``/tmp/verifier.sock``),
validates telemetry schema against ``near_misses.jsonl`` records, cascades
Tier 1/2 events with a deterministic rule-based router (no model judges), and
mocks a sub-2ms SIGKILL routing latency.

This is *strictly* an abstraction proof -- it validates the control/data
routing contract before any kernel eBPF driver is wired in.  It measures
nothing about production kernel-level mitigation latency.
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_SOCKET = "/tmp/verifier.sock"
SIGKILL_LATENCY_BUDGET_US = 2000  # < 2ms
MOCK_SIGKILL_LATENCY_US = 125.0  # deterministic simulated kernel-call cost


# --------------------------------------------------------------------------
# Telemetry schema
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: type
    required: bool = True


TELEMETRY_SCHEMA: Tuple[FieldSpec, ...] = (
    FieldSpec("ts", (int, float)),
    FieldSpec("event_id", str),
    FieldSpec("tier", int),
    FieldSpec("token", str),
    FieldSpec("logit", (int, float)),
    FieldSpec("kind", str),
)

_TIER_ALLOWED = {1, 2}


def validate_telemetry(record: dict) -> List[str]:
    """Return a list of schema violations for a telemetry record.

    An empty list means the record is schema-valid.
    """
    errors: List[str] = []
    for spec in TELEMETRY_SCHEMA:
        if spec.name not in record:
            if spec.required:
                errors.append(f"missing required field: {spec.name}")
            continue
        val = record[spec.name]
        if not isinstance(val, spec.type):
            errors.append(
                f"field {spec.name} has type {type(val).__name__}, "
                f"expected {getattr(spec.type, '__name__', spec.type)}"
            )
    if "tier" in record and record["tier"] not in _TIER_ALLOWED:
        errors.append(f"tier must be in {sorted(_TIER_ALLOWED)}, got {record['tier']}")
    return errors


def parse_near_misses(path: str) -> List[dict]:
    """Load ground-truth / historical telemetry from a ``near_misses.jsonl`` file."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------
# Deterministic (judge-free) cascade router
# --------------------------------------------------------------------------
@dataclass
class CascadeRouter:
    """Cascades Tier 1 events by promoting every outstanding Tier 2 event.

    No model judges are involved: a Tier 1 telemetry event unconditionally
    escalates a Tier 2 low-priority candidate into a mitigation candidate.
    """

    _pending: Dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )  # token -> tier

    def ingest(self, record: dict) -> Tuple[str, int]:
        """Ingest a validated event.

        Returns ``(action, routed_tier)`` where action is one of
        ``"tier2_defer"``, ``"tier1_cascade"`` or ``"route_sigkill"``.
        """
        token = record["token"]
        tier = record["tier"]
        if tier == 2:
            self._pending[token] = 2
            return "tier2_defer", 2
        # tier 1
        if token in self._pending:
            # promote the deferred tier-2 candidate to a mitigation candidate
            self._pending[token] = 1
            return "tier1_cascade", 1
        return "route_sigkill", 1

    def outstanding_tier2(self) -> int:
        return sum(1 for t in self._pending.values() if t == 2)


class MockSigkillGate:
    """Mocks <2ms SIGKILL routing. Records latency and enforces the budget."""

    def __init__(self, latency_us: float = MOCK_SIGKILL_LATENCY_US) -> None:
        self.latency_us = latency_us
        self.calls: List[float] = []

    def route(self) -> float:
        t0 = time.perf_counter()  # noqa: F841 -- kept for realism
        # The kernel call is mocked, so we just report the simulated cost.
        self.calls.append(self.latency_us)
        return self.latency_us

    @property
    def max_latency_us(self) -> float:
        return max(self.calls, default=0.0)

    def within_budget(self) -> bool:
        # No calls -> vacuously within budget; otherwise max must be < 2ms.
        if not self.calls:
            return True
        return sub_budget(self.max_latency_us)


def sub_budget(latency_us: float) -> bool:
    return latency_us < SIGKILL_LATENCY_BUDGET_US


# --------------------------------------------------------------------------
# UNIX socket server
# --------------------------------------------------------------------------
class _VerifierHandler(socketserver.StreamRequestHandler):
    schema_snapshot: List[list] = []  # populated at class-construction time

    def handle(self) -> None:
        router = CascadeRouter()
        gate = MockSigkillGate()
        for raw in self.rfile:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self._reply({"ok": False, "error": "invalid_json"})
                continue
            errors = validate_telemetry(record)
            if errors:
                self._reply({"ok": False, "errors": errors})
                continue
            action, routed_tier = router.ingest(record)
            latency_us = 0.0
            if action == "route_sigkill":
                latency_us = gate.route()
            self._reply(
                {
                    "ok": True,
                    "action": action,
                    "routed_tier": routed_tier,
                    "latency_us": latency_us,
                    "within_budget": sub_budget(latency_us),
                }
            )

    def _reply(self, payload: dict) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()


class MockVerifierServer:
    """Threading UNIX-socket mock verifier.

    Usage::

        with MockVerifierServer(path) as server:
            reply = server.query({...})
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET) -> None:
        self.socket_path = socket_path
        self._httpd: Optional[socketserver.ThreadingUnixStreamServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "MockVerifierServer":
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._httpd = socketserver.ThreadingUnixStreamServer(
            self.socket_path, _VerifierHandler
        )
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def query(self, record: dict) -> dict:
        """Send one telemetry record and return the verifier's reply dict."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.socket_path)
            sock.sendall((json.dumps(record) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf.decode("utf-8"))

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def __enter__(self) -> "MockVerifierServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()