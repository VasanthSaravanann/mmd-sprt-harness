"""Task 1: userspace verifier abstraction proof."""

import json
import os

import pytest

from harness.mock_verifier import (
    CascadeRouter,
    DEFAULT_SOCKET,
    MockSigkillGate,
    MockVerifierServer,
    parse_near_misses,
    sub_budget,
    validate_telemetry,
)

SOCKET_PATH = "/tmp/test_verifier.sock"
NM_PATH = os.path.join(os.path.dirname(__file__), "near_misses.jsonl")


def _rec(tier=2, token="tk_alpha", **overrides):
    r = {"ts": 1.0, "event_id": "e1", "tier": tier, "token": token,
         "logit": 0.4, "kind": "near_miss"}
    r.update(overrides)
    return r


def test_parse_near_misses_schema_source():
    records = parse_near_misses(NM_PATH)
    assert len(records) == 3
    assert all(validate_telemetry(r) == [] for r in records)


def test_valid_telemetry_passes_schema():
    assert validate_telemetry(_rec()) == []


@pytest.mark.parametrize("record", [
    _rec(ts="not-a-number"),
    _rec(logit="high"),
    {"tier": 1, "token": "x"},           # missing required fields
    _rec(tier=3),                        # invalid tier
])
def test_invalid_telemetry_fails_schema(record):
    assert validate_telemetry(record) != []


def test_cascade_tier2_then_tier1_promotes():
    router = CascadeRouter()
    assert router.ingest(_rec(tier=2, token="tk_a"))[0] == "tier2_defer"
    assert router.outstanding_tier2() == 1
    action, routed = router.ingest(_rec(tier=1, token="tk_a"))
    assert action == "tier1_cascade"
    assert routed == 1
    assert router.outstanding_tier2() == 0


def test_tier1_without_pending_routes_sigkill():
    router = CascadeRouter()
    action, routed = router.ingest(_rec(tier=1, token="tk_new"))
    assert action == "route_sigkill"
    assert routed == 1


def test_sigkill_latency_is_sub_2ms():
    gate = MockSigkillGate()
    latency = gate.route()
    assert sub_budget(latency)
    assert sub_budget(gate.max_latency_us)
    assert gate.within_budget()


def test_server_on_unix_socket():
    with MockVerifierServer(SOCKET_PATH) as server:
        assert os.path.exists(SOCKET_PATH)
        reply = server.query(_rec())
        assert reply["ok"] is True
        assert reply["action"] == "tier2_defer"


def test_server_rejects_invalid_json():
    with MockVerifierServer(SOCKET_PATH) as server:
        reply = server.query(_rec(tier=99))
        assert reply["ok"] is False
        assert reply["errors"]


def test_server_routes_sigkill_latency():
    with MockVerifierServer(SOCKET_PATH) as server:
        reply = server.query(_rec(tier=1, token="tk_sig"))
        assert reply["ok"] is True
        assert reply["action"] == "route_sigkill"
        assert reply["latency_us"] < 2000
        assert reply["within_budget"] is True


def test_default_socket_path_is_tmp_verifier():
    assert DEFAULT_SOCKET == "/tmp/verifier.sock"