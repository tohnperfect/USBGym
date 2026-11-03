"""Data manifest tests: the schema validates a good manifest, rejects contract violations, and the
committed data/manifest.json is valid (the 'manifest schema validates' check)."""
from __future__ import annotations

import json
import os

import pytest

from usbg import manifest as M


def test_default_manifest_is_valid():
    assert M.validate_manifest(M.default_manifest()) == []


def test_stream_entry_roles():
    e = M.stream_entry("sim", "sim_pairs", 10, labeled=True, license="synthetic")
    assert e["role"] == "sim_pairs" and e["count"] == 10
    with pytest.raises(ValueError):
        M.stream_entry("bad", "not_a_role", 1, labeled=False, license="x")


def test_validate_catches_contract_violations():
    m = M.default_manifest()
    m["normalization"]["split_by"] = "frame"                 # leakage: not allowed
    problems = M.validate_manifest(m)
    assert any("split_by" in p for p in problems)

    m2 = M.default_manifest()
    m2["normalization"]["range"] = [0, 255]                  # must be [0,1]
    assert any("range" in p for p in M.validate_manifest(m2))

    m3 = M.default_manifest()
    m3["streams"] = [{"name": "x", "role": "sim_pairs"}]     # missing required keys
    assert M.validate_manifest(m3)                            # non-empty problem list


def test_round_trip(tmp_path):
    m = M.default_manifest()
    m["streams"].append(M.stream_entry("sim", "sim_pairs", 5, labeled=True, license="synthetic"))
    p = str(tmp_path / "manifest.json")
    M.write_manifest(p, m)
    loaded = M.load_manifest(p)
    assert M.validate_manifest(loaded) == []
    assert loaded["streams"][0]["name"] == "sim"


def test_committed_manifest_valid():
    path = "data/manifest.json"
    if not os.path.exists(path):
        pytest.skip("data/manifest.json not present")
    with open(path) as fh:
        assert M.validate_manifest(json.load(fh)) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
