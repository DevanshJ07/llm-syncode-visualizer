"""
End-to-end test: strict failure handling + successful generations.
Run with backend on http://127.0.0.1:8000
"""
import json
import sys

import requests

BASE = "http://127.0.0.1:8000"


def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200, r.text
    print("OK health")


def test_generate_raw():
    payload = {
        "prompt": "add two integers",
        "mode": "raw",
        "max_new_tokens": 8,
        "use_syncode": False,
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 10,
    }
    r = requests.post(f"{BASE}/generate", json=payload, timeout=300)
    assert r.status_code == 201, f"raw failed: {r.status_code} {r.text[:500]}"
    d = r.json()
    assert d["total_steps"] > 0, d
    assert len(d["steps"]) > 0, d
    assert d["generated_text"].strip(), d
    print(f"OK raw: {d['total_steps']} steps, text={d['generated_text'][:60]!r}")


def test_generate_syncode():
    payload = {
        "prompt": "add two integers",
        "mode": "syncode",
        "max_new_tokens": 8,
        "use_syncode": True,
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 10,
    }
    r = requests.post(f"{BASE}/generate", json=payload, timeout=300)
    assert r.status_code == 201, f"syncode failed: {r.status_code} {r.text[:500]}"
    d = r.json()
    assert d["total_steps"] > 0, d
    assert len(d["steps"]) > 0, d
    print(f"OK syncode: {d['total_steps']} steps")


def test_consecutive():
    for i in range(2):
        test_generate_raw()
    print("OK consecutive generations")


if __name__ == "__main__":
    try:
        test_health()
        test_generate_raw()
        test_generate_syncode()
        test_consecutive()
        print("\nAll e2e tests passed.")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
