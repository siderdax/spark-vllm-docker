#!/usr/bin/env python3
"""
context-bench.py - Needle-in-a-haystack context-length benchmark

Verifies that a running vLLM OpenAI-compatible server can actually serve
requests near its configured --max-model-len, and correctly recall a fact
planted at various depths in the prompt (not just avoid crashing/OOMing).

Usage:
    ./context-bench.py [--sizes 120000,300000,490000] [--depths 0.1,0.5,0.9]
    ./context-bench.py --base-url http://localhost:8000/v1 --model deepseek-ai/DeepSeek-V4-Flash-0731
    API_KEY=sk-... ./context-bench.py --sizes 490000 --depths 0.5

Sizes are target *token* counts (approximate; actual prompt_tokens from the
server is what gets recorded). The needle is a random 6-digit number inserted
into filler text at the given fractional depth (0.0 = start, 1.0 = end); the
model is asked to recall it with thinking disabled for a clean, fast signal.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent

FILLER_SENTENCES = [
    "The city council debated the new zoning proposal for most of the afternoon.",
    "Researchers observed a gradual shift in migratory patterns over the decade.",
    "The recipe calls for two cups of flour and a pinch of sea salt.",
    "Quarterly revenue exceeded expectations despite supply chain disruptions.",
    "The old lighthouse has guided ships along this rocky coast since 1887.",
    "Engineers ran a battery of stress tests on the new suspension design.",
    "The orchestra rehearsed the third movement twice before the intermission.",
    "Local farmers reported an unusually dry growing season this year.",
    "The museum's new wing features rotating exhibits on regional pottery.",
    "A minor software update resolved the intermittent connectivity issue.",
]


def resolve_api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    key_file = SCRIPT_DIR / "api_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


def resolve_model(base_url: str, headers: dict, explicit: str | None) -> str:
    if explicit:
        return explicit
    r = requests.get(f"{base_url}/models", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    if not data:
        raise SystemExit("No model reported by /v1/models; pass --model explicitly.")
    return data[0]["id"]


def build_haystack(target_chars: int, needle: str, depth: float) -> str:
    rng = random.Random(1234)
    sentences = []
    total = 0
    while total < target_chars:
        s = rng.choice(FILLER_SENTENCES)
        sentences.append(s)
        total += len(s) + 1
    insert_at = int(len(sentences) * depth)
    sentences.insert(insert_at, needle)
    return " ".join(sentences)


def count_tokens(base_url: str, headers: dict, model: str, text: str) -> int:
    """Cheap prompt-token probe: max_tokens=1 so we pay for prefill only."""
    payload = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1}
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


def calibrate(base_url: str, headers: dict, model: str, sample_chars: int = 20000) -> float:
    sample = build_haystack(sample_chars, "The sky is blue today.", 0.5)
    tokens = count_tokens(base_url, headers, model, sample)
    ratio = len(sample) / tokens
    print(f"[calibrate] {len(sample)} chars -> {tokens} tokens ({ratio:.3f} chars/tok)")
    return ratio


def run_case(base_url, headers, model, thinking, target_tokens, depth, chars_per_token, secret):
    needle = f"The special magic number for this test is {secret}. Remember this number."
    target_chars = int(target_tokens * chars_per_token)
    haystack = build_haystack(target_chars, needle, depth)
    question = (
        "\n\nWhat is the special magic number mentioned in the text above? "
        "Reply with ONLY the number, nothing else."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": haystack + question}],
        "max_tokens": 1500,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": thinking},
    }
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=1800)
        elapsed = time.time() - t0
        r.raise_for_status()
        body = r.json()
        usage = body["usage"]
        answer = body["choices"][0]["message"]["content"]
        found = str(secret) in answer
        return {
            "target_tokens": target_tokens,
            "depth": depth,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage.get("completion_tokens"),
            "elapsed_s": round(elapsed, 1),
            "secret": secret,
            "answer": answer.strip()[:200],
            "found": found,
            "status": "ok",
        }
    except Exception as e:
        return {
            "target_tokens": target_tokens,
            "depth": depth,
            "elapsed_s": round(time.time() - t0, 1),
            "secret": secret,
            "status": "error",
            "error": str(e)[:500],
        }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default=None, help="Defaults to whatever /v1/models reports.")
    p.add_argument("--api-key", default=None, help="Defaults to $API_KEY, then ./api_key.txt.")
    p.add_argument("--sizes", default="120000,300000,490000", help="Comma-separated target token counts.")
    p.add_argument("--depths", default="0.1,0.5,0.9", help="Comma-separated needle depths (0.0-1.0).")
    p.add_argument("--thinking", action="store_true", help="Leave reasoning/thinking mode on (slower).")
    p.add_argument("--output", default=None, help="Where to write JSON results (default: alongside this script).")
    args = p.parse_args()

    api_key = resolve_api_key(args.api_key)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model = resolve_model(args.base_url, headers, args.model)
    print(f"[model] {model}")

    ratio = calibrate(args.base_url, headers, model)

    sizes = [int(x) for x in args.sizes.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    results = []
    rng = random.Random(42)
    for tt in sizes:
        for d in depths:
            secret = rng.randint(100000, 999999)
            print(f"\n[run] target_tokens={tt} depth={d} secret={secret}")
            res = run_case(args.base_url, headers, model, args.thinking, tt, d, ratio, secret)
            print(f"[result] {json.dumps(res)}")
            results.append(res)

    print("\n=== SUMMARY ===")
    for r in results:
        if r["status"] == "ok":
            mark = "PASS" if r["found"] else "FAIL"
            print(f"{mark} tokens~{r['target_tokens']:>7} (actual {r['prompt_tokens']:>7}) depth={r['depth']:.1f} "
                  f"time={r['elapsed_s']:>6.1f}s answer={r['answer']!r}")
        else:
            print(f"ERROR tokens~{r['target_tokens']:>7} depth={r['depth']:.1f} time={r['elapsed_s']:>6.1f}s "
                  f"err={r['error']}")

    out_path = Path(args.output) if args.output else SCRIPT_DIR / "context-bench-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out_path}")

    if any(r["status"] != "ok" or not r.get("found") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
