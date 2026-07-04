#!/usr/bin/env python3
"""
eval_agent.py — Two-phase benchmark for the MateriAl agent.

Phase 1 — run the agent, capture its predictions:
    python eval/eval_agent.py --run
    python eval/eval_agent.py --run --model 14b --keys E1 T4 S1

    Saves:  eval/results/eval_<model>_<timestamp>.csv
    Prints: GUI worksheet — exact inputs to enter in gui.py for each prompt

Phase 2 — score after you fill in the gui_value column:
    python eval/eval_agent.py --score eval/results/eval_14b_20260628_1430.csv

    Prints: MAPE per prompt, per category, overall
"""

import argparse
import csv
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# ── Path + JAX env ────────────────────────────────────────────────────────────
_EVAL   = Path(__file__).parent
_FINAL  = _EVAL.parent
if str(_FINAL) not in sys.path:
    sys.path.insert(0, str(_FINAL))

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64",    "1")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from agent.graph import build_app

from eval.prompts import (
    PROMPTS, GUI_INPUTS, SCORED_PROPERTIES, EXPECTED_TOOL, PROMPT_GOALS,
)

PROMPTS_DIR = _FINAL / "agent" / "prompts"
RESULTS_DIR = _EVAL  / "results"

_MODEL_ALIASES = {
    "14b":   "qwen2.5:14b-instruct-q4_K_M",
    "32b":   "qwen2.5:32b-instruct-q3_K_M",
    "8b":    "qwen3:8b",
    "q3-14": "qwen3:14b",
}

_CATEGORY_LABEL = {
    "E": "ELASTIC",
    "T": "THERMOELASTIC",
    "K": "THERMAL",
    "S": "SWEEP",
}


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(model_tag: str):
    think = False if model_tag.startswith("qwen3") else None
    kwargs = {"think": think} if think is not None else {}
    llm    = ChatOllama(model=model_tag, temperature=0, streaming=False, **kwargs)
    system = (PROMPTS_DIR / "system_prompt.md").read_text()
    vocab  = (PROMPTS_DIR / "vocabulary.md").read_text()
    return build_app(llm, f"{system}\n\n---\n\n{vocab}")


# ── Output parser ─────────────────────────────────────────────────────────────

def parse_tool_output(tool_text: str, key: str) -> dict[str, float]:
    """
    Extract predicted numerical values from a tool's return string.

    Handles three formats produced by the tools:
      Elastic/thermoelastic  — "  E1     =   23332.1 MPa"  /  "  CTE11 = -2.5e-07 /K"
      Thermal single-temp    — "  k11 (print direction) = 1.234 W/m·K"
      Thermal matrix         — table rows: "  25.0  0.278  1.234  0.456  0.456"
    """
    values: dict[str, float] = {}
    cat = key[0]

    if cat in ("E", "T"):
        # Elastic outputs: E1, E2, E3, G12, G13, G23, nu12, nu13, nu23
        for m in re.finditer(
            r'^\s+(E[123]|G(?:12|13|23)|nu(?:12|13|23))\s*=\s*([-\d.e+]+)',
            tool_text, re.MULTILINE,
        ):
            values[m.group(1)] = float(m.group(2))

        # CTE outputs: CTE11, CTE22, CTE33
        for m in re.finditer(
            r'^\s+(CTE\d{2})\s*=\s*([-\d.e+]+)',
            tool_text, re.MULTILINE,
        ):
            values[m.group(1)] = float(m.group(2))

    elif cat == "K":
        # Thermal single-temperature format:
        # "    k11 (print direction)       = 1.23456 W/m·K"
        # First try to find any temperature context lines "PREDICTION AT T = XX.X°C"
        temps_seen: list[float] = []
        current_T: float | None = None

        for line in tool_text.splitlines():
            t_match = re.search(r'PREDICTION AT T\s*=\s*([\d.]+)', line)
            if t_match:
                current_T = float(t_match.group(1))
                temps_seen.append(current_T)

            k_match = re.match(
                r'\s+(k\d{2})\s*\([^)]+\)\s*=\s*([\d.e+\-]+)', line
            )
            if k_match and current_T is not None:
                prop_key = f"{k_match.group(1)}_{current_T:.0f}C"
                values[prop_key] = float(k_match.group(2))

        # Temperature matrix format (if above found nothing):
        # "  T (°C)  k_m  k11  k22  k33"  then data rows
        if not values:
            in_table = False
            for line in tool_text.splitlines():
                if "T (°C)" in line and "k11" in line:
                    in_table = True
                    continue
                if not in_table:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        T   = float(parts[0])
                        k11 = float(parts[2])
                        k22 = float(parts[3])
                        k33 = float(parts[4]) if len(parts) > 4 else None
                        values[f"k11_{T:.0f}C"] = k11
                        values[f"k22_{T:.0f}C"] = k22
                        if k33 is not None:
                            values[f"k33_{T:.0f}C"] = k33
                    except (ValueError, IndexError):
                        continue

    return values


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(app, key: str) -> dict:
    """Run one prompt in a fresh thread; return tool calls, parsed values, timing."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0     = time.perf_counter()
    error  = None
    all_messages: list = []

    try:
        result       = app.invoke(
            {"messages": [HumanMessage(content=PROMPTS[key])]},
            config,
        )
        all_messages = result.get("messages", [])
    except Exception as exc:
        error = str(exc)

    elapsed = round(time.perf_counter() - t0, 2)

    # Extract all tool calls (name + args)
    tool_calls = []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc.get("args", {})})

    # Extract tool result text(s) — each ToolMessage content
    tool_outputs = [
        {"tool": m.name, "content": m.content}
        for m in all_messages
        if isinstance(m, ToolMessage)
    ]

    # Parse predicted values from tool outputs that match expected tool
    predicted: dict[str, float] = {}
    expected = EXPECTED_TOOL[key]
    for out in tool_outputs:
        if out["tool"] == expected:
            predicted.update(parse_tool_output(out["content"], key))

    # First tool called by the agent
    first_tool = tool_calls[0]["tool"] if tool_calls else None

    return {
        "key":          key,
        "first_tool":   first_tool,
        "expected_tool":expected,
        "tool_correct": first_tool == expected,
        "n_calls":      len(tool_calls),
        "predicted":    predicted,
        "elapsed_s":    elapsed,
        "error":        error,
    }


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(run_results: list[dict], model_tag: str) -> Path:
    """
    Write results to CSV.  gui_value column is empty — fill it from the GUI.

    Columns:
        key          prompt key (E1, T4, K1 …)
        property     output field (E1, CTE11, k11_25C …)
        agent_value  what the agent predicted (blank if tool failed / not applicable)
        gui_value    YOU FILL THIS IN from gui.py
        tool_correct did the agent call the right tool? (True/False)
        n_calls      how many tool calls were made
        elapsed_s    response time in seconds
        notes        warnings, errors, or "sweep — tool routing only"
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    safe_tag = model_tag.replace(":", "_").replace("/", "_")
    path     = RESULTS_DIR / f"eval_{safe_tag}_{ts}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "property", "agent_value", "gui_value",
                        "tool_correct", "n_calls", "elapsed_s", "notes"],
        )
        writer.writeheader()

        for r in run_results:
            key        = r["key"]
            props      = SCORED_PROPERTIES[key]
            tool_ok    = r["tool_correct"]
            n_calls    = r["n_calls"]
            elapsed    = r["elapsed_s"]
            err        = r["error"] or ""
            predicted  = r["predicted"]

            if not props:
                # Sweeps — one summary row
                notes = "sweep — tool routing only"
                if err:
                    notes += f" | ERROR: {err}"
                writer.writerow({
                    "key":          key,
                    "property":     "—",
                    "agent_value":  "",
                    "gui_value":    "",
                    "tool_correct": tool_ok,
                    "n_calls":      n_calls,
                    "elapsed_s":    elapsed,
                    "notes":        notes,
                })
                continue

            for prop in props:
                agent_val = predicted.get(prop, "")
                notes     = ""
                if err:
                    notes = f"ERROR: {err}"
                elif not tool_ok:
                    notes = f"wrong tool: {r['first_tool']}"
                elif prop not in predicted:
                    notes = "not found in output"

                writer.writerow({
                    "key":          key,
                    "property":     prop,
                    "agent_value":  agent_val,
                    "gui_value":    "",          # ← you fill this in
                    "tool_correct": tool_ok,
                    "n_calls":      n_calls,
                    "elapsed_s":    elapsed,
                    "notes":        notes,
                })

    return path


# ── Scorer ────────────────────────────────────────────────────────────────────

_PASS_THRESHOLD = 1.5  # % — MAPE below this = pass (any real input error exceeds this)


def _short_challenge(key: str) -> str:
    """One short phrase describing what a prompt tests, for the results table."""
    goal = PROMPT_GOALS.get(key, "")
    # The goals are formatted "Label: description — detail."
    # Extract just the description part (after colon, before " — ")
    if ":" in goal:
        goal = goal.split(":", 1)[1].strip()
    if " — " in goal:
        goal = goal.split(" — ")[0].strip()
    return goal[:40]


def score_csv(csv_path: Path) -> None:
    """
    Read a filled CSV and print a pass/fail results table.

    Pass criteria (per prompt):
      - tool_correct == True
      - MAPE across all scored properties < _PASS_THRESHOLD %

    Any prompt above the threshold is listed in a failure-detail section
    with per-property breakdown so the reader knows exactly what went wrong.
    """
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Empty CSV.")
        return

    # ── Compute per-prompt result ─────────────────────────────────────────────
    prompt_results: dict[str, dict] = {}

    for key in PROMPTS:
        key_rows = [r for r in rows if r["key"] == key]
        if not key_rows:
            continue

        tool_ok     = key_rows[0]["tool_correct"].strip().lower() == "true"
        scored_rows = [
            r for r in key_rows
            if r.get("gui_value", "").strip() and r["property"] != "—"
        ]

        if not scored_rows:
            prompt_results[key] = {
                "tool_ok": tool_ok, "scored": False,
                "passed": False, "mape": None,
                "prop_errors": [], "n_filled": 0,
            }
            continue

        prop_errors: list[tuple[str, float]] = []
        for r in scored_rows:
            try:
                agent = float(r["agent_value"])
                gui   = float(r["gui_value"])
            except ValueError:
                continue
            denom = abs(gui) if abs(gui) > 1e-10 else 1.0
            prop_errors.append((r["property"], abs(agent - gui) / denom * 100))

        if not prop_errors:
            prompt_results[key] = {
                "tool_ok": tool_ok, "scored": False,
                "passed": False, "mape": None,
                "prop_errors": [], "n_filled": 0,
            }
            continue

        mape   = sum(e for _, e in prop_errors) / len(prop_errors)
        passed = tool_ok and mape < _PASS_THRESHOLD

        prompt_results[key] = {
            "tool_ok":     tool_ok,
            "scored":      True,
            "passed":      passed,
            "mape":        mape,
            "prop_errors": sorted(prop_errors, key=lambda x: -x[1]),
            "n_filled":    len(prop_errors),
            "n_expected":  len(SCORED_PROPERTIES.get(key, [])),
        }

    # ── Main results table ────────────────────────────────────────────────────
    W = 76
    print(f"\n{'═' * W}")
    print(f"  MateriAl Agent — Benchmark Results  (pass threshold: {_PASS_THRESHOLD}%)")
    print(f"{'═' * W}")
    print(f"\n  {'Key':<5}  {'Challenge':<40}  {'Tool':<5}  {'Pass':<5}  Notes")
    print(f"  {'─'*5}  {'─'*40}  {'─'*5}  {'─'*5}  {'─'*16}")

    prev_cat   = None
    n_pass     = 0
    n_total    = 0
    hard_cases: list[tuple[str, dict]] = []

    for key in PROMPTS:
        if key not in prompt_results:
            continue

        cat = key[0]
        if cat != prev_cat:
            print(f"\n  ── {_CATEGORY_LABEL.get(cat, cat)} ──")
            prev_cat = cat

        res       = prompt_results[key]
        tool_str  = "✓" if res["tool_ok"] else "✗"
        challenge = _short_challenge(key)

        if not res["scored"]:
            print(f"  {key:<5}  {challenge:<40}  {tool_str:<5}  {'—':<5}  not filled")
            continue

        n_total += 1
        if res["passed"]:
            n_pass += 1
            pass_str = "✓"
            notes    = ""
        else:
            pass_str = "✗"
            if not res["tool_ok"]:
                notes = "wrong tool"
            elif res["prop_errors"]:
                worst_prop, worst_err = res["prop_errors"][0]
                notes = f"{worst_prop} off {worst_err:.1f}%"
            else:
                notes = f"MAPE {res['mape']:.1f}%"
            hard_cases.append((key, res))

        filled_str = f"{res['n_filled']}/{res['n_expected']}"
        print(f"  {key:<5}  {challenge:<40}  {tool_str:<5}  {pass_str:<5}  "
              f"{notes}  [{filled_str} props]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    n_prompts  = len(prompt_results)
    tool_ok_n  = sum(1 for r in prompt_results.values() if r.get("tool_ok"))
    tool_pct   = tool_ok_n / n_prompts * 100 if n_prompts else 0
    pass_pct   = n_pass / n_total * 100 if n_total else 0

    print(f"  Tool routing : {tool_ok_n}/{n_prompts}  ({tool_pct:.0f}%)")
    print(f"  Pass rate    : {n_pass}/{n_total}  ({pass_pct:.0f}%)"
          + ("  — all filled prompts passed" if n_pass == n_total else ""))

    # ── Failed prompt detail ──────────────────────────────────────────────────
    if hard_cases:
        print(f"\n  ── Failed prompts ──────────────────────────────────────────")
        for key, res in hard_cases:
            goal = PROMPT_GOALS.get(key, "")
            print(f"\n  {key}  {goal}")
            if not res["tool_ok"]:
                print(f"    → wrong tool called")
            if res["prop_errors"]:
                print(f"    → MAPE {res['mape']:.2f}%")
                for prop, err in res["prop_errors"]:
                    flag = "  ←" if err >= _PASS_THRESHOLD else ""
                    print(f"       {prop:<10}  {err:>7.3f}%{flag}")

    print(f"{'═' * W}\n")


# ── GUI worksheet printer ─────────────────────────────────────────────────────

def print_worksheet(keys: list[str]) -> None:
    """Print the inputs to enter manually in gui.py for each prompt."""
    print("\n" + "═" * 65)
    print("  GUI WORKSHEET — enter these values in gui.py to get ground truth")
    print("═" * 65)

    prev_cat = None
    for key in keys:
        if key not in GUI_INPUTS:
            continue
        cat = key[0]
        if cat != prev_cat:
            print(f"\n── {_CATEGORY_LABEL.get(cat, cat)} ─────────────────────────────")
            prev_cat = cat

        info = GUI_INPUTS[key]
        print(f"\n  {key}")
        for field in ("goal", "fiber", "matrix", "micro", "CTE", "k", "density",
                      "score_at", "note"):
            val = info.get(field)
            if val:
                label = {
                    "goal":  "Goal    ", "fiber": "Fiber   ", "matrix": "Matrix  ",
                    "micro": "Micro   ", "CTE":   "CTE     ",
                    "k":     "k inputs", "density":"Density ",
                    "score_at": "Score @ ", "note": "Note    ",
                }[field]
                print(f"    {label}: {val}")

    print("\n" + "═" * 65)
    print("  Fill eval/results/*.csv → gui_value column, then:")
    print("  python eval/eval_agent.py --score <csv>")
    print("  Pass = tool correct + MAPE < 1% vs gui_value")
    print("═" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase MateriAl agent benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/eval_agent.py --run\n"
            "  python eval/eval_agent.py --run --model 14b --keys E1 T4\n"
            "  python eval/eval_agent.py --score eval/results/eval_14b_20260628_1430.csv\n"
            "  python eval/eval_agent.py --worksheet\n"
        ),
    )
    p.add_argument("--run",       action="store_true", help="Run agent on benchmark prompts")
    p.add_argument("--score",     metavar="CSV",       help="Score a filled CSV file")
    p.add_argument("--worksheet", action="store_true", help="Print GUI worksheet only (no agent run)")
    p.add_argument("--model",     default="14b",       help="Model alias: 14b (default) or 32b")
    p.add_argument("--keys",      nargs="+",           help="Subset of prompt keys, e.g. E1 T4 S1")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    model_tag = _MODEL_ALIASES.get(args.model, args.model)
    keys      = args.keys or list(PROMPTS.keys())
    keys      = [k for k in keys if k in PROMPTS]

    # ── Worksheet only ────────────────────────────────────────────────────────
    if args.worksheet:
        print_worksheet(keys)
        return

    # ── Score mode ────────────────────────────────────────────────────────────
    if args.score:
        csv_path = Path(args.score)
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            sys.exit(1)
        score_csv(csv_path)
        return

    # ── Run mode ──────────────────────────────────────────────────────────────
    if not args.run:
        print("Specify --run, --score <csv>, or --worksheet")
        sys.exit(1)

    unknown = [k for k in (args.keys or []) if k not in PROMPTS]
    if unknown:
        print(f"Unknown prompt keys: {unknown}")

    print(f"\nMateriAl Agent Benchmark — Run Phase")
    print(f"  Model   : {model_tag}")
    print(f"  Prompts : {keys}\n")

    # Pre-warm JAX JIT for all three surrogate models so the first thermal
    # prompt (K1) doesn't pay the 30-60s compilation cost mid-benchmark.
    print("  Pre-warming JAX models ...", end="", flush=True)
    import core.services.service_forward as _sfw
    _dummy = {
        "e1": 230000.0, "e2": 15000.0, "g12": 15000.0,
        "f_nu12": 0.2, "f_nu23": 0.25,
        "ar": 20.0, "fiber_massfrac": 0.20,
        "fiber_density": 1760.0, "matrix_modulus": 3100.0,
        "matrix_poisson": 0.37, "matrix_density": 1280.0,
        "a11": 0.60, "a22": 0.15, "a12": 0.0, "a13": 0.0, "a23": 0.0,
        "f_cte1": -0.5e-6, "f_cte2": 15.0e-6, "m_cte": 60.0e-6,
    }
    _thermal_dummy = {
        "k_f1": 8.0, "k_f2": 1.0, "k_m": 0.2,
        "ar_f": 20.0, "w_f": 0.20, "rho_f": 1760.0, "rho_m": 1280.0,
        "a11": 0.60, "a22": 0.15, "a12": 0.0, "a13": 0.0, "a23": 0.0,
    }
    try:
        _sfw.run_forward("elastic",      _dummy)
        _sfw.run_forward("thermoelastic", _dummy)
        _sfw.run_forward("thermal",       _thermal_dummy)
        print(" done")
    except Exception as exc:
        print(f" skipped ({exc})")

    app          = build_agent(model_tag)
    run_results  = []

    for i, key in enumerate(keys, 1):
        print(f"  [{i:>2}/{len(keys)}] {key} ... ", end="", flush=True)
        result = run_one(app, key)
        run_results.append(result)

        if result["error"]:
            status = "ERROR"
        elif not result["tool_correct"]:
            status = f"✗ wrong tool ({result['first_tool']})"
        elif result["predicted"]:
            n = len(result["predicted"])
            status = f"✓  ({n} values parsed, {result['elapsed_s']:.1f}s)"
        else:
            status = f"✓  tool called but no values parsed  ({result['elapsed_s']:.1f}s)"

        print(status)

    # Save CSV
    csv_path = write_csv(run_results, model_tag)
    print(f"\n  Results saved → {csv_path}")

    # Print quick tool-routing summary
    n_correct = sum(1 for r in run_results if r["tool_correct"])
    n_retry   = sum(1 for r in run_results if r["n_calls"] > 1)
    print(f"  Tool routing : {n_correct}/{len(run_results)} correct")
    print(f"  Retries      : {n_retry}/{len(run_results)} prompts needed >1 call")

    # Print worksheet so user knows what to enter in GUI
    print_worksheet(keys)

    print(f"  Next: fill gui_value column in {csv_path.name}")
    print(f"  Then: python eval/eval_agent.py --score {csv_path}\n")


if __name__ == "__main__":
    main()
