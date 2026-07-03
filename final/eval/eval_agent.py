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
    PROMPTS, GUI_INPUTS, SCORED_PROPERTIES, EXPECTED_TOOL,
)

PROMPTS_DIR = _FINAL / "agent" / "prompts"
RESULTS_DIR = _EVAL  / "results"

_MODEL_ALIASES = {
    "14b": "qwen2.5:14b-instruct-q4_K_M",
    "32b": "qwen2.5:32b-instruct-q3_K_M",
}

_CATEGORY_LABEL = {
    "E": "ELASTIC",
    "T": "THERMOELASTIC",
    "K": "THERMAL",
    "S": "SWEEP",
}


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(model_tag: str):
    llm    = ChatOllama(model=model_tag, temperature=0, streaming=False)
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

def score_csv(csv_path: Path) -> None:
    """
    Read a filled CSV and print MAPE report.

    Skips rows where gui_value is empty (not yet filled in) or property is "—".
    Uses relative error: |agent - gui| / |gui|  × 100 %.
    For near-zero values (|gui| < 1e-10) falls back to absolute error.
    """
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Split into scoreable vs skipped
    scoreable = [
        r for r in rows
        if r["gui_value"].strip() and r["property"] != "—"
    ]
    skipped = [r for r in rows if not r["gui_value"].strip() or r["property"] == "—"]

    if not scoreable:
        print("No rows have gui_value filled in yet. Fill the gui_value column and re-run.")
        return

    # Per-row error
    errors_by_key: dict[str, list[float]] = {}
    problem_rows = []

    for r in scoreable:
        try:
            agent = float(r["agent_value"])
            gui   = float(r["gui_value"])
        except ValueError:
            problem_rows.append(r)
            continue

        denom = abs(gui) if abs(gui) > 1e-10 else 1.0
        pct   = abs(agent - gui) / denom * 100
        errors_by_key.setdefault(r["key"], []).append(pct)

    # ── Per-prompt table ──────────────────────────────────────────────────────
    print(f"\n{'Key':<5}  {'Tool':<6}  {'Properties':>12}  {'MAPE':>8}  {'Max err':>9}  Notes")
    print("-" * 65)

    prev_cat = None
    all_mapes: list[float] = []

    for key in PROMPTS:
        if key not in errors_by_key and key not in [r["key"] for r in skipped]:
            continue

        cat = key[0]
        if cat != prev_cat:
            print(f"\n  ── {_CATEGORY_LABEL.get(cat, cat)} ──")
            prev_cat = cat

        errs = errors_by_key.get(key, [])
        # Tool routing from first row for this key
        key_rows = [r for r in rows if r["key"] == key]
        tool_ok  = key_rows[0]["tool_correct"] if key_rows else "?"
        tool_str = "✓" if tool_ok == "True" else "✗"

        if not errs:
            props_str = f"0/{len(SCORED_PROPERTIES.get(key, []))}"
            print(f"{key:<5}  {tool_str:<6}  {props_str:>12}  {'—':>8}  {'—':>9}  "
                  f"not filled / sweep")
            continue

        mape    = sum(errs) / len(errs)
        max_err = max(errs)
        filled  = f"{len(errs)}/{len(SCORED_PROPERTIES.get(key,[]))}"
        all_mapes.append(mape)

        flag = ""
        if mape > 10:
            flag = " ← HIGH"
        elif mape > 2:
            flag = " ← check"

        print(f"{key:<5}  {tool_str:<6}  {filled:>12}  {mape:>7.2f}%  {max_err:>8.2f}%{flag}")

    # ── Per-category summary ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    cat_keys = {"E": [], "T": [], "K": [], "S": []}
    for k, errs in errors_by_key.items():
        cat_keys[k[0]].append(sum(errs) / len(errs))

    for cat, mapes in cat_keys.items():
        if mapes:
            print(f"  {_CATEGORY_LABEL[cat]:<18}  avg MAPE = {sum(mapes)/len(mapes):.2f}%")

    if all_mapes:
        overall = sum(all_mapes) / len(all_mapes)
        print(f"  {'OVERALL':<18}  avg MAPE = {overall:.2f}%")

    tool_correct_rows = [r for r in rows if r["tool_correct"] == "True"]
    tool_acc = len(tool_correct_rows) / len(rows) * 100 if rows else 0
    print(f"\n  Tool routing accuracy : {tool_acc:.0f}%  ({len(tool_correct_rows)}/{len(rows)} prompts)")

    if skipped:
        print(f"  Skipped (no gui_value): {len(skipped)} rows")
    if problem_rows:
        print(f"  Parse errors          : {len(problem_rows)} rows")
    print(f"{'='*65}\n")


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
        for field in ("fiber", "matrix", "micro", "CTE", "k", "density",
                      "score_at", "note"):
            val = info.get(field)
            if val:
                label = {
                    "fiber": "Fiber   ", "matrix": "Matrix  ",
                    "micro": "Micro   ", "CTE":    "CTE     ",
                    "k":     "k inputs", "density":"Density ",
                    "score_at": "Score @ ", "note":  "Note    ",
                }[field]
                print(f"    {label}: {val}")

    print("\n" + "═" * 65)
    print("  Fill eval/results/*.csv → gui_value column, then run --score")
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
