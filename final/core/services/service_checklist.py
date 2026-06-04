"""service_checklist.py — quality gate appended to inverse solver tool results."""
from __future__ import annotations

import re
import yaml
from pathlib import Path

_ROOT      = Path(__file__).parent.parent.parent
_YAML_PATH = _ROOT / "config" / "quality_checklist.yaml"
_MD_PATH   = _ROOT / "config" / "quality_guidelines.md"

# ── Field extractors (matched against tool output strings) ────────────────────

_FIBER_RE   = re.compile(r"(?i)fiber[:\s]+([^\n|]+)")
_POLYMER_RE = re.compile(r"(?i)polymer[:\s]+([^\n|]+)")

_PATTERNS: dict[str, re.Pattern] = {
    "fit_error":        re.compile(r"fit_error\s*=\s*([\d.e+\-]+)"),
    "a11":              re.compile(r"\ba11\s*[=:]\s*([\d.e+\-]+)"),
    "a22":              re.compile(r"\ba22\s*[=:]\s*([\d.e+\-]+)"),
    "ar":               re.compile(r"\baspect_ratio\s*[=:]\s*([\d.e+\-]+)"),
    "fiber_massfrac":   re.compile(r"\bfiber_massfrac\s*[=:]\s*([\d.e+\-]+)"),
    "matrix_modulus":   re.compile(r"\bmatrix_modulus\s*[=:]\s*([\d.e+\-]+)"),
    "matrix_poisson":   re.compile(r"\bmatrix_poisson\s*[=:]\s*([\d.e+\-]+)"),
    "f_cte1_ppmK":      re.compile(r"f_cte1\b.*?([\-\d.]+)\s*ppm"),
    "f_cte2_ppmK":      re.compile(r"f_cte2\b.*?([\-\d.]+)\s*ppm"),
    "m_cte_ppmK":       re.compile(r"m_cte\b.*?([\-\d.]+)\s*ppm"),
    "k_f1":             re.compile(r"k_f1\b.*?=\s*([\d.]+)\s*W"),
    "k_f2":             re.compile(r"k_f2\b.*?=\s*([\d.]+)\s*W"),
    "k_m":              re.compile(r"k_m\s*@\s*25.*?=\s*([\d.]+)\s*W"),
}


def _parse(tool_output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(tool_output)
        if m:
            try:
                values[key] = float(m.group(1))
            except ValueError:
                pass
    return values


# ── Rule loading ──────────────────────────────────────────────────────────────

def _load_rules(tool_name: str, fiber: str, polymer: str) -> dict:
    """Merge global rules with any material-system overrides.

    Material-system keys in the YAML are short partial strings, e.g. "T300_PESU".
    A key matches if every underscore-separated token appears in the fiber or polymer
    name (case-insensitive). The first matching key wins.
    """
    try:
        data = yaml.safe_load(_YAML_PATH.read_text())
    except Exception:
        return {}

    rules = dict(data.get("global", {}).get(tool_name, {}))

    fiber_lower   = fiber.lower()
    polymer_lower = polymer.lower()
    for ms_key, ms_rules in (data.get("material_systems") or {}).items():
        tokens = [t.lower() for t in ms_key.split("_") if t]
        if all(t in fiber_lower or t in polymer_lower for t in tokens):
            rules.update(ms_rules.get(tool_name, {}))
            break

    return rules


# ── Rule evaluation ───────────────────────────────────────────────────────────

def _eval_rules(values: dict, rules: dict) -> list[dict]:
    checks = []
    for rule_key, threshold in rules.items():
        if rule_key.endswith("_message"):
            continue  # handled below alongside its rule
        message = rules.get(f"{rule_key}_message", "")
        if rule_key.endswith("_max"):
            field = rule_key[:-4]
            val = values.get(field)
            if val is None:
                continue
            passed = val <= threshold
            checks.append({
                "field": field, "value": val,
                "rule": f"<= {threshold}",
                "status": "PASS" if passed else "FAIL",
                "message": "" if passed else message,
            })
        elif rule_key.endswith("_min"):
            field = rule_key[:-4]
            val = values.get(field)
            if val is None:
                continue
            passed = val >= threshold
            checks.append({
                "field": field, "value": val,
                "rule": f">= {threshold}",
                "status": "PASS" if passed else "FAIL",
                "message": "" if passed else message,
            })
        elif rule_key.endswith("_range"):
            field = rule_key[:-6]
            lo, hi = threshold
            val = values.get(field)
            if val is None:
                continue
            passed = lo <= val <= hi
            checks.append({
                "field": field, "value": val,
                "rule": f"in [{lo}, {hi}]",
                "status": "PASS" if passed else "FAIL",
                "message": "" if passed else message,
            })
    return checks


# ── Guideline extraction ──────────────────────────────────────────────────────

def _get_guidelines(tool_name: str, fiber: str = "", polymer: str = "") -> str:
    """Extract sections from quality_guidelines.md relevant to this tool and material."""
    try:
        text = _MD_PATH.read_text()
    except Exception:
        return ""

    material_hints = {w.lower() for w in (fiber + " " + polymer).split() if len(w) > 3}

    sections: list[str] = []
    current_lines: list[str] = []
    in_section = False

    for line in text.splitlines():
        if line.startswith("## "):
            if in_section and current_lines:
                sections.append("\n".join(current_lines).strip())
            current_lines = [line]
            header = line.lower()
            matches_tool = tool_name in header
            matches_material = "all systems" in header or any(h in header for h in material_hints)
            in_section = matches_tool and matches_material
        elif in_section:
            current_lines.append(line)

    if in_section and current_lines:
        sections.append("\n".join(current_lines).strip())

    return "\n\n".join(sections)


# ── Public entry point ────────────────────────────────────────────────────────

def _extract_material(tool_output: str) -> tuple[str, str]:
    """Try to extract fiber and polymer names from tool output text."""
    fiber = ""
    polymer = ""
    m = _FIBER_RE.search(tool_output)
    if m:
        fiber = m.group(1).strip()
    m = _POLYMER_RE.search(tool_output)
    if m:
        polymer = m.group(1).strip()
    return fiber, polymer


def evaluate(
    tool_name: str,
    tool_output: str,
    fiber: str = "",
    polymer: str = "",
) -> str:
    """
    Parse tool_output, run hard rules from quality_checklist.yaml, and retrieve
    relevant qualitative guidelines from quality_guidelines.md.

    Returns a formatted report string to append to the tool message.
    """
    if not fiber and not polymer:
        fiber, polymer = _extract_material(tool_output)
    values     = _parse(tool_output)
    rules      = _load_rules(tool_name, fiber, polymer)
    checks     = _eval_rules(values, rules)
    guidelines = _get_guidelines(tool_name, fiber, polymer)

    lines = [f"\n[QUALITY CHECK - {tool_name}]"]

    if checks:
        for c in checks:
            lines.append(
                f"  {c['field']:<22}  {c['value']:<12.4g}  {c['rule']:<22}  {c['status']}"
            )
            if c.get("message"):
                lines.append(f"    ! {c['message']}")
        overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
        lines.append(f"  Overall: {overall}")
    else:
        lines.append("  (no hard rules defined for this tool)")
        overall = "PASS"

    if guidelines:
        lines += ["", "[GUIDELINES]", guidelines]

    return "\n".join(lines)
