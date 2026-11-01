#!/usr/bin/env python3
"""Automatic pre-checks for writing audit.

Usage: python3 audit.py <file.md> [--json]

Runs stdlib only checks:
- exact duplicate sentences
- near duplicate sentences (difflib > 0.85)
- duplicate paragraph openings
- terminology flags from terminology-db.json (structured: regex, severity, suggestion)
- passive voice heuristics
- leading questions
- long sentences
- banned subtitles (Why it matters)
- self reference, totalizers, vague time, trivia IDs
- repeated time claims
- first person pronouns (docs voice)
- standout boxes and admonitions

Outputs a checklist report (text) or a structured report (--json).
Exit code 1 when any severity=error finding exists, else 0.
Holistic assessment stays with the agent.
"""
import json
import re
import sys
import difflib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PASSIVE_RE = re.compile(
    r"\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b"
    r"|\b(has|have|had)\s+been\s+\w+ed\b"
    r"|\b(am|is|are|was|were|be|been|being)\b\s+"
    r"(written|built|done|taken|given|grown|known|shown|drawn|thrown|broken|"
    r"chosen|frozen|spoken|stolen|worn|torn|born|beaten|made|paid|said|heard)\b",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(r"^\s*(#{1,6}\s+)?(Want|Need|Tired|Curious|Did you know|What if|Why)[^?.!]*\?\s*$", re.IGNORECASE)
SELF_REF_RE = re.compile(r"you are reading|this page you|as you can see here", re.IGNORECASE)
TOTALIZER_RE = re.compile(r"\beverything here\b|\ball of this\b|\beverything below\b", re.IGNORECASE)
TIME_CLAIM_RE = re.compile(r"\btwo days\b|\b48 hours\b|\bunder two days\b|\bin two days\b", re.IGNORECASE)
VAGUE_TIME_RE = re.compile(r"\bby evening\b|\blater that day\b", re.IGNORECASE)
FIRST_PERSON_RE = re.compile(r"\b(we|our|us|ourselves)\b", re.IGNORECASE)
BOX_RE = re.compile(r"^>\s*\[!|^\s*:::+|<div[^>]*class=\"[^\"]*(box|callout|admonition)", re.IGNORECASE)

DB_PATH = Path(__file__).parent.parent / "terminology-db.json"
DB_MD_FALLBACK = Path(__file__).parent.parent / "terminology-db.md"


def load_rules():
    """Load structured terminology rules. JSON is canonical; md is fallback."""
    if DB_PATH.exists():
        data = json.loads(DB_PATH.read_text())
        rules = []
        for r in data.get("rules", []):
            flags = 0 if r.get("case_sensitive") else re.IGNORECASE
            pattern = r["term"] if r.get("regex") else re.escape(r["term"])
            # multiline for heading-anchored patterns
            if r["term"].startswith("^"):
                flags |= re.MULTILINE
            try:
                compiled = re.compile(pattern, flags)
            except re.error:
                continue
            rules.append({**r, "compiled": compiled})
        return rules
    # fallback: flat md list, warning severity
    rules = []
    if DB_MD_FALLBACK.exists():
        for i, line in enumerate(DB_MD_FALLBACK.read_text().splitlines()):
            line = line.strip()
            if line.startswith("- "):
                core = re.split(r"\s*\(", line[2:].strip())[0].strip()
                if core and len(core) > 1:
                    rules.append({
                        "id": f"MD_{i}", "category": "legacy", "term": core,
                        "regex": False, "case_sensitive": False,
                        "severity": "warning", "suggestion": "",
                        "message": "Terminology flag.",
                        "compiled": re.compile(re.escape(core), re.IGNORECASE),
                    })
    return rules


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def line_of(offset, line_starts):
    import bisect
    return bisect.bisect_right(line_starts, offset)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_json = "--json" in sys.argv[1:]
    if not args:
        print("Usage: python3 audit.py <file.md> [--json]")
        sys.exit(2)
    path = Path(args[0])
    text = path.read_text()
    rules = load_rules()

    line_starts = [0]
    for m in re.finditer(r"\n", text):
        line_starts.append(m.end())

    paragraphs = split_paragraphs(text)
    sentences = []  # (para_idx, sentence, offset)
    for pi, p in enumerate(paragraphs):
        base = text.find(p)
        for s in split_sentences(p):
            off = text.find(s, base)
            sentences.append((pi, s, off))

    findings = []

    def add(check, severity, para, offset, matched, message="", suggestion="", rule_id=None):
        findings.append({
            "check": check,
            "rule_id": rule_id,
            "severity": severity,
            "file": path.name,
            "line": line_of(offset, line_starts) if offset is not None else None,
            "para": (para + 1) if para is not None else None,
            "matched": matched[:160],
            "message": message,
            "suggestion": suggestion,
        })

    # 1. Exact duplicate sentences
    norm = {}
    for pi, s, off in sentences:
        key = re.sub(r"\s+", " ", s.lower().strip())
        norm.setdefault(key, []).append((pi, s, off))
    for key, vs in norm.items():
        if len(vs) > 1 and len(key) > 20:
            paras = sorted(set(pi + 1 for pi, _, _ in vs))
            add("duplicate-sentence", "warning", vs[0][0], vs[0][2],
                f"{len(vs)}x (paras {paras}): {vs[0][1][:100]}",
                "Same sentence repeated. Say it once.")

    # 2. Near duplicates
    seen = set()
    shown = 0
    keys = list(norm.keys())
    for i in range(len(keys)):
        if len(keys[i]) < 20:
            continue
        for j in range(i + 1, len(keys)):
            if len(keys[j]) < 20:
                continue
            r = difflib.SequenceMatcher(None, keys[i], keys[j]).ratio()
            if r > 0.85 and (keys[i], keys[j]) not in seen:
                seen.add((keys[i], keys[j]))
                vs = norm[keys[i]]
                add("near-duplicate", "warning", vs[0][0], vs[0][2],
                    f"{r:.2f}: \"{keys[i][:80]}\" vs \"{keys[j][:80]}\"",
                    "Near-duplicate sentences. Merge or cut one.")
                shown += 1
                if shown >= 10:
                    break
        if shown >= 10:
            break

    # 3. Repeated paragraph openings
    openings = Counter()
    first_off = {}
    for pi, p in enumerate(paragraphs):
        first = " ".join(p.split()[:6]).lower()
        openings[first] += 1
        first_off.setdefault(first, text.find(p))
    for k, c in openings.items():
        if c > 1:
            add("repeated-opening", "warning", None, first_off[k],
                f"{c}x: {k}...",
                "Paragraphs open the same way. Vary or merge.")

    # 4. Terminology flags (structured DB; banned subtitles handled in section 8)
    shown = 0
    for pi, s, off in sentences:
        for r in rules:
            if r["category"] == "banned subtitle":
                continue
            m = r["compiled"].search(s)
            if m:
                add("terminology", r["severity"], pi, off,
                    m.group()[:80], r["message"], r["suggestion"], r["id"])
                shown += 1
                if shown >= 40:
                    break
        if shown >= 40:
            break

    # 5. Passive voice heuristic
    shown = 0
    for pi, s, off in sentences:
        if PASSIVE_RE.search(s):
            add("passive-voice", "warning", pi, off, s[:120],
                "Possible passive. Rewrite active with a named actor. Confirm with PassivePy.")
            shown += 1
            if shown >= 15:
                break

    # 6. Leading questions
    for pi, s, off in sentences:
        if s.strip().endswith("?"):
            para = paragraphs[pi]
            if para.lstrip().startswith(s.strip()[:20]) or QUESTION_RE.search(s):
                add("leading-question", "warning", pi, off, s[:120],
                    "Leading/rhetorical question. Replace with a plain statement.")

    # 7. Long sentences
    shown = 0
    for pi, s, off in sentences:
        if len(s.split()) > 28:
            add("long-sentence", "warning", pi, off,
                f"({len(s.split())} words): {s[:100]}...",
                "Over 28 words. Split or cut.")
            shown += 1
            if shown >= 10:
                break

    # 8. Structural checks over raw lines
    for ln, line in enumerate(text.splitlines(), 1):
        for r in rules:
            if r["category"] == "banned subtitle" and r["compiled"].search(line):
                add("banned-subtitle", r["severity"], None, None, line.strip()[:100],
                    r["message"], r["suggestion"], r["id"])

    # 9. Self reference / totalizers / vague time (also covered by DB; keep structural)
    shown = 0
    for pi, s, off in sentences:
        hits = []
        if SELF_REF_RE.search(s):
            hits.append("self-ref")
        if TOTALIZER_RE.search(s):
            hits.append("totalizer")
        if VAGUE_TIME_RE.search(s):
            hits.append("vague-time")
        if hits:
            add("no-value", "warning", pi, off, s[:120],
                f"{','.join(hits)}: cut or name the thing.")
            shown += 1
            if shown >= 15:
                break

    # 10. Repeated time claims
    tc = TIME_CLAIM_RE.findall(text)
    if len(tc) >= 3:
        add("repeated-time-claim", "warning", None, None,
            f"{len(tc)}x: {dict(Counter(t.lower() for t in tc))}",
            "Time claim repeated 3+ times. Say it once.")

    # 11. First person pronouns
    fp = FIRST_PERSON_RE.findall(text)
    if fp:
        c = Counter(w.lower() for w in fp)
        shown = 0
        for pi, s, off in sentences:
            if FIRST_PERSON_RE.search(s):
                add("first-person", "warning", pi, off, s[:100],
                    f"In docs use system voice. ({dict(c)} overall)")
                shown += 1
                if shown >= 8:
                    break

    # 12. Standout boxes
    for pi, p in enumerate(paragraphs):
        for line in p.splitlines():
            if BOX_RE.search(line):
                add("standout-box", "warning", pi, text.find(p), line.strip()[:100],
                    "Standout box/admonition. Delete; fold essentials into body.")
                break

    errors = sum(1 for f in findings if f["severity"] == "error")
    warnings = sum(1 for f in findings if f["severity"] == "warning")

    if want_json:
        report = {
            "tool": "audit.py",
            "file": path.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_issues": len(findings),
            "errors": errors,
            "warnings": warnings,
            "details": findings,
        }
        print(json.dumps(report, indent=2))
    else:
        print(f"# Auto audit: {path.name}")
        print(f"Paragraphs: {len(paragraphs)}, Sentences: {len(sentences)}")
        print(f"Issues: {len(findings)} ({errors} error, {warnings} warning)\n")
        by_check = {}
        for f in findings:
            by_check.setdefault(f["check"], []).append(f)
        for check, fs in by_check.items():
            print(f"## {check} ({len(fs)})")
            for f in fs[:12]:
                loc = f"line {f['line']}" if f["line"] else (f"para {f['para']}" if f["para"] else "")
                tag = f"[{f['rule_id']}] " if f["rule_id"] else ""
                sev = "ERROR " if f["severity"] == "error" else ""
                sug = f" -> {f['suggestion']}" if f["suggestion"] else ""
                print(f"- {sev}{tag}{loc}: {f['matched']}{sug}")
                if f["message"]:
                    print(f"    {f['message']}")
            if len(fs) > 12:
                print(f"  ... {len(fs) - 12} more")
            print()
        print("Done. Next: paragraph by paragraph holistic review per SKILL.md.")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
