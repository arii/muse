#!/usr/bin/env python3
"""Automatic pre-checks for writing audit.

Usage: python3 audit.py <file.md>

Runs stdlib only checks:
- exact duplicate sentences
- near duplicate sentences (difflib > 0.85)
- duplicate paragraph openings
- terminology flags from terminology-db.md
- passive voice heuristics
- leading questions
- long sentences
- banned subtitles (Why it matters)
- self reference, totalizers, vague time, trivia IDs
- repeated time claims
- first person pronouns (docs voice)
- standout boxes and admonitions

Outputs a checklist report. Holistic assessment stays with the agent.
"""
import re
import sys
import difflib
from collections import Counter
from pathlib import Path

PASSIVE_RE = re.compile(
    r"\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b"
    r"|\b(has|have|had)\s+been\s+\w+ed\b"
    r"|\b(am|is|are|was|were|be|been|being)\b\s+"
    r"(written|built|done|taken|given|grown|known|shown|drawn|thrown|broken|"
    r"chosen|frozen|spoken|stolen|worn|torn|born|beaten|made|paid|said|heard)\b",
    re.IGNORECASE,
)
BY_RE = re.compile(r"\bby\b", re.IGNORECASE)
QUESTION_RE = re.compile(r"^\s*(#{1,6}\s+)?(Want|Need|Tired|Curious|Did you know|What if|Why)[^?.!]*\?\s*$", re.IGNORECASE)
WHY_MATTERS_RE = re.compile(r"^\s*#{1,6}\s*why\s+(this\s+|it\s+)?matters\s*$", re.IGNORECASE)
SELF_REF_RE = re.compile(r"you are reading|this page you|as you can see here", re.IGNORECASE)
TOTALIZER_RE = re.compile(r"\beverything here\b|\ball of this\b|\beverything below\b", re.IGNORECASE)
TIME_CLAIM_RE = re.compile(r"\btwo days\b|\b48 hours\b|\bunder two days\b|\bin two days\b", re.IGNORECASE)
TRIVIA_RE = re.compile(r"\(unit\s+[^)]+\)|\bserial number\b|\bunit\s+[a-z0-9]+\b", re.IGNORECASE)
VAGUE_TIME_RE = re.compile(r"\bby evening\b|\blater that day\b", re.IGNORECASE)
FIRST_PERSON_RE = re.compile(r"\b(we|our|us|ourselves)\b", re.IGNORECASE)
BOX_RE = re.compile(r"^>\s*\[!|^\s*:::+|<div[^>]*class=\"[^\"]*(box|callout|admonition)", re.IGNORECASE)

TERM_FILES = [Path(__file__).parent.parent / "terminology-db.md"]

def load_terms():
    terms = []
    for tf in TERM_FILES:
        if not tf.exists():
            continue
        for line in tf.read_text().splitlines():
            line = line.strip()
            if line.startswith("- "):
                t = line[2:].strip()
                # skip pattern descriptions in parens heavy lines, keep core phrase
                core = re.split(r"\s*\(e\.g\.", t)[0].strip()
                if core and len(core) > 1:
                    terms.append(core.lower())
    return terms

def split_sentences(text):
    # simple sentence splitter, good enough for audit pre-check
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]

def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 audit.py <file.md>")
        sys.exit(1)
    path = Path(sys.argv[1])
    text = path.read_text()
    terms = load_terms()

    paragraphs = split_paragraphs(text)
    sentences = []
    for pi, p in enumerate(paragraphs):
        for s in split_sentences(p):
            sentences.append((pi, s))

    print(f"# Auto audit: {path.name}")
    print(f"Paragraphs: {len(paragraphs)}, Sentences: {len(sentences)}\n")

    # 1. Exact duplicate sentences (normalized)
    norm = {}
    for pi, s in sentences:
        key = re.sub(r"\s+", " ", s.lower().strip())
        norm.setdefault(key, []).append((pi, s))
    dups = {k: v for k, v in norm.items() if len(v) > 1 and len(k) > 20}
    print("## Exact duplicate sentences")
    if dups:
        for k, v in dups.items():
            paras = sorted(set(pi + 1 for pi, _ in v))
            print(f"- {len(v)}x (paras {paras}): {v[0][1][:120]}")
    else:
        print("- none found")
    print()

    # 2. Near duplicates
    print("## Near duplicate sentences (similarity > 0.85)")
    seen = set()
    found = 0
    keys = list(norm.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if len(keys[i]) < 20 or len(keys[j]) < 20:
                continue
            r = difflib.SequenceMatcher(None, keys[i], keys[j]).ratio()
            if r > 0.85:
                pair = (keys[i], keys[j])
                if pair in seen:
                    continue
                seen.add(pair)
                print(f"- {r:.2f}: \"{keys[i][:90]}\" vs \"{keys[j][:90]}\"")
                found += 1
                if found >= 10:
                    break
        if found >= 10:
            break
    if not found:
        print("- none found")
    print()

    # 3. Duplicate paragraph openings
    print("## Repeated paragraph openings")
    openings = Counter()
    for p in paragraphs:
        first = " ".join(p.split()[:6]).lower()
        openings[first] += 1
    reps = {k: c for k, c in openings.items() if c > 1}
    if reps:
        for k, c in reps.items():
            print(f"- {c}x: {k}...")
    else:
        print("- none found")
    print()

    # 4. Terminology flags
    print("## Terminology flags")
    flagged = 0
    for pi, s in sentences:
        low = s.lower()
        hits = [t for t in terms if t in low and len(t) > 2]
        if hits:
            print(f"- para {pi+1}: {hits} :: {s[:110]}")
            flagged += 1
            if flagged >= 20:
                print("  ... truncated")
                break
    if not flagged:
        print("- none found")
    print()

    # 5. Passive voice heuristic
    print("## Possible passive voice (heuristic, confirm with PassivePy for accuracy)")
    pv = 0
    for pi, s in sentences:
        if PASSIVE_RE.search(s):
            print(f"- para {pi+1}: {s[:120]}")
            pv += 1
            if pv >= 15:
                print("  ... truncated")
                break
    if not pv:
        print("- none found")
    print()

    # 6. Leading / attention seeking questions
    print("## Leading questions")
    q = 0
    for pi, s in sentences:
        if s.strip().endswith("?"):
            # flag if at paragraph start or matches attention pattern
            para = paragraphs[pi]
            is_opener = para.lstrip().startswith(s.strip()[:20])
            if is_opener or QUESTION_RE.search(s):
                print(f"- para {pi+1}: {s[:120]}")
                q += 1
    if not q:
        print("- none found")
    print()

    # 7. Long sentences (simplicity check)
    print("## Long sentences (>28 words)")
    lg = 0
    for pi, s in sentences:
        if len(s.split()) > 28:
            print(f"- para {pi+1} ({len(s.split())} words): {s[:120]}...")
            lg += 1
            if lg >= 10:
                break
    if not lg:
        print("- none found")
    print()

    # 8. Banned subtitles and structural bloat
    print("## Banned subtitles (Why it matters)")
    wm = 0
    for pi, p in enumerate(paragraphs):
        for line in p.splitlines():
            if WHY_MATTERS_RE.search(line):
                print(f"- para {pi+1}: {line.strip()}")
                wm += 1
    if not wm:
        print("- none found")
    print()

    # 9. Self reference, totalizers, vague time, trivia
    print("## Self reference / totalizers / vague time / trivia")
    st = 0
    for pi, s in sentences:
        hits = []
        if SELF_REF_RE.search(s):
            hits.append("self-ref")
        if TOTALIZER_RE.search(s):
            hits.append("totalizer")
        if VAGUE_TIME_RE.search(s):
            hits.append("vague-time")
        if TRIVIA_RE.search(s):
            hits.append("trivia-id")
        if hits:
            print(f"- para {pi+1} [{','.join(hits)}]: {s[:120]}")
            st += 1
            if st >= 15:
                break
    if not st:
        print("- none found")
    print()

    # 10. Repeated time claims (two days / 48 hours said N times)
    print("## Repeated time claims")
    tc = TIME_CLAIM_RE.findall(text)
    if len(tc) >= 3:
        print(f"- FLAGGED: time claim repeated {len(tc)}x: {Counter(t.lower() for t in tc)}")
    elif tc:
        print(f"- {len(tc)}x (ok): {Counter(t.lower() for t in tc)}")
    else:
        print("- none found")
    print()

    # 11. First person pronouns (technical docs use system voice)
    print("## First person pronouns (rewrite as 'the system' in docs)")
    fp = FIRST_PERSON_RE.findall(text)
    if fp:
        print(f"- {len(fp)}x: {Counter(w.lower() for w in fp)}")
        shown = 0
        for pi, s in sentences:
            if FIRST_PERSON_RE.search(s):
                print(f"  - para {pi+1}: {s[:100]}")
                shown += 1
                if shown >= 8:
                    break
    else:
        print("- none found")
    print()

    # 12. Standout boxes and admonitions
    print("## Standout boxes / admonitions")
    bx = 0
    for pi, p in enumerate(paragraphs):
        for line in p.splitlines():
            if BOX_RE.search(line):
                print(f"- para {pi+1}: {line.strip()[:100]}")
                bx += 1
                break
    if not bx:
        print("- none found")
    print()
    print("Done. Next: paragraph by paragraph holistic review per SKILL.md.")

if __name__ == "__main__":
    main()
