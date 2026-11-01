#!/usr/bin/env python3
"""Corpus-level duplicate detection for the writing audit.

Usage: python3 audit_corpus.py <file> [<file> ...] [--json]

Finds sentences and paragraphs repeated ACROSS pages — the redundancy a
per-page audit cannot see. Complements audit.py (which works per file).

Checks (stdlib only):
- exact duplicate sentences across files (normalized, >40 chars)
- near-duplicate sentences across files (difflib > 0.88)
- exact duplicate paragraphs across files (>80 chars)
- repeated distinctive claims: sentences sharing rare 4-word phrases

HTML tags are stripped before comparison so md and html pages compare as prose.
Output is clusters with file:para references — flags for review, not verdicts.
--json emits a structured report. Exit code 1 when clusters exist, else 0.
"""
import json
import re
import sys
import difflib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit import split_sentences, split_paragraphs  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
# Boilerplate blocks that are not prose: styles, scripts, SVG figures.
STRIP_BLOCKS_RE = re.compile(
    r"<(style|script|svg)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)


def clean(text):
    text = STRIP_BLOCKS_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    return text


def norm(s):
    return WS_RE.sub(" ", s.lower().strip())


def main():
    raw = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_json = "--json" in sys.argv[1:]
    if len(raw) < 2:
        print("Usage: python3 audit_corpus.py <file> [<file> ...] [--json]")
        sys.exit(2)

    files = [Path(a) for a in raw]
    # (file_idx, para_idx, sentence)
    sents = []
    paras = []
    for fi, path in enumerate(files):
        text = clean(path.read_text())
        for pi, p in enumerate(split_paragraphs(text)):
            if len(norm(p)) > 80:
                paras.append((fi, pi, p))
            for s in split_sentences(p):
                if len(norm(s)) > 40:
                    sents.append((fi, pi, s))

    names = [f.name for f in files]
    clusters = []

    def locs(v):
        return [{"file": names[fi], "para": pi + 1} for fi, pi, _ in v]

    # 1. Exact duplicate sentences across files
    by_norm = defaultdict(list)
    for fi, pi, s in sents:
        by_norm[norm(s)].append((fi, pi, s))
    dups = {k: v for k, v in by_norm.items()
            if len({fi for fi, _, _ in v}) > 1}
    for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1])):
        clusters.append({
            "type": "exact-sentence",
            "severity": "warning",
            "occurrences": len(v),
            "locations": locs(v),
            "text": v[0][2][:200],
            "message": "Same sentence on multiple pages. One page owns it; cut or condense elsewhere.",
        })

    # 2. Near-duplicate sentences across files
    keys = list({norm(s) for _, _, s in sents})
    shown = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = difflib.SequenceMatcher(None, keys[i], keys[j]).ratio()
            if r > 0.88:
                pair = tuple(sorted((keys[i], keys[j])))
                if pair in shown:
                    continue
                shown.add(pair)
                li = sorted({names[fi] for fi, _, s in sents if norm(s) == keys[i]})
                lj = sorted({names[fi] for fi, _, s in sents if norm(s) == keys[j]})
                clusters.append({
                    "type": "near-duplicate",
                    "severity": "warning",
                    "similarity": round(r, 2),
                    "locations": [{"file": f, "para": None} for f in sorted(set(li) | set(lj))],
                    "text": f"\"{keys[i][:120]}\" vs \"{keys[j][:120]}\"",
                    "message": "Near-duplicate sentences across pages. Merge or cut one.",
                })
                if len([c for c in clusters if c["type"] == "near-duplicate"]) >= 15:
                    break
        if len([c for c in clusters if c["type"] == "near-duplicate"]) >= 15:
            break

    # 3. Exact duplicate paragraphs across files
    by_para = defaultdict(list)
    for fi, pi, p in paras:
        by_para[norm(p)].append((fi, pi, p))
    pdups = {k: v for k, v in by_para.items()
             if len({fi for fi, _, _ in v}) > 1}
    for k, v in sorted(pdups.items(), key=lambda kv: -len(kv[1])):
        clusters.append({
            "type": "exact-paragraph",
            "severity": "warning",
            "occurrences": len(v),
            "locations": locs(v),
            "text": v[0][2][:200],
            "message": "Same paragraph on multiple pages. One page owns it.",
        })

    # 4. Repeated distinctive claims (rare 4-grams appearing in 2+ files)
    gram_files = defaultdict(set)
    for fi, _, s in sents:
        words = norm(s).split()
        for n in range(len(words) - 3):
            g = " ".join(words[n:n + 4])
            if len(g) < 20:
                continue
            gram_files[g].add(fi)
    rep = {g: fs for g, fs in gram_files.items() if len(fs) > 1}
    rep_grams = sorted(rep, key=len, reverse=True)
    kept = []
    for g in rep_grams:
        if not any(g != k and g in k for k in kept):
            kept.append(g)
        if len(kept) >= 12:
            break
    for g in kept:
        clusters.append({
            "type": "repeated-phrase",
            "severity": "warning",
            "locations": [{"file": names[fi], "para": None} for fi in sorted(rep[g])],
            "text": f"...{g}...",
            "message": "Distinctive phrase repeated across pages. Check for a restated claim.",
        })

    if want_json:
        report = {
            "tool": "audit_corpus.py",
            "files": names,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sentences": len(sents),
            "paragraphs": len(paras),
            "clusters": len(clusters),
            "details": clusters,
        }
        print(json.dumps(report, indent=2))
        sys.exit(1 if clusters else 0)

    print(f"# Corpus audit: {len(files)} files, "
          f"{len(sents)} sentences, {len(paras)} paragraphs\n")

    def show(title, items, fmt):
        print(f"## {title}")
        if items:
            for c in items:
                print("- " + fmt(c))
        else:
            print("- none found")
        print()

    show("Exact duplicate sentences (across files)",
         [c for c in clusters if c["type"] == "exact-sentence"],
         lambda c: f"{c['occurrences']}x "
                   f"[{', '.join(l['file'] + ':p' + str(l['para']) for l in c['locations'])}]: "
                   f"{c['text'][:130]}")
    show("Near-duplicate sentences (similarity > 0.88, across files)",
         [c for c in clusters if c["type"] == "near-duplicate"],
         lambda c: f"{c['similarity']} "
                   f"[{', '.join(l['file'] for l in c['locations'])}]: {c['text'][:130]}")
    show("Exact duplicate paragraphs (across files)",
         [c for c in clusters if c["type"] == "exact-paragraph"],
         lambda c: f"{c['occurrences']}x "
                   f"[{', '.join(l['file'] + ':p' + str(l['para']) for l in c['locations'])}]: "
                   f"{c['text'][:130]}...")
    show("Repeated distinctive phrases (4-grams in 2+ files)",
         [c for c in clusters if c["type"] == "repeated-phrase"],
         lambda c: f"[{', '.join(l['file'] for l in c['locations'])}]: \"{c['text']}\"")
    print("Done. Agent step: for each cluster, decide which page owns the "
          "content and cut or condense the rest.")
    sys.exit(1 if clusters else 0)


if __name__ == "__main__":
    main()
