#!/usr/bin/env python3
"""Corpus-level duplicate detection for the writing audit.

Usage: python3 audit_corpus.py <file> [<file> ...]

Finds sentences and paragraphs repeated ACROSS pages — the redundancy a
per-page audit cannot see. Complements audit.py (which works per file).

Checks (stdlib only):
- exact duplicate sentences across files (normalized, >40 chars)
- near-duplicate sentences across files (difflib > 0.88)
- exact duplicate paragraphs across files (>80 chars)
- repeated distinctive claims: sentences sharing rare 4-word phrases

HTML tags are stripped before comparison so md and html pages compare as prose.
Output is clusters with file:para references — flags for review, not verdicts.
"""
import re
import sys
import difflib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit import split_sentences, split_paragraphs  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean(text):
    text = TAG_RE.sub(" ", text)
    return text


def norm(s):
    return WS_RE.sub(" ", s.lower().strip())


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 audit_corpus.py <file> [<file> ...]")
        sys.exit(1)

    files = [Path(a) for a in sys.argv[1:]]
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
    print(f"# Corpus audit: {len(files)} files, "
          f"{len(sents)} sentences, {len(paras)} paragraphs\n")

    # 1. Exact duplicate sentences across files
    print("## Exact duplicate sentences (across files)")
    by_norm = defaultdict(list)
    for fi, pi, s in sents:
        by_norm[norm(s)].append((fi, pi, s))
    dups = {k: v for k, v in by_norm.items()
            if len({fi for fi, _, _ in v}) > 1}
    if dups:
        for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1])):
            locs = ", ".join(f"{names[fi]}:p{pi+1}" for fi, pi, _ in v)
            print(f"- {len(v)}x [{locs}]: {v[0][2][:130]}")
    else:
        print("- none found")
    print()

    # 2. Near-duplicate sentences across files
    print("## Near-duplicate sentences (similarity > 0.88, across files)")
    keys = list({norm(s) for _, _, s in sents})
    shown = set()
    found = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = difflib.SequenceMatcher(None, keys[i], keys[j]).ratio()
            if r > 0.88:
                pair = tuple(sorted((keys[i], keys[j])))
                if pair in shown:
                    continue
                shown.add(pair)
                loc_i = [f"{names[fi]}" for fi, _, s in sents
                         if norm(s) == keys[i]]
                loc_j = [f"{names[fi]}" for fi, _, s in sents
                         if norm(s) == keys[j]]
                print(f"- {r:.2f} [{','.join(sorted(set(loc_i)))}] vs "
                      f"[{','.join(sorted(set(loc_j)))}]")
                print(f"    \"{keys[i][:100]}\"")
                print(f"    \"{keys[j][:100]}\"")
                found += 1
                if found >= 15:
                    break
        if found >= 15:
            break
    if not found:
        print("- none found")
    print()

    # 3. Exact duplicate paragraphs across files
    print("## Exact duplicate paragraphs (across files)")
    by_para = defaultdict(list)
    for fi, pi, p in paras:
        by_para[norm(p)].append((fi, pi, p))
    pdups = {k: v for k, v in by_para.items()
             if len({fi for fi, _, _ in v}) > 1}
    if pdups:
        for k, v in sorted(pdups.items(), key=lambda kv: -len(kv[1])):
            locs = ", ".join(f"{names[fi]}:p{pi+1}" for fi, pi, _ in v)
            print(f"- {len(v)}x [{locs}]: {v[0][2][:130]}...")
    else:
        print("- none found")
    print()

    # 4. Repeated distinctive claims (rare 4-grams appearing in 2+ files)
    print("## Repeated distinctive phrases (4-grams in 2+ files)")
    gram_files = defaultdict(set)
    gram_example = {}
    for fi, _, s in sents:
        words = norm(s).split()
        for n in range(len(words) - 3):
            g = " ".join(words[n:n + 4])
            if len(g) < 20:
                continue
            gram_files[g].add(fi)
            gram_example.setdefault(g, s)
    rep = {g: fs for g, fs in gram_files.items() if len(fs) > 1}
    # keep the longest / most distinctive: filter grams contained in a longer one
    rep_grams = sorted(rep, key=len, reverse=True)
    kept = []
    for g in rep_grams:
        if not any(g != k and g in k for k in kept):
            kept.append(g)
        if len(kept) >= 12:
            break
    if kept:
        for g in kept:
            fs = sorted(names[fi] for fi in rep[g])
            print(f"- [{','.join(fs)}]: \"...{g}...\"")
    else:
        print("- none found")
    print()
    print("Done. Agent step: for each cluster, decide which page owns the "
          "content and cut or condense the rest.")


if __name__ == "__main__":
    main()
