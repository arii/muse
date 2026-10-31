# Writing audit checklist

Copy this template for each page reviewed. Fill every item. Check off what
passes, flag what fails with a note. One filled copy per page, saved under
the audit run directory.

Auditor:
Date:
Page: (repo-relative path)
Source: (repo @ branch or commit)
Scope: (README / docs / website)

## Step 1 — Deterministic analysis

Run `python3 ~/workspace/writing-audit/scripts/audit.py <page>` first.
The script is mechanical and reproducible. Its output is flags for review,
not verdicts. Confirm or dismiss each flag in context.

- [ ] Script ran cleanly on the page
- [ ] Every flag below confirmed or dismissed with a note

Flag summary:
- Exact duplicate sentences:
- Near duplicate sentences:
- Repeated paragraph openings:
- Terminology hits:
- Passive voice:
- Leading questions:
- Long sentences:
- Banned subtitles:
- Self reference / totalizers / vague time / trivia:
- Repeated time claims:
- First person pronouns:
- Standout boxes:

## Step 2 — Agent assessment

### Paragraph walk

- [ ] Read every paragraph in document order, every sentence inside it
- [ ] Each paragraph judged: required? already said? accurate or hallucinated? citable? adds value?

Notes:

### Terminology database

- [ ] Every sentence checked against `terminology-db.md`
- [ ] Corporate filler, redundant restatements, vague marketing modifiers, AI metacognition, buzzwords flagged

Notes:

### Duplication and redundancy

- [ ] No exact or near-duplicate sentences across paragraphs
- [ ] No repeated paragraph openings, no say-it-twice captions
- [ ] No caption restating a graphic, no redundant heading, no repeated time claims (3+)
- [ ] Fits-another-robot test passed, no sections restating --help flags or tool-enforced behavior

Notes:

### No-value sentences

- [ ] No sentimental filler stating the obvious
- [ ] No self-referential meta, no vague totalizers, no vague time markers
- [ ] No trivia identifiers, no startup jargon

Notes:

### Voice and grammar

- [ ] Active voice throughout, named actor in every sentence
- [ ] Technical docs use system voice (no we / our / us); blog posts stay first person
- [ ] No hedging, no sentences over 28 words, no unmeasured claims

Notes:

### Framing and tone

- [ ] No negative framing; every sentence states what is true
- [ ] No leading or rhetorical questions
- [ ] No hyperbole, punchy lines, slogans, or cute phrasing; no "Why it matters" section

Notes:

### Headers and formatting

- [ ] Headers are descriptive phrases that say something, not single-word labels
- [ ] Every header earns its place; fewer beats more
- [ ] Bold only for the term being defined; no standout boxes, admonitions, or callouts

Notes:

### Holistic

- [ ] Flow works across paragraphs; no throat-clearing opener; no summary restating the intro
- [ ] Information density high; text that adds nothing is cut even if it passes mechanical checks
- [ ] Ari's rewrites preserved word for word

Notes:

## Verdict

- [ ] PASS — no flags
- [ ] FLAG — findings listed below

## Findings

Numbered list. Each: the flagged sentence, the rule it breaks, a concrete
rewrite. Keep rewrites short, active, and plain.
