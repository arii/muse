# writing-audit

Systematic writing audit tools. Two steps, always in order.

## The two steps

**Step 1 — Deterministic analysis.** Run the script on the page:

```bash
python3 scripts/audit.py <file.md>
python3 scripts/audit.py <file.md> --json   # structured report
```

When auditing a set of pages, also run the corpus pass:

```bash
python3 scripts/audit_corpus.py <file1> <file2> ...
python3 scripts/audit_corpus.py <file1> <file2> ... --json
```

It reports sentences, paragraphs, and distinctive claims repeated
*across* pages — redundancy a per-page audit cannot see. HTML tags are
stripped so markdown and HTML pages compare as prose.

Stdlib only, no installs needed. Same input gives same output every time.
Output is flags for review, not verdicts. Exit code is 1 when a
severity=error finding exists (banned terms, prompt leakage, banned
subtitles), else 0 — so CI can fail the build on errors.

## Terminology database

`terminology-db.json` is the source of truth: 70 rules with ids,
categories, regex support, case sensitivity, severity (error/warning),
suggestions, and messages. `terminology-db.md` is the human-readable
companion. Extend per project by adding rules to the JSON.

## CI

`github-workflow.yml.draft` is a GitHub Actions workflow that runs the
audit on PRs touching docs and fails on errors. The connector cannot write
under `.github/workflows/`, so commit it manually at
`.github/workflows/writing-audit.yml`.

**Step 2 — Agent checklist.** Copy `checklist-template.md` for each page
reviewed and fill every item: paragraph walk, terminology, duplication,
no-value sentences, voice, framing, headers, holistic. Check off what
passes, flag what fails with a note, record a PASS or FLAG verdict.

Never skip step 1. Never present step 1 output as the audit. The audit is
both steps.

## What the script flags

- Exact and near-duplicate sentences (difflib, 0.85 threshold)
- Duplicate paragraph openings
- Terminology hits from `terminology-db.md`
- Passive voice (heuristic; use PassivePy for accuracy)
- Leading and rhetorical questions
- Long sentences (over 28 words)
- Banned subtitles (`Why it matters`)
- Self reference, vague totalizers, vague time markers, trivia IDs
- Repeated time claims (two days / 48 hours said 3+ times)
- First person pronouns (technical docs use system voice)
- Standout boxes and admonitions

Every flag is a review prompt, not a verdict.

## The checklist

The full paragraph-by-paragraph method lives in the Muse skill
`writing-audit`. In short:

1. Run the script.
2. Walk the document paragraph by paragraph, every sentence, in order.
   Each paragraph must be required, not already said, accurate or
   hallucinated, citable, and adding value.
3. Apply the terminology database.
4. Check duplication (within-page and corpus-level), no-value sentences, voice, framing, headers,
   and formatting.
5. Holistic pass last: flow, density, and the author's voice.

## Upgrades

- PassivePy for accurate passive voice detection.
- LanguageTool for grammar.
- Sentence embeddings for paraphrased duplicates difflib misses.

## License

MIT
