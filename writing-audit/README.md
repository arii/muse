# writing-audit

Systematic writing audit tools. Mechanical checks run first, holistic judgment runs last.

## Quick start

```bash
python3 scripts/audit.py <file.md>
```

Stdlib only. No installs needed.

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
4. Check duplication, no-value sentences, voice, framing, headers,
   and formatting.
5. Holistic pass last: flow, density, and the author's voice.

## Upgrades

- PassivePy for accurate passive voice detection.
- LanguageTool for grammar.
- Sentence embeddings for paraphrased duplicates difflib misses.

## License

MIT
