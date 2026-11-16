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

## Code-slop audit

`scripts/slop_audit.py` scans Python and shell using Python 3.10+ and the stdlib
(`.sh`, `.bash`, `.zsh`, or extensionless
shell shebangs), skips `.git`, `venv`, `.venv`, `env`, `node_modules` and
`__pycache__`, and prints findings grouped by detector with `file:line` and
a final per-detector summary. No paths means the current directory.

```bash
python3 writing-audit/scripts/slop_audit.py .
python3 writing-audit/scripts/slop_audit.py --diff origin/main
python3 writing-audit/scripts/slop_audit.py src --detectors fallback-chain,swallowed-exception
python3 writing-audit/scripts/slop_audit.py --disable near-duplicate
python3 -m unittest discover -s writing-audit/scripts -p 'test_slop_audit.py' -v
```

Exit status: **1** for findings, **0** when clean, **2** for CLI, Git or
file errors. Invalid Python produces a `parse-error` finding.
`--diff BASE` compares BASE to the working tree (including staged and
unstaged edits to tracked files), reporting only new-side changed lines;
untracked files require adding to Git first. Deleted lines are not audited.
Use a merge base explicitly if that is the desired comparison.

Detectors:

| Detector | Flags |
| --- | --- |
| `fallback-chain` | Similar try/handler ASTs or the same shell command with near-identical arguments after `||` (0.88 similarity) |
| `swallowed-exception` | Bare, Exception or BaseException handlers containing only pass, None or ellipsis |
| `dead-fallback` | Statements immediately after return/raise/continue/break; identical if/else bodies; analogous shell control transfers and branches |
| `near-duplicate` | Python statement-list blocks of at least four statements, AST-normalized for names and literal values; four-line shell blocks normalized for numeric variations |
| `over-defensive` | Four nested if guards or four hasattr/getattr/is-not-None guards on one name per Python function; four nested shell ifs |
| `agent-leftover` | TODO/FIXME/HACK/XXX comments without an issue reference such as #123 or owner/repo#123 |

Thresholds are named module constants: `SIMILARITY_RATIO=0.88`,
`GUARD_THRESHOLD=4`, `DUPLICATE_BLOCK_SIZE=4`,
`PYTHON_DUPLICATE_MIN_CHARS=200` (normalized AST dump) and
`SHELL_DUPLICATE_MIN_CHARS=100` (normalized shell text).
Duplication is checked
within each file, not across files; small repeated idioms are intentionally
ignored. Python strings/docstrings are not comments. Shell analysis is
line-based, not a complete shell parser: complex heredocs, command
substitutions and multiline constructs may need manual review. Python's shlex
cannot tokenize all valid Bash, including ANSI-C strings with escaped apostrophes
such as `echo $'it\'s'`. Untokenizable lines are skipped with a stderr warning
`file:line: skipped: shell line not tokenizable`, not a finding; the warning
does not change the exit code, even with every detector disabled. Broad
handlers which log or re-raise are not considered swallowed exceptions.

Suppress a finding **on the reported line**, specifying a real reason:

```python
except Exception:  # slop: disable=swallowed-exception -- cleanup must not mask the original error
    pass
```

Multiple detector names may be comma-separated. Missing reasons or unknown
names produce an unsuppressible `suppression` finding, even when that
detector is disabled. Suppressions do not cover entire files or blocks.
The hand-written `projects/grasping-2026/planning/primitives.py` acceptance
fixture requires no suppressions.

### Pre-merge checks

For repositories using pre-commit, add this local hook to
`.pre-commit-config.yaml` (pre-commit itself is optional, not a tool dependency):

```yaml
repos:
  - repo: local
    hooks:
      - id: slop-audit
        name: Code-slop audit
        entry: python3 writing-audit/scripts/slop_audit.py
        language: system
        types_or: [python, shell]
```

For a diff-only GitHub Actions pre-merge gate, use the standalone
[`slop-workflow.yml.draft`](slop-workflow.yml.draft) template.

Mark this job as a required status check in branch protection. Existing
repository CI runs the detector unit tests, not a whole-repository slop
gate: existing findings should be reviewed before enabling that gate.
