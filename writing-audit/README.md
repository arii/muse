# writing-audit

Audit prose with deterministic checks and a per-page checklist.

## The two steps

**Step 1: Deterministic analysis.** Run the script on the page:

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
across pages. HTML tags are
stripped so markdown and HTML pages compare as prose.

The scripts use the Python standard library. Review each flag in context.
Exit code is 1 when a
severity=error finding exists (banned terms, prompt leakage, banned
subtitles), else 0.

## Terminology database

`terminology-db.json` is the source of truth: 70 rules with ids,
categories, regex support, case sensitivity, severity (error/warning),
suggestions, and messages. `terminology-db.md` is the human-readable
companion. Extend per project by adding rules to the JSON.

## CI

Copy [`github-workflow.yml.draft`](github-workflow.yml.draft) to
`.github/workflows/writing-audit.yml` to audit PRs touching docs and fail
on errors.

**Step 2: Agent checklist.** Copy `checklist-template.md` for each page
reviewed and fill every item: paragraph walk, terminology, duplication,
no-value sentences, voice, framing, headers, holistic. Check off what
passes, flag what fails with a note, record a PASS or FLAG verdict.

The full paragraph-by-paragraph method lives in the Muse skill `writing-audit`.

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
| `fallback-chain` | Similar try/handler ASTs or the same shell command with near-identical arguments after `||` |
| `swallowed-exception` | Bare, Exception or BaseException handlers containing only pass, None or ellipsis |
| `dead-fallback` | Statements immediately after return/raise/continue/break; identical if/else bodies; analogous shell control transfers and branches |
| `near-duplicate` | Python statement-list blocks AST-normalized for names and literal values; shell blocks normalized for numeric variations |
| `over-defensive` | Nested if guards or repeated hasattr/getattr/is-not-None guards on one name per Python function; nested shell ifs |
| `agent-leftover` | TODO/FIXME/HACK/XXX comments without an issue reference such as #123 or owner/repo#123 |

Thresholds are named module constants: `SIMILARITY_RATIO=0.88`,
`GUARD_THRESHOLD=4`, `DUPLICATE_BLOCK_SIZE=4`,
`PYTHON_DUPLICATE_MIN_CHARS=200` (normalized AST dump) and
`SHELL_DUPLICATE_MIN_CHARS=100` (normalized shell text).
Duplication checks operate within each file and exclude blocks below those
thresholds. Python comment analysis uses comment tokens. Shell analysis
processes individual lines; review complex heredocs, command substitutions
and multiline constructs manually. Python's shlex
cannot tokenize all valid Bash, including ANSI-C strings with escaped apostrophes
such as `echo $'it\'s'`. Untokenizable lines emit the stderr warning
`file:line: skipped: shell line not tokenizable` and leave the exit status
unchanged, even with every detector disabled. They also end open branch
comparisons and interrupt contiguous-block and control-transfer checks.

Suppress a finding **on the reported line**, specifying a real reason:

```python
except Exception:  # slop: disable=swallowed-exception -- preserve the original error during cleanup
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

Mark the audit job as a required status check in branch protection after
reviewing existing findings. Repository CI currently runs detector unit tests.
