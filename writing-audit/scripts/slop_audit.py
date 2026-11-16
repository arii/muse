#!/usr/bin/env python3
"""Conservative, stdlib-only Python and shell code-slop review prompts."""
import argparse
import ast
import copy
import difflib
import io
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass

DETECTORS = ("fallback-chain", "swallowed-exception", "dead-fallback",
             "near-duplicate", "over-defensive", "agent-leftover")
SKIP = {".git", ".venv", "venv", "env", "node_modules", "__pycache__"}


@dataclass(frozen=True, order=True)
class Finding:
    detector: str
    file: str
    line: int
    message: str


def signature(body, normalized=False):
    tree = ast.Module(body=copy.deepcopy(body), type_ignores=[])
    if normalized:
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.setdefault(node.id, f"v{len(names)}")
                node.id = names[node.id]
            elif isinstance(node, ast.Constant):
                node.value = f"<{type(node.value).__name__}>"
    return ast.dump(tree, include_attributes=False)


def similar(left, right):
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.88


def scope_nodes(node):
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            yield from scope_nodes(child)


def python_findings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if similar(signature(node.body), signature(handler.body)):
                    yield "fallback-chain", handler.lineno, "handler repeats the attempted operation"
        if isinstance(node, ast.ExceptHandler):
            types = list(ast.walk(node.type)) if node.type else []
            broad = node.type is None or any(
                isinstance(t, ast.Name) and t.id in {"Exception", "BaseException"}
                for t in types)
            if broad and all(isinstance(s, ast.Pass) or (
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    and s.value.value in (None, Ellipsis)) for s in node.body):
                yield "swallowed-exception", node.lineno, "broad exception handler discards the error"
        if isinstance(node, ast.If):
            if node.orelse and signature(node.body) == signature(node.orelse):
                yield "dead-fallback", node.orelse[0].lineno, "identical branch bodies"
            depth, child = 1, node
            while len(child.body) == 1 and isinstance(child.body[0], ast.If):
                depth += 1
                child = child.body[0]
            if depth >= 4:
                yield "over-defensive", node.lineno, f"{depth} nested if guards"
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(s, ast.stmt) for s in value):
                for previous, statement in zip(value, value[1:]):
                    if isinstance(previous, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                        yield "dead-fallback", statement.lineno, "unreachable statement after control transfer"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            checks = Counter()
            for guard in scope_nodes(node):
                test = guard.test if isinstance(guard, (ast.If, ast.While, ast.IfExp)) else None
                if test is None:
                    continue
                for part in scope_nodes(test):
                    target = None
                    if isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and part.func.id in {"hasattr", "getattr"} and part.args:
                        target = part.args[0]
                    if isinstance(part, ast.Compare) and any(isinstance(op, ast.IsNot) for op in part.ops) and any(isinstance(c, ast.Constant) and c.value is None for c in part.comparators):
                        target = part.left
                    if isinstance(target, ast.Name):
                        checks[target.id] += 1
            for name, count in checks.items():
                if count >= 4:
                    yield "over-defensive", node.lineno, f"{count} defensive checks on {name}"
    seen = {}
    for node in ast.walk(tree):
        for _, body in ast.iter_fields(node):
            if not isinstance(body, list) or len(body) < 4 or not all(isinstance(s, ast.stmt) for s in body):
                continue
            if any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for s in body):
                continue
            key = signature(body, normalized=True)
            if len(key) < 200:
                continue
            if key in seen:
                yield "near-duplicate", body[0].lineno, f"normalized block repeats line {seen[key]}"
            else:
                seen[key] = body[0].lineno


def shell_lines(source):
    for number, line in enumerate(source.splitlines(), 1):
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            continue
        if tokens:
            yield number, tokens


def shell_findings(source):
    lines = list(shell_lines(source))
    seen = {}
    branches = []
    for index, (number, tokens) in enumerate(lines):
        separator = tokens[0] in {"else", "elif", "fi"}
        for _, alternatives in branches[:-1] if separator else branches:
            alternatives[-1].append(tokens)
        if tokens[0] == "if":
            branches.append([number, [[]]])
            if len(branches) == 4:
                yield "over-defensive", number, "four nested shell if guards"
        elif tokens[0] in {"else", "elif"} and branches:
            branches[-1][1].append([])
        elif tokens[0] == "fi" and branches:
            start, alternatives = branches.pop()
            if len(alternatives) > 1 and alternatives[0] and all(
                    body == alternatives[0] for body in alternatives[1:]):
                yield "dead-fallback", start, "identical shell branch bodies"
        if "||" in tokens:
            split = tokens.index("||")
            left, right = tokens[:split], tokens[split + 1:]
            if left and right and left[0] == right[0] and similar(" ".join(left), " ".join(right)):
                yield "fallback-chain", number, "near-identical command after ||"
        if tokens[0] in {"return", "exit", "continue", "break"} and index + 1 < len(lines):
            following, next_tokens = lines[index + 1]
            if next_tokens[0] not in {"fi", "done", "}", "else", "elif", "esac", ";;"}:
                yield "dead-fallback", following, "unreachable command after control transfer"
        block = lines[index:index + 4]
        if len(block) != 4 or any(t[0] in {"if", "then", "else", "fi", "done", "}"} for _, t in block):
            continue
        key = tuple(tuple(re.sub(r"\d+", "N", t) for t in row) for _, row in block)
        if sum(len(" ".join(row)) for row in key) < 100:
            continue
        if key in seen and number >= seen[key] + 4:
            yield "near-duplicate", number, f"text block repeats line {seen[key]}"
        else:
            seen.setdefault(key, number)


def comments(source, python):
    if python:
        try:
            return [(t.start[0], t.string) for t in tokenize.generate_tokens(io.StringIO(source).readline)
                    if t.type == tokenize.COMMENT]
        except (tokenize.TokenError, IndentationError):
            return []
    result = []
    for number, line in enumerate(source.splitlines(), 1):
        # Track quotes so hashes in quoted strings are not comments.
        quote = None
        escaped = False
        for i, char in enumerate(line):
            if escaped:
                escaped = False
            elif char == "\\" and quote != "'":
                escaped = True
            elif char in "\"'" and (quote is None or quote == char):
                quote = char if quote is None else None
            elif char == "#" and quote is None and (i == 0 or line[i - 1].isspace()):
                result.append((number, line[i:]))
                break
    return result


def audit_source(source, filename, enabled=DETECTORS):
    python = str(filename).endswith(".py")
    raw, suppressions = [], {}
    for line, comment in comments(source, python):
        if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", comment) and not re.search(r"#\d+\b", comment):
            raw.append(("agent-leftover", line, "unfinished comment without an issue reference"))
        if "slop: disable=" in comment:
            match = re.search(r"slop: disable=([\w,-]+)\s*--\s*(\S.*)", comment)
            if not match or any(d not in DETECTORS for d in match[1].split(",")):
                raw.append(("suppression", line, "suppression requires valid detector names and -- reason"))
            else:
                suppressions[line] = set(match[1].split(","))
    if python:
        try:
            tree = ast.parse(source, filename=str(filename))
        except SyntaxError as error:
            raw.append(("parse-error", error.lineno or 1, error.msg))
        else:
            raw.extend(python_findings(tree))
    else:
        raw.extend(shell_findings(source))
    return sorted({Finding(d, str(filename), line, message) for d, line, message in raw
                   if (d in enabled or d in {"suppression", "parse-error"})
                   and d not in suppressions.get(line, set())})


def changed_lines(base, root):
    output = subprocess.check_output(
        ["git", "-c", "core.quotePath=false", "diff", "--no-ext-diff", "--no-color",
         "--src-prefix=a/", "--dst-prefix=b/", "--no-renames", "--unified=0", base, "--"],
        cwd=root, text=True)
    changed, current = {}, None
    for line in output.splitlines():
        if line.startswith("+++ "):
            name = line[4:].split("\t", 1)[0]
            if name.startswith('"'):
                name = ast.literal_eval(name)
            current = root / name[2:] if name.startswith("b/") else None
            if current is not None:
                changed[current.resolve()] = set()
        elif line.startswith("@@") and current is not None:
            match = re.search(r"\+(\d+)(?:,(\d+))? @@", line)
            if match:
                start, count = int(match[1]), int(match[2] or 1)
                changed[current.resolve()].update(range(start, start + count))
    return changed


def source_files(paths):
    found = set()
    for argument in paths:
        path = Path(argument)
        if not path.exists():
            raise OSError(f"path does not exist: {path}")
        if path.is_dir():
            candidates = []
            for directory, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in SKIP)
                candidates.extend(Path(directory) / name for name in files)
        else:
            candidates = [path]
        for candidate in candidates:
            if any(part in SKIP for part in candidate.parts) or candidate.is_symlink() or not candidate.is_file():
                continue
            if candidate.suffix in {".py", ".sh", ".bash", ".zsh"}:
                found.add(candidate)
            elif not candidate.suffix:
                with candidate.open("rb") as stream:
                    if re.match(rb"#!.*\b(?:ba|z|da|k)?sh\b", stream.readline(256)):
                        found.add(candidate)
    return sorted(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."])
    parser.add_argument("--diff", metavar="BASE", help="compare BASE to the working tree; report added/modified lines only")
    parser.add_argument("--detectors", default=",".join(DETECTORS), help="comma-separated detector names")
    parser.add_argument("--disable", default="", help="comma-separated detector names")
    args = parser.parse_args(argv)
    selected = set(args.detectors.split(","))
    disabled = set(filter(None, args.disable.split(",")))
    if (selected | disabled) - set(DETECTORS):
        parser.error("unknown detector name")
    enabled = selected - disabled
    findings = []
    try:
        changes = None
        if args.diff:
            root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
            changes = changed_lines(args.diff, root)
        for path in source_files(args.paths):
            if changes is not None and path.resolve() not in changes:
                continue
            found = audit_source(path.read_text(encoding="utf-8"), path, enabled)
            findings.extend(f for f in found if changes is None or f.line in changes[path.resolve()])
    except (OSError, UnicodeError, subprocess.CalledProcessError) as error:
        print(f"slop_audit: {error}", file=sys.stderr)
        return 2
    counts = Counter(f.detector for f in findings)
    for detector in (*DETECTORS, "suppression", "parse-error"):
        group = sorted(f for f in findings if f.detector == detector)
        if group:
            print(f"\n{detector} ({len(group)}):")
            for finding in group:
                print(f"  {finding.file}:{finding.line}: {finding.message}")
    print("\nSummary: " + ", ".join(f"{d}={counts[d]}" for d in DETECTORS)
          + f"; suppression={counts['suppression']}, parse-error={counts['parse-error']}; total={len(findings)}")
    return int(bool(findings))


if __name__ == "__main__":
    sys.exit(main())
