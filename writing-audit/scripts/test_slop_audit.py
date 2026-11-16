"""Regression tests for the code audit (stdlib unittest)."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest

import slop_audit as audit


class DetectorTests(unittest.TestCase):
    def assert_detector(self, detector, positive, negative, filename="sample.py"):
        self.assertIn(detector, {f.detector for f in audit.audit_source(positive, filename)})
        self.assertNotIn(detector, {f.detector for f in audit.audit_source(negative, filename)})

    def test_fallback_chain(self):
        self.assert_detector("fallback-chain", "try:\n    fetch(timeout=10)\nexcept OSError:\n    fetch(timeout=20)\n",
                             "try:\n    fetch()\nexcept OSError:\n    raise\n")
        self.assert_detector("fallback-chain", "curl --retry 1 host || curl --retry 2 host\n",
                             "build || report_failure\n", "sample.sh")

    def test_swallowed_exception(self):
        for handler in ("except:", "except Exception:", "except BaseException:"):
            self.assert_detector("swallowed-exception", f"try:\n    run()\n{handler}\n    pass\n",
                                 "try:\n    run()\nexcept ValueError:\n    pass\n")
        self.assert_detector("swallowed-exception", "try:\n    run()\nexcept Exception:\n    ...\n",
                             "try:\n    run()\nexcept Exception:\n    raise\n")

    def test_dead_fallback(self):
        self.assert_detector("dead-fallback", "def f():\n    return 1\n    recover()\n",
                             "def f():\n    recover()\n    return 1\n")
        self.assert_detector("dead-fallback", "if x:\n    run()\nelse:\n    run()\n",
                             "if x:\n    run()\nelse:\n    recover()\n")
        self.assert_detector("dead-fallback", "if test x; then\n  run\nelse\n  run\nfi\n",
                             "if test x; then\n  run\nelse\n  recover\nfi\n", "sample.sh")
        self.assert_detector("dead-fallback", "return 1\nrecover\n", "return 1\nfi\n", "sample.sh")

    def test_near_duplicate(self):
        first = "def first():\n    a = read('a')\n    b = transform(a)\n    save(b)\n    return b\n"
        second = "def second():\n    x = read('b')\n    y = transform(x)\n    save(y)\n    return y\n"
        self.assert_detector("near-duplicate", first + second, first)
        block = "download --timeout 10 remote\ntransform --quality 10 input\narchive --level 10 output\nupload --retries 10 result\n"
        self.assert_detector("near-duplicate", block + block.replace("10", "20"), block, "sample.sh")

    def test_over_defensive(self):
        self.assert_detector("over-defensive", "if a:\n if b:\n  if c:\n   if d:\n    run()\n",
                             "if a:\n if b:\n  run()\n")
        self.assert_detector("over-defensive", "def f(x):\n" + "    if x is not None:\n        run()\n" * 4,
                             "def f(x):\n    if x is not None:\n        run()\n")
        self.assert_detector("over-defensive", "def f(x):\n" + "    if hasattr(x, 'a'):\n        run()\n" * 4,
                             "def f(x):\n    if getattr(x, 'a', None):\n        run()\n")

    def test_agent_leftover(self):
        for filename in ("sample.py", "sample.sh"):
            self.assert_detector("agent-leftover", "# TODO implement\n", "# TODO owner/repo#123\n", filename)
            self.assertEqual([], audit.audit_source('echo="# TODO not a comment"\n', filename))

    def test_suppressions(self):
        source = "try:\n    run()\nexcept Exception:  # slop: disable=swallowed-exception -- best-effort cleanup\n    pass\n"
        self.assertEqual([], audit.audit_source(source, "sample.py"))
        invalid = source.replace(" -- best-effort cleanup", "")
        self.assertEqual({"suppression", "swallowed-exception"}, {f.detector for f in audit.audit_source(invalid, "sample.py")})
        self.assertEqual({"suppression"}, {f.detector for f in audit.audit_source(invalid, "sample.py", [])})
        self.assertEqual([], audit.audit_source('s = "# slop: disable=invalid"\n', "sample.py"))

    def test_empty_and_invalid(self):
        for name in ("sample.py", "sample.sh"):
            self.assertEqual([], audit.audit_source("", name))
        self.assertEqual("parse-error", audit.audit_source("def invalid(\n", "sample.py")[0].detector)


class CLITests(unittest.TestCase):
    def test_paths_selection_skips_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("# TODO implement\n")
            (root / "venv").mkdir()
            (root / "venv" / "bad.py").write_text("invalid Python !")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(1, audit.main([directory]))
                self.assertEqual(0, audit.main([directory, "--disable", "agent-leftover"]))
                self.assertEqual(1, audit.main([directory, "--detectors", "agent-leftover"]))
            self.assertIn("agent-leftover=1", output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(2, audit.main([str(root / "missing")]))
                with self.assertRaises(SystemExit):
                    audit.main([directory, "--detectors", "invalid"])

    def test_diff_changed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT)
            git("init")
            path = root / "space name.py"
            path.write_text("# TODO old\nvalue = 1\n")
            git("add", str(path))
            git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture")
            path.write_text("# TODO old\nvalue = 1\n# FIXME new\n")
            changed = audit.changed_lines("HEAD", root)
            findings = audit.audit_source(path.read_text(), path)
            self.assertEqual([3], [f.line for f in findings if f.line in changed[path.resolve()]])
            command = ["python3", str(Path(audit.__file__).resolve()), "--diff", "HEAD"]
            result = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertEqual(1, result.returncode)
            self.assertIn("agent-leftover=1", result.stdout)
            self.assertNotIn("space name.py:1:", result.stdout)
            path.unlink()
            self.assertEqual(set(), audit.changed_lines("HEAD", root).get(path.resolve(), set()))


if __name__ == "__main__":
    unittest.main()
