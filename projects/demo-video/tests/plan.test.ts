import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  compileScript,
  deriveCues,
  estimateChapterMs,
  PlanError,
  slugify,
} from "../scripts/plan";
import type { Chapter } from "../src/script-types";

const SCRIPT = `---
id: demo
title: Demo Video Toolkit
---

# Demo Video Toolkit

## One command, one video

<!-- visual: title -->

Demo video turns a repo into a demo.

## Numbers are spoken

77 percent of 1,200 runs finished in 3.5 seconds.

## Shown as bullets

<!-- visual: bullets -->

- First bullet
- Second bullet

Spoken narration under the bullets.

## Shown as a terminal

<!-- visual: terminal -->

Runs every stage alone.

\`\`\`bash
$ npx demo-video plan script.md
plan: ok
\`\`\`

## Silent card

<!-- visual: bullets -->

- No narration here
`;

describe("compileScript", () => {
  const script = compileScript(SCRIPT, "demo.md");

  it("reads front matter id and title", () => {
    assert.equal(script.id, "demo");
    assert.equal(script.title, "Demo Video Toolkit");
  });

  it("extracts chapters in order", () => {
    assert.deepEqual(
      script.chapters.map((c) => c.name),
      [
        "One command, one video",
        "Numbers are spoken",
        "Shown as bullets",
        "Shown as a terminal",
        "Silent card",
      ]
    );
  });

  it("parses visual hints", () => {
    assert.deepEqual(
      script.chapters.map((c) => c.visual),
      ["title", "title", "bullets", "terminal", "bullets"]
    );
  });

  it("accepts the screenshot visual hint", () => {
    const shot = compileScript(
      "# T\n## App screen\n<!-- visual: screenshot -->\n\nNarrated over the image.\n",
      "shot.md"
    );
    assert.equal(shot.chapters[0]!.visual, "screenshot");
  });

  it("spells numbers as read in narration and cues", () => {
    const spoken = script.chapters[1]!;
    assert.equal(
      spoken.narration,
      "seventy-seven percent of one thousand two hundred runs finished in three point five seconds."
    );
    // Sentence-level cues cut verbatim from the same narration string.
    assert.equal(spoken.cues.map((c) => c.text).join(" "), spoken.narration);
  });

  it("collects bullets and terminal lines separately from narration", () => {
    const bullets = script.chapters[2]!;
    assert.deepEqual(bullets.bullets, ["First bullet", "Second bullet"]);
    assert.equal(bullets.narration, "Spoken narration under the bullets.");
    const terminal = script.chapters[3]!;
    assert.deepEqual(terminal.terminal, ["$ npx demo-video plan script.md", "plan: ok"]);
    assert.equal(terminal.narration, "Runs every stage alone.");
  });

  it("renders a narration-less chapter as an explicit timed silent card", () => {
    const silent = script.chapters[4]!;
    assert.equal(silent.narration, "");
    assert.deepEqual(silent.cues, []);
    assert.equal(silent.durationMs, 3000);
  });

  it("sums chapter durations into the script duration", () => {
    const sum = script.chapters.reduce((s, c) => s + c.durationMs, 0);
    assert.equal(script.durationMs, sum);
  });

  it("slugifies chapter names", () => {
    assert.equal(slugify("One command, one video!"), "one-command-one-video");
  });

  it("rejects unknown visuals with a named error", () => {
    assert.throws(
      () =>
        compileScript(
          "# T\n## C\n<!-- visual: kenburns -->\n\nText.\n",
          "bad.md"
        ),
      (err: unknown) => err instanceof PlanError && /kenburns/.test(err.message)
    );
  });

  it("parses a replay chapter with its clip spec in the fence block", () => {
    const replay = compileScript(
      "---\nid: r\ntitle: R\n---\n# R\n## Live footage\n\n<!-- visual: replay -->\n\nNarration here.\n\n```replay\nlive-dashboard.mp4\n```\n",
      "replay.md"
    ).chapters[0]!;
    assert.equal(replay.visual, "replay");
    assert.deepEqual(replay.terminal, ["live-dashboard.mp4"]);
    assert.ok(replay.narration.includes("Narration here."));
  });

  it("rejects scripts with no chapters", () => {
    assert.throws(
      () => compileScript("# T\n\nJust prose.\n", "bad.md"),
      PlanError
    );
  });

  it("rejects invalid composition ids", () => {
    assert.throws(
      () =>
        compileScript("---\nid: my_comp\n---\n# T\n## C\n\nText.\n", "bad.md"),
      (err: unknown) => err instanceof PlanError && /my_comp/.test(err.message)
    );
  });
});

describe("estimateChapterMs", () => {
  it("scales with word count at ~160 wpm plus a pad", () => {
    const chapter: Chapter = {
      narration: "word ".repeat(32).trim(),
    } as Chapter;
    const ms = estimateChapterMs(chapter);
    assert.ok(ms >= 12000 && ms < 13200, `got ${ms}`);
  });

  it("floors empty narration at the silent-card minimum", () => {
    assert.equal(estimateChapterMs({ narration: "" }), 3000);
  });
});

describe("deriveCues", () => {
  it("covers the whole chapter with no gaps and verbatim text", () => {
    const cues = deriveCues(["short.", "a much longer narration paragraph."], 4000);
    assert.equal(cues[0]!.startMs, 0);
    assert.equal(cues[cues.length - 1]!.endMs, 4000);
    for (let i = 1; i < cues.length; i++) {
      assert.equal(cues[i]!.startMs, cues[i - 1]!.endMs);
    }
    assert.deepEqual(
      cues.map((c) => c.text),
      ["short.", "a much longer narration paragraph."]
    );
  });

  it("gives longer paragraphs proportionally more time", () => {
    const cues = deriveCues(["one.", "one two three four five six."], 4000);
    const short = cues[0]!.endMs - cues[0]!.startMs;
    const long = cues[1]!.endMs - cues[1]!.startMs;
    assert.ok(long > short * 3, `short=${short} long=${long}`);
  });
});
