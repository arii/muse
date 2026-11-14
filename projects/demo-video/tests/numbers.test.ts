import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { spellAsRead } from "../scripts/numbers";

describe("spellAsRead", () => {
  it("converts integers to spoken words", () => {
    assert.equal(spellAsRead("7 chapters"), "seven chapters");
    assert.equal(spellAsRead("77 frames"), "seventy-seven frames");
    assert.equal(spellAsRead("1,200 runs"), "one thousand two hundred runs");
    assert.equal(
      spellAsRead("1200000 frames"),
      "one million two hundred thousand frames"
    );
  });

  it("appends percent for % tokens", () => {
    assert.equal(spellAsRead("77% pass rate"), "seventy-seven percent pass rate");
  });

  it("spells decimals digit by digit after 'point'", () => {
    assert.equal(spellAsRead("3.5 seconds"), "three point five seconds");
  });

  it("leaves already-spelled text untouched (idempotent)", () => {
    const text = "seventy-seven percent of one thousand";
    assert.equal(spellAsRead(text), text);
  });

  it("leaves words without digits alone", () => {
    assert.equal(
      spellAsRead("No numbers in this sentence."),
      "No numbers in this sentence."
    );
  });
});
