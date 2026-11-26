import assert from "node:assert/strict";
import { test } from "node:test";

import { pickAnchor, pickNext, progress, toCsv, validateRating } from "../src/logic.js";

const APPS = ["a001", "a002", "a003"];

test("pickNext prefers the least-rated app the rater has not rated", () => {
  assert.equal(pickNext(APPS, { a001: 2, a002: 0, a003: 1 }, new Set()), "a002");
  assert.equal(pickNext(APPS, { a001: 2, a002: 0, a003: 1 }, new Set(["a002"])), "a003");
  assert.equal(pickNext(APPS, {}, new Set(APPS)), null);
  const seen = new Set();
  for (const r of [0, 0.5, 0.99]) seen.add(pickNext(APPS, {}, new Set(), () => r));
  assert.deepEqual([...seen].sort(), APPS);
});

test("every app reaches the target before any gets a fourth rating", () => {
  const counts = {};
  const raters = Array.from({ length: 12 }, () => new Set());
  for (let round = 0; round < 3; round++) {
    for (const rated of raters) {
      const a = pickNext(APPS, counts, rated);
      if (a) {
        rated.add(a);
        counts[a] = (counts[a] || 0) + 1;
      }
    }
  }
  assert.ok(Math.max(...Object.values(counts)) - Math.min(...Object.values(counts)) <= 1);
  assert.deepEqual(progress(APPS, { a001: 5, a002: 1 }), { ratings_toward_target: 4, target: 9, apps_done: 1, apps: 3 });
});

test("validateRating", () => {
  const ok = { app: "a001", rater: "r-12345678", usefulness: 3, novelty: 5, works: "partly", seconds: 61.2, lang: "de" };
  assert.equal(validateRating(ok, APPS), null);
  assert.equal(validateRating({ ...ok, app: "x" }, APPS), "app");
  assert.equal(validateRating({ ...ok, rater: "short" }, APPS), "rater");
  assert.equal(validateRating({ ...ok, usefulness: 6 }, APPS), "usefulness");
  assert.equal(validateRating({ ...ok, novelty: 2.5 }, APPS), "novelty");
  assert.equal(validateRating({ ...ok, works: "maybe" }, APPS), "works");
  assert.equal(validateRating({ ...ok, lang: "fr" }, APPS), "lang");
});

test("toCsv quotes", () => {
  assert.equal(toCsv([{ id: 1, app: "a,1" }]).split("\n")[1], '1,"a,1",,,,,,,');
});

test("anchors come first, each once", () => {
  const K = ["k1", "k2", "k3"];
  const rated = new Set();
  for (let i = 0; i < 3; i++) {
    const a = pickAnchor(K, rated);
    assert.ok(K.includes(a) && !rated.has(a));
    rated.add(a);
  }
  assert.equal(pickAnchor(K, rated), null);
});
