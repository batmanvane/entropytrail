// Pure helpers of the ratings Worker (tested with node --test).

export const TARGET = 3; // ratings per app (METHODS.md, D-043)

const RATER = /^[a-z0-9-]{8,64}$/;

// Next app for a rater: among apps the rater has not rated, the ones with the fewest ratings;
// random choice among those. null when the rater has rated every app.
export function pickNext(apps, counts, rated, rand = Math.random) {
  const open = apps.filter((a) => !rated.has(a));
  if (open.length === 0) return null;
  const min = Math.min(...open.map((a) => counts[a] || 0));
  const best = open.filter((a) => (counts[a] || 0) === min);
  return best[Math.floor(rand() * best.length)];
}

export function validateRating(body, apps) {
  if (!body || typeof body !== "object") return "body";
  const { app, rater, usefulness, novelty, works, seconds, lang } = body;
  if (!apps.includes(app)) return "app";
  if (typeof rater !== "string" || !RATER.test(rater)) return "rater";
  for (const [k, v] of [["usefulness", usefulness], ["novelty", novelty]]) {
    if (!Number.isInteger(v) || v < 1 || v > 5) return k;
  }
  if (!["yes", "partly", "no"].includes(works)) return "works";
  if (seconds != null && (typeof seconds !== "number" || !(seconds >= 0) || seconds > 86400)) return "seconds";
  if (lang != null && !["en", "de"].includes(lang)) return "lang";
  return null;
}

export function progress(apps, counts) {
  let total = 0;
  let done = 0;
  for (const a of apps) {
    const c = counts[a] || 0;
    total += Math.min(c, TARGET);
    if (c >= TARGET) done += 1;
  }
  return { ratings_toward_target: total, target: apps.length * TARGET, apps_done: done, apps: apps.length };
}

export function toCsv(rows) {
  const cols = ["id", "app", "rater", "usefulness", "novelty", "works", "seconds", "lang", "created_at"];
  const esc = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n") + "\n";
}
