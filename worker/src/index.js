// Ratings Worker for the human rating session (D-043).
//   GET  /next?rater=<id>   next app for this rater (fewest ratings first) + study progress
//   POST /rate              {app, rater, usefulness, novelty, works, seconds, lang}
//   GET  /progress          study progress
//   GET  /export            all ratings as CSV (Authorization: Bearer <EXPORT_KEY>)
// No IP addresses or other personal data are stored.
import APPS from "./apps.json";
import { pickNext, progress, toCsv, validateRating } from "./logic.js";

const ORIGINS = ["https://batmanvane.github.io", "http://127.0.0.1:8772", "http://localhost:8772"];

function cors(req) {
  const o = req.headers.get("Origin");
  return {
    "Access-Control-Allow-Origin": ORIGINS.includes(o) ? o : ORIGINS[0],
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    Vary: "Origin",
  };
}

function json(req, data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...cors(req) } });
}

async function counts(db) {
  const { results } = await db.prepare("SELECT app, COUNT(*) AS n FROM ratings GROUP BY app").all();
  return Object.fromEntries(results.map((r) => [r.app, r.n]));
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const db = env.DB;
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(req) });
    try {
      if (req.method === "GET" && url.pathname === "/progress") {
        return json(req, progress(APPS, await counts(db)));
      }
      if (req.method === "GET" && url.pathname === "/next") {
        const rater = url.searchParams.get("rater") || "";
        if (!/^[a-z0-9-]{8,64}$/.test(rater)) return json(req, { error: "rater" }, 400);
        const c = await counts(db);
        const { results } = await db.prepare("SELECT app FROM ratings WHERE rater = ?").bind(rater).all();
        const rated = new Set(results.map((r) => r.app));
        return json(req, { app: pickNext(APPS, c, rated), rated_by_you: rated.size, progress: progress(APPS, c) });
      }
      if (req.method === "POST" && url.pathname === "/rate") {
        let body;
        try {
          body = await req.json();
        } catch {
          return json(req, { error: "json" }, 400);
        }
        const bad = validateRating(body, APPS);
        if (bad) return json(req, { error: bad }, 400);
        const r = await db
          .prepare(
            "INSERT OR IGNORE INTO ratings (app, rater, usefulness, novelty, works, seconds, lang, created_at) " +
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
          )
          .bind(body.app, body.rater, body.usefulness, body.novelty, body.works, body.seconds ?? null,
                body.lang ?? null, new Date().toISOString())
          .run();
        return json(req, { saved: r.meta.changes === 1 });
      }
      if (req.method === "GET" && url.pathname === "/export") {
        if (!env.EXPORT_KEY || req.headers.get("Authorization") !== `Bearer ${env.EXPORT_KEY}`) {
          return new Response("forbidden", { status: 403 });
        }
        const { results } = await db.prepare("SELECT * FROM ratings ORDER BY id").all();
        return new Response(toCsv(results), { headers: { "Content-Type": "text/csv; charset=utf-8" } });
      }
      return json(req, { error: "not found" }, 404);
    } catch (e) {
      return json(req, { error: "server" }, 500);
    }
  },
};
