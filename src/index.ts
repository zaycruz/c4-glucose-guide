// C4 Glucose Guide: meals, meal timing and movement vs blood sugar.
// Serves precomputed analysis (scripts/analyze.py -> public/data) under /api/c4.
// Workers AI may rephrase advice; every number it writes must already be in the evidence.

interface Env {
  ASSETS: Fetcher;
  AI: Ai;
}

type Evidence = { id: string; value: number | null; ci: [number | null, number | null] | null; n: number | null };
type Rec = { id: string; title: string; text: string; basis: string; evidenceIds: string[] };
type Detail = { survivorId: string; evidence: Evidence[]; recommendations: Rec[]; [k: string]: unknown };

const AI_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const ID_RE = /^c4-N\d{4}$/;
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, OPTIONS",
  "access-control-allow-headers": "content-type",
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "public, max-age=60", ...CORS },
  });

async function asset(env: Env, req: Request, path: string): Promise<unknown | null> {
  const res = await env.ASSETS.fetch(new URL(path, req.url));
  return res.ok ? res.json() : null;
}

/** Every number in a text, as absolute values (signs are worded as "less"/"more"). */
const numbers = (text: string) => (text.match(/\d+(?:\.\d+)?/g) ?? []).map((n) => Math.abs(Number(n)));

/** Numbers an AI rewrite may use: those in the template text and in its evidence. */
function allowedNumbers(rec: Rec, evidence: Map<string, Evidence>) {
  const allowed = new Set(numbers(rec.text));
  for (const id of rec.evidenceIds) {
    const e = evidence.get(id);
    if (!e) continue;
    for (const v of [e.value, e.n, ...(e.ci ?? [])]) if (typeof v === "number") allowed.add(Math.abs(v));
  }
  return allowed;
}

function untraced(text: string, allowed: Set<number>) {
  return numbers(text).filter((n) => ![...allowed].some((a) => Math.abs(a - n) < 0.051));
}

async function phrase(env: Env, detail: Detail) {
  const evidence = new Map(detail.evidence.map((e) => [e.id, e]));
  const started = Date.now();
  let raw: unknown = null;
  let error: string | null = null;
  try {
    raw = await env.AI.run(AI_MODEL as keyof AiModels, {
      messages: [
        {
          role: "system",
          content:
            "You rewrite health-coaching advice for a cancer survivor in warm, plain English (reading age 12). " +
            "Keep each item to two short sentences. Use ONLY numbers that appear in the given text; never add, change or round numbers. " +
            "Do not add medical claims. Answer JSON: {\"items\":[{\"id\":string,\"text\":string}]}.",
        },
        { role: "user", content: JSON.stringify(detail.recommendations.map((r) => ({ id: r.id, text: r.text }))) },
      ],
      max_tokens: 1200,
      temperature: 0.3,
    } as never);
  } catch (e) {
    error = String(e);
  }
  const content = (() => {
    const r = raw as { choices?: { message?: { content?: unknown } }[]; response?: unknown } | null;
    const c = r?.choices?.[0]?.message?.content ?? r?.response;
    if (typeof c === "object" && c) return c as { items?: unknown };
    if (typeof c !== "string") return null;
    try {
      return JSON.parse(c.slice(c.indexOf("{"), c.lastIndexOf("}") + 1)) as { items?: unknown };
    } catch {
      return null;
    }
  })();
  const aiItems = new Map<string, string>();
  if (content && Array.isArray(content.items))
    for (const it of content.items as { id?: unknown; text?: unknown }[])
      if (typeof it.id === "string" && typeof it.text === "string") aiItems.set(it.id, it.text.trim());

  const items = detail.recommendations.map((rec) => {
    const ai = aiItems.get(rec.id);
    if (!ai) return { id: rec.id, title: rec.title, text: rec.text, source: "template", evidenceIds: rec.evidenceIds, rejected: null };
    const bad = untraced(ai, allowedNumbers(rec, evidence));
    return bad.length
      ? { id: rec.id, title: rec.title, text: rec.text, source: "template", evidenceIds: rec.evidenceIds, rejected: { text: ai, untracedNumbers: bad } }
      : { id: rec.id, title: rec.title, text: ai, source: "ai", evidenceIds: rec.evidenceIds, rejected: null };
  });
  const rawPreview = aiItems.size ? null : JSON.stringify(raw)?.slice(0, 600) ?? null;
  return { survivorId: detail.survivorId, model: AI_MODEL, ms: Date.now() - started, error, rawPreview, items };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    if (!path.startsWith("/api/c4")) return env.ASSETS.fetch(req);
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
    if (req.method !== "GET") return json({ error: "method not allowed" }, 405);

    if (path === "/api/c4/survivors") return json(await asset(env, req, "/data/roster.json"));
    if (path === "/api/c4/cohort") return json(await asset(env, req, "/data/cohort.json"));

    const m = path.match(/^\/api\/c4\/survivors\/([^/]+)(\/advice)?$/);
    if (m) {
      const id = decodeURIComponent(m[1]);
      if (!ID_RE.test(id)) return json({ error: "unknown survivorId", survivorId: id }, 404);
      const detail = (await asset(env, req, `/data/survivors/${id}.json`)) as Detail | null;
      if (!detail) return json({ error: "unknown survivorId", survivorId: id }, 404);
      return m[2] ? json(await phrase(env, detail)) : json(detail);
    }
    return json({ error: "not found", routes: ["/api/c4/survivors", "/api/c4/survivors/:id", "/api/c4/survivors/:id/advice", "/api/c4/cohort"] }, 404);
  },
} satisfies ExportedHandler<Env>;
