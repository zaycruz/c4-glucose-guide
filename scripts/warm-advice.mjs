// Precompute the AI-rephrased advice for every survivor (Workers AI, guarded), so the API serves it instantly.
// Writes public/data/advice/<id>.json with the recommendation hash the Worker checks; redeploy afterwards.
// Usage: node scripts/warm-advice.mjs [baseUrl]
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const BASE = (process.argv[2] ?? "https://c4-glucose-guide.zay-259.workers.dev").replace(/\/$/, "");
const roster = await (await fetch(`${BASE}/api/c4/survivors`)).json();
mkdirSync("public/data/advice", { recursive: true });
let ai = 0, template = 0;
for (const { survivorId } of roster) {
  let body = null;
  for (let attempt = 0; attempt < 3 && !body; attempt++) {
    const res = await fetch(`${BASE}/api/c4/survivors/${survivorId}/advice?fresh=1`);
    const b = await res.json();
    if (res.ok && Array.isArray(b.items) && !b.error) body = b;
    else await new Promise((r) => setTimeout(r, 4000 * (attempt + 1))); // rate limit: back off
  }
  if (!body) { console.error(`${survivorId}: no AI rewrite, the Worker will use the live path`); continue; }
  // The rewrite must be of the committed recommendations; a just-deployed Worker can still serve the old detail (60 s cache).
  const local = JSON.parse(readFileSync(`public/data/survivors/${survivorId}.json`, "utf8"));
  const want = createHash("sha256").update(JSON.stringify(local.recommendations.map((r) => [r.id, r.text]))).digest("hex");
  if (body.recHash !== want) { console.error(`${survivorId}: live advice is for stale recommendations; wait a minute after deploying and rerun`); process.exit(1); }
  const { cached, ms, rawPreview, ...keep } = body;
  writeFileSync(`public/data/advice/${survivorId}.json`, JSON.stringify({ ...keep, generatedAt: new Date().toISOString(), generationMs: ms }));
  for (const it of body.items) it.source === "ai" ? ai++ : template++;
  await new Promise((r) => setTimeout(r, 1500));
}
console.log(JSON.stringify({ survivors: roster.length, itemsAi: ai, itemsTemplate: template }));
