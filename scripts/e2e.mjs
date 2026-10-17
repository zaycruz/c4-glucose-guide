// E2E check: hits every endpoint of a deployed (or local) Worker, asserts the contract
// and the "every number traces to evidence" rule, and saves the responses under runs/.
// Usage: node scripts/e2e.mjs [baseUrl]
import { mkdirSync, writeFileSync } from "node:fs";

const BASE = (process.argv[2] ?? "https://c4-glucose-guide.zay-259.workers.dev").replace(/\/$/, "");
const STATUS = ["alert", "watch", "ok"];
// Fixed numbers in advice wording that are rules, not measurements: 10 to 20 min walk, 8 pm,
// 13-hour fast, 70-180 range, 70% target, walk bins 10-19 / 20-29 / 30+, weeks 1-2 / 11-12,
// dinner clock times 6 pm / 8-9 pm, and the 95% confidence level.
const RULE_NUMBERS = new Set([1, 2, 6, 8, 9, 10, 11, 12, 13, 19, 20, 29, 30, 70, 95, 180]);

const failures = [];
const check = (ok, msg) => { if (!ok) failures.push(msg); return ok; };
const nums = (t) => (t.match(/\d+(?:\.\d+)?/g) ?? []).map((n) => Math.abs(Number(n)));
const traced = (n, ev) =>
  ev.some((e) => [e.value, e.n, ...(e.ci ?? [])].some((v) => typeof v === "number" && Math.abs(Math.abs(v) - n) < 0.051));

async function get(path, init) {
  const t = Date.now();
  const res = await fetch(BASE + path, init);
  const text = await res.text();
  let body = null;
  try { body = JSON.parse(text); } catch { body = text.slice(0, 200); }
  return { path, status: res.status, ms: Date.now() - t, contentType: res.headers.get("content-type"), cors: res.headers.get("access-control-allow-origin"), body };
}

const run = { base: BASE, startedAt: new Date().toISOString(), responses: {} };

const home = await get("/");
check(home.status === 200 && home.contentType?.includes("text/html"), "GET / is not an HTML 200");
check(typeof home.body === "string" && home.body.includes("Glucose Guide"), "GET / is not the demo page");
run.responses["/"] = { status: home.status, contentType: home.contentType };

const roster = await get("/api/c4/survivors");
run.responses["/api/c4/survivors"] = roster;
check(roster.status === 200 && Array.isArray(roster.body) && roster.body.length > 0, "roster is not a non-empty array");
check(roster.cors === "*", "roster has no CORS header");
const rank = (s) => STATUS.indexOf(s);
roster.body.forEach((s, i) => {
  check(typeof s.survivorId === "string" && /^c4-/.test(s.survivorId), `roster[${i}] bad survivorId`);
  for (const k of ["displayName", "headline", "updatedAt"]) check(typeof s[k] === "string" && s[k].length > 0, `roster[${i}].${k} missing`);
  check(STATUS.includes(s.status), `roster[${i}] bad status ${s.status}`);
  if (i) check(rank(roster.body[i - 1].status) <= rank(s.status), `roster not sorted by status at ${i}`);
});

let recCount = 0, numberCount = 0, ruleNumberCount = 0;
for (const tile of roster.body) {
  const d = await get(`/api/c4/survivors/${tile.survivorId}`);
  run.responses[`/api/c4/survivors/${tile.survivorId}`] = d;
  const b = d.body, id = tile.survivorId;
  if (!check(d.status === 200 && b && typeof b === "object", `${id}: detail not 200`)) continue;
  check(b.survivorId === id, `${id}: survivorId mismatch`);
  check(b.status === tile.status, `${id}: status ${b.status} differs from roster ${tile.status}`);
  for (const k of ["headline", "summary"]) check(typeof b[k] === "string" && b[k].length > 0, `${id}: ${k} missing`);
  check(Array.isArray(b.evidence) && b.evidence.length > 0, `${id}: evidence empty`);
  check(Array.isArray(b.recommendations) && b.recommendations.length > 0, `${id}: recommendations empty`);
  check(b.charts && typeof b.charts === "object", `${id}: charts missing`);
  for (const [name, c] of Object.entries(b.charts ?? {}))
    check(Object.values(c).some((v) => Array.isArray(v)), `${id}: chart ${name} has no array series`);
  const ev = new Map(b.evidence.map((e) => [e.id, e]));
  check(ev.size === b.evidence.length, `${id}: duplicate evidence ids`);
  for (const r of b.recommendations) {
    recCount++;
    const refs = r.evidenceIds.map((x) => ev.get(x));
    check(refs.every(Boolean), `${id}/${r.id}: evidenceId does not resolve`);
    for (const n of nums(r.text)) {
      numberCount++;
      if (traced(n, refs.filter(Boolean))) continue;
      if (RULE_NUMBERS.has(n)) { ruleNumberCount++; continue; }
      check(false, `${id}/${r.id}: number ${n} in advice is not in its evidence`);
    }
  }
}

const cohort = await get("/api/c4/cohort");
run.responses["/api/c4/cohort"] = cohort;
check(cohort.status === 200 && cohort.body?.effects?.walk && cohort.body?.trial, "cohort endpoint incomplete");

const alertId = roster.body.find((s) => s.status === "alert")?.survivorId ?? roster.body[0].survivorId;
const advice = await get(`/api/c4/survivors/${alertId}/advice`);
run.responses[`/api/c4/survivors/${alertId}/advice`] = advice;
check(advice.status === 200 && Array.isArray(advice.body?.items), "advice endpoint failed");
const detailEv = run.responses[`/api/c4/survivors/${alertId}`].body;
for (const it of advice.body?.items ?? []) {
  const rec = detailEv.recommendations.find((r) => r.id === it.id);
  const refs = it.evidenceIds.map((x) => detailEv.evidence.find((e) => e.id === x)).filter(Boolean);
  const allowed = new Set(nums(rec.text));
  for (const n of nums(it.text)) check(allowed.has(n) || traced(n, refs), `advice ${it.id}: AI number ${n} not traced`);
}
const aiUsed = (advice.body?.items ?? []).filter((i) => i.source === "ai").length;

const missing = await get("/api/c4/survivors/c4-N9999");
check(missing.status === 404, "unknown survivor is not 404");
const bad = await get("/api/c4/survivors/..%2Fdata");
check(bad.status === 404, "malformed id is not 404");
const pre = await get("/api/c4/survivors", { method: "OPTIONS" });
check(pre.status === 204 && pre.cors === "*", "CORS preflight failed");
run.responses.negative = { missing: missing.status, malformed: bad.status, preflight: pre.status };

const counts = Object.fromEntries(STATUS.map((s) => [s, roster.body.filter((x) => x.status === s).length]));
run.summary = {
  pass: failures.length === 0, failures,
  survivors: roster.body.length, statusCounts: counts, recommendations: recCount,
  adviceNumbers: numberCount, numbersTracedToEvidence: numberCount - ruleNumberCount, ruleNumbers: ruleNumberCount,
  aiAdvice: { survivorId: alertId, model: advice.body?.model, ms: advice.body?.ms, itemsRephrasedByAi: aiUsed, items: advice.body?.items?.length },
  finishedAt: new Date().toISOString(),
};

mkdirSync("runs", { recursive: true });
const stamp = run.startedAt.replace(/[:.]/g, "-");
writeFileSync(`runs/e2e-${stamp}.json`, JSON.stringify(run, null, 1));
writeFileSync("runs/e2e-latest-summary.json", JSON.stringify({ base: BASE, file: `runs/e2e-${stamp}.json`, ...run.summary }, null, 1));
console.log(JSON.stringify(run.summary, null, 1));
process.exit(failures.length ? 1 : 0);
