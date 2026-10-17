// E2E check: hits every endpoint of a deployed (or local) Worker, asserts the contract
// and the "every number traces to evidence" rule, and saves the responses under runs/.
// Usage: node scripts/e2e.mjs [baseUrl]
import { createHash } from "node:crypto";
import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const STRICT = args.includes("--strict"); // also require cached advice and a verification record that matches the live data
const BASE = (args.find((a) => !a.startsWith("--")) ?? "https://c4-glucose-guide.zay-259.workers.dev").replace(/\/$/, "");
const PATIENT_NUMBERS = new Set([4, 8, 10, 13, 15, 20, 211, 911]); // fixed rule numbers allowed in patient copy
const STATUS = ["alert", "watch", "ok"];
// Exact wording constants (rules and labels, not measurements). They are removed before
// number extraction; every number left must trace to the cited evidence.
const CONSTANT_PHRASES = [
  /\b8-9 pm\b/g, /\bbefore 6 pm\b/g, /\b8 pm\b/g, /\b13-hour\b/g, /\b13 hours\b/g, /\b70-180\b/g, /95% CI\b/g,
  /\bweeks 1-2\b/g, /\bweeks 11-12\b/g, /\b10 to 20 minutes?\b/g, /\b10 to 19 minutes?\b/g, /\b20-29 minutes\b/g,
  /\b30\+ minutes\b/g, /under 4%/g, /under 5%/g, /more than 70%/g, /below 70 mg\/dL/g, /below 70%/g, /below 90%/g,
  /below 54( mg\/dL)?/g, /under 1%/g, /more than 1% (are )?below 54/g, /\bHbA1c\b/g, /\btype 2 diabetes\b/g, /more than 5% of readings above (180|250)/g, /\babove 250\b/g, /more than 4% of readings below 70/g, /up 10\+ mg\/dL/g,
  /readings below 70\b/g, /Spikes above 180\b/g, // headline thresholds
];
const CAUSAL = /\b(caus\w*|help(s|ed)?|lower(s|ed)?|reduc\w*|prevent\w*|cur(e|es|ed))\b/gi;
const strip = (t) => CONSTANT_PHRASES.reduce((acc, re) => acc.replace(re, " "), t);

const failures = [];
const personEvOf = (b) => b.evidence.filter((e) => e.scope === "person");
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

let recCount = 0, numberCount = 0, constantsStripped = 0;
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
    const rdc = r.realDataCheck;
    if (check(rdc && ["agrees", "same direction, not proven", "no clear signal", "disagrees", "not tested"].includes(rdc.label), `${id}/${r.id}: realDataCheck missing or bad label`)) {
      if (rdc.label === "not tested") check(rdc.evidenceId === null, `${id}/${r.id}: untested check cites evidence`);
      else check(r.evidenceIds.includes(rdc.evidenceId) && ev.get(rdc.evidenceId)?.scope === "real", `${id}/${r.id}: realDataCheck evidence not cited or not real-data`);
    }
    constantsStripped += nums(r.text).length - nums(strip(r.text)).length;
    for (const n of nums(strip(r.text))) {
      numberCount++;
      check(traced(n, refs.filter(Boolean)), `${id}/${r.id}: number ${n} in advice is not in its evidence`);
    }
  }
  check(Array.isArray(b.tags) && JSON.stringify(b.tags) === JSON.stringify(tile.tags), `${id}: tags missing or differ from roster`);
  for (const t of b.tags ?? []) {
    check(["risk", "warn", "good", "info"].includes(t.tone), `${id}: bad tag tone ${t.tone}`);
    const e = ev.get(t.evidenceId);
    if (!check(Boolean(e), `${id}: tag ${t.label} evidenceId does not resolve`)) continue;
    const label = t.evidenceId === "walk_meals" ? t.label.replace(/\/\d+/, "") : t.label; // "Walks after 2/37 meals": 37 = meals_scored
    for (const n of nums(label)) check([e.value, e.n].some((v) => typeof v === "number" && Math.abs(Math.abs(v) - n) <= 0.5), `${id}: tag ${t.label} number ${n} not in evidence`);
  }
  // safety routing: frequent lows (>4% below 70 or >1% below 54), average above 250 or sustained highs (>5% above 250) -> care team today
  const tbr = ev.get("tbr70")?.value, tbr54 = ev.get("tbr54")?.value, mean = ev.get("mean_glucose")?.value, tar = ev.get("tar250")?.value;
  const lows = (typeof tbr === "number" && tbr > 4) || (typeof tbr54 === "number" && tbr54 > 1), high = (typeof mean === "number" && mean > 250) || (typeof tar === "number" && tar > 5);
  const cf = b.careTeamFlag;
  if (check(cf && typeof cf.today === "boolean" && Array.isArray(cf.reasons), `${id}: careTeamFlag missing`)) {
    check(cf.today === (lows || high), `${id}: careTeamFlag.today=${cf.today} but lows=${lows} high=${high}`);
    // urgency: only level 2 lows (>1% below 54) are same-day; every other care-team flag is this week
    const urgency = cf.today ? (typeof tbr54 === "number" && tbr54 > 1 ? "today" : "this-week") : null;
    check(cf.urgency === urgency && tile.careTeamUrgency === urgency, `${id}: urgency ${cf.urgency} (roster ${tile.careTeamUrgency}), expected ${urgency}`);
    check(cf.level === (urgency === "today" ? "today" : urgency ? "this week" : b.status === "alert" ? "next visit" : null), `${id}: careTeamFlag.level=${cf.level} does not match urgency/status`);
    check((cf.level !== null) === cf.reasons.length > 0, `${id}: careTeamFlag reasons do not match the level`);
    const word = urgency === "today" ? /\btoday\b/ : /\bthis week\b/;
    if (cf.today) check(b.recommendations[0]?.id === "talk_to_team" && word.test(b.recommendations[0].title) && word.test(b.recommendations[0].text)
      && (urgency === "today" || !/\btoday\b/i.test(b.recommendations[0].title + " " + b.recommendations[0].text)), `${id}: first card does not match urgency ${urgency}`);
    for (const rs of cf.reasons) check(ev.has(rs.evidenceId), `${id}: care reason evidence ${rs.evidenceId} missing`);
    check(tile.careTeamToday === cf.today, `${id}: roster careTeamToday differs from detail`);
  }
  check(lows === (b.hypoglycaemiaWarning != null), `${id}: hypoglycaemiaWarning presence does not match frequent lows`);
  check(typeof tbr54 === "number", `${id}: no below-54 evidence`);
  for (const r of b.recommendations.filter((x) => x.id === "walk")) // medicines are not recorded: every walk card asks about insulin / sulfonylureas
    check(/sulfonylurea/.test(r.text) && /does not change medication/.test(r.medicationCaution ?? ""), `${id}: walk advice has no medication caution`);
  if (lows) { // care team first: no exercise, fasting or meal-timing nudges, and a medication question
    check(b.recommendations.every((r) => r.id === "talk_to_team"), `${id}: frequent lows but given ${b.recommendations.map((r) => r.id)} (care team only)`);
    check(/medicine/.test(b.recommendations[0]?.text ?? ""), `${id}: lows card does not ask about medicines`);
    check(cf?.medicationCheck?.recorded === false && /sulfonylurea/.test(cf.medicationCheck.ask), `${id}: lows but no medicationCheck`);
    check((b.tags ?? []).every((t) => ["risk", "info"].includes(t.tone)), `${id}: lows but behaviour tags shown`);
  } else check(cf?.medicationCheck == null, `${id}: medicationCheck without lows`);
  check(typeof b.todayReason === "string" && b.todayReason.length <= 90 && b.todayReason === tile.todayReason, `${id}: todayReason missing, too long or differs from roster`);
  check(b.priority === tile.priority && Number.isInteger(b.priority), `${id}: priority missing or differs`);
  // patient copy: three languages, same actions, fixed rule numbers only, care-team line when flagged, help route
  const pv = b.patientView;
  if (check(pv && ["en", "es", "ht"].every((l) => pv[l]), `${id}: patientView missing a language`)) {
    const ids = JSON.stringify(pv.en.actions.map((a) => a.id));
    for (const l of ["en", "es", "ht"]) {
      const v = pv[l];
      check(JSON.stringify(v.actions.map((a) => a.id)) === ids && v.actions.length >= (lows ? 0 : 1) && v.actions.length <= (lows ? 0 : 3), `${id}: patientView.${l} actions differ`);
      check(/211/.test(v.help), `${id}: patientView.${l} has no help route`);
      check(v.reviewed === (l === "en") && (l === "en" || /machine-translated/.test(v.translation)), `${id}: patientView.${l} translation status missing`);
      check(!cf?.today || Boolean(v.careTeam), `${id}: flagged but patientView.${l} has no care-team line`);
      if (cf?.today) check(/^(Today|Hoy|Jodi a):/.test(v.careTeam) === (cf.urgency === "today"), `${id}: patientView.${l} care-team line does not match urgency`);
      check(lows === Boolean(v.lowSugarWarning), `${id}: patientView.${l} low-sugar warning mismatch`);
      for (const t of [v.careTeam, v.lowSugarWarning, v.help, ...v.actions.map((a) => a.text)].filter(Boolean))
        for (const n of nums(t)) check(PATIENT_NUMBERS.has(n), `${id}: patientView.${l} has measured number ${n}`);
    }
  }
  check(typeof b.plainSummary === "string" && b.plainSummary.length > 0, `${id}: plainSummary missing`);
  for (const n of nums(b.plainSummary ?? "")) check(PATIENT_NUMBERS.has(n), `${id}: plainSummary has measured number ${n}`);
  // todayReason numbers are person evidence (or rule thresholds)
  for (const n of nums(strip((b.todayReason ?? "").replace(/target under [45]%|above 250|below 70/g, " "))))
    check(personEvOf(b).some((e) => typeof e.value === "number" && Math.abs(Math.abs(e.value) - n) <= 0.5), `${id}: todayReason number ${n} not in evidence`);
  // headline and summary numbers are rounded person-level evidence values (+-0.5)
  const personEv = b.evidence.filter((e) => e.scope === "person");
  for (const field of ["headline", "summary"]) {
    const text = strip(b[field].replace(/\d+-year-old|[\d.]+ years after treatment|\(made-up data\)/g, " "));
    for (const n of nums(text))
      check(personEv.some((e) => typeof e.value === "number" && Math.abs(Math.abs(e.value) - n) <= 0.5), `${id}: ${field} number ${n} not in evidence`);
  }
}

const cohort = await get("/api/c4/cohort");
run.responses["/api/c4/cohort"] = cohort;
check(cohort.status === 200 && cohort.body?.effects?.walk && cohort.body?.trial, "cohort endpoint incomplete");
const C = cohort.body ?? {};
for (const p of ["/api/c4", "/api/c4/"]) {
  const r = await fetch(BASE + p);
  check(r.status === 200 && Array.isArray((await r.json()).routes), `${p} index missing`);
}
const pl = C.plainLanguage;
if (check(pl && Array.isArray(pl.points) && pl.points.length === 3, "cohort plainLanguage missing")) {
  const rcw = C.robustness?.realDataChecks?.walk?.real;
  const vals = [C.effects.lateDinner, C.effects.walkAfterDinner, ...(C.walkDose ?? []), ...(rcw ? [rcw] : [])].flatMap((e) => [e.est, e.lo, e.hi]).concat(rcw ? [rcw.people] : []);
  for (const t of [pl.headline, ...pl.points.map((p) => p.text)])
    for (const n of nums(strip(t.replace(/10 minutes|95% CI/g, " ").replace(/\b(10-19|20-29|30\+?) min\b/g, " "))))
      check(vals.some((v) => typeof v === "number" && Math.abs(Math.abs(v) - n) <= 0.5), `plainLanguage number ${n} not in cohort effects`);
  const rc = C.robustness?.realDataChecks;
  for (const [k, word] of [["walk", "walk effect was"], ["lateDinner", "late-dinner effect"]]) {
    const ratio = rc?.[k]?.real?.est / rc?.[k]?.program;
    const ok = { "about a fifth of": [0.15, 0.22], "about a quarter of": [0.22, 0.3], "about a third of": [0.3, 0.4], "about half of": [0.4, 0.6] };
    const said = Object.keys(ok).find((w) => pl.realData?.includes(`${word} ${w}`) || pl.realData?.includes(`${word} ${w}`.replace(" was", "")));
    if (said) check(ratio >= ok[said][0] && ratio < ok[said][1], `plainLanguage says ${k} is ${said} but the ratio is ${ratio}`);
  }
}
const fair = C.fairness?.attributes;
// fairness is served only once the engine run includes it; when present it must be complete
if (C.fairness !== undefined && check(fair && Object.keys(fair).length >= 4, "cohort fairness table incomplete")) {
  for (const [attr, t] of Object.entries(fair))
    for (const g of t.groups) {
      check(typeof g.forecastMae?.est === "number", `fairness ${attr}/${g.group}: no forecast error`);
      if (g.people < 20) check(g.walk === null && /fewer than 20/.test(g.note ?? ""), `fairness ${attr}/${g.group}: small group without a note`);
      else check(g.walk === null ? Boolean(g.note) : typeof g.walk.est === "number", `fairness ${attr}/${g.group}: walk effect missing`);
    }
}
check(C.careTeamToday?.count === roster.body.filter((x) => x.careTeamToday).length, "cohort careTeamToday count differs from roster");
check(C.careTeamToday?.today === roster.body.filter((x) => x.careTeamUrgency === "today").length && C.careTeamToday?.thisWeek === roster.body.filter((x) => x.careTeamUrgency === "this-week").length, "cohort urgency counts differ from roster");

const alertId = roster.body.find((s) => s.status === "alert")?.survivorId ?? roster.body[0].survivorId;
const advice = await get(`/api/c4/survivors/${alertId}/advice`);
run.responses[`/api/c4/survivors/${alertId}/advice`] = advice;
check(advice.status === 200 && Array.isArray(advice.body?.items), "advice endpoint failed");
check(!advice.body?.error || /unavailable/.test(advice.body.error), "advice passes a raw model error through");
check(advice.body?.model === "@cf/meta/llama-3.3-70b-instruct-fp8-fast" && advice.body?.fallbackModel === null && !("servedModel" in (advice.body ?? {}))
  && /no fallback model/.test(advice.body?.modelNote ?? ""), "advice model fields do not state the requested model and the no-fallback rule");
check(!("rawPreview" in (advice.body ?? {})) && !JSON.stringify(advice.body ?? {}).includes("chatcmpl"), "advice exposes the raw provider response");
if (STRICT) check(advice.body?.cached === true && advice.ms < 2000, `advice not served from cache (cached=${advice.body?.cached}, ${advice.ms} ms)`);
const detailEv = run.responses[`/api/c4/survivors/${alertId}`].body;
for (const it of advice.body?.items ?? []) {
  const rec = detailEv.recommendations.find((r) => r.id === it.id);
  const refs = it.evidenceIds.map((x) => detailEv.evidence.find((e) => e.id === x)).filter(Boolean);
  const allowed = new Set(nums(rec.text));
  for (const n of nums(strip(it.text))) check(allowed.has(n) || traced(n, refs), `advice ${it.id}: AI number ${n} not traced`);
  if (it.source === "ai") {
    const added = (it.text.match(CAUSAL) ?? []).filter((w) => !new RegExp(`\\b${w}\\b`, "i").test(rec.text));
    check(added.length === 0, `advice ${it.id}: AI added causal wording ${added.join(",")}`);
  }
}
const aiUsed = (advice.body?.items ?? []).filter((i) => i.source === "ai").length;

const missing = await get("/api/c4/survivors/c4-N9999");
check(missing.status === 404, "unknown survivor is not 404");
const bad = await get("/api/c4/survivors/..%2Fdata");
check(bad.status === 404, "malformed id is not 404");
const pre = await get("/api/c4/survivors", { method: "OPTIONS" });
check(pre.status === 204 && pre.cors === "*", "CORS preflight failed");
run.responses.negative = { missing: missing.status, malformed: bad.status, preflight: pre.status };

const dataHash = createHash("sha256").update(JSON.stringify([roster.body, cohort.body])).digest("hex");
const ver = await get("/api/c4/verification");
const verFresh = ver.status === 200 && ver.body?.dataHash === dataHash;
if (STRICT) check(verFresh, "published verification record does not match the live data");
// The flagged count must agree in the README, the live roster and the published verification record.
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"];
const flaggedLive = roster.body.filter((x) => x.careTeamToday).length;
const readmeM = readFileSync(new URL("../README.md", import.meta.url), "utf8").match(/(\w+) of the (\d+) survivors are flagged for the care team/i);
const readmeN = readmeM ? (/^\d+$/.test(readmeM[1]) ? Number(readmeM[1]) : WORDS.indexOf(readmeM[1].toLowerCase())) : null;
check(readmeN === flaggedLive && Number(readmeM?.[2]) === roster.body.length, `README says ${readmeM?.[0]} but the live roster flags ${flaggedLive} of ${roster.body.length}`);
if (STRICT) check(ver.body?.e2e?.careTeamToday === flaggedLive, `published verification says ${ver.body?.e2e?.careTeamToday} flagged, live roster ${flaggedLive}`);
const counts = Object.fromEntries(STATUS.map((s) => [s, roster.body.filter((x) => x.status === s).length]));
run.summary = {
  pass: failures.length === 0, failures,
  survivors: roster.body.length, statusCounts: counts, recommendations: recCount,
  adviceNumbersTracedToEvidence: numberCount, wordingConstantsStripped: constantsStripped,
  aiAdvice: { survivorId: alertId, model: advice.body?.model, ms: advice.body?.ms, itemsRephrasedByAi: aiUsed, items: advice.body?.items?.length },
  careTeamToday: roster.body.filter((x) => x.careTeamToday).length,
  advice: { cached: advice.body?.cached ?? null, ms: advice.ms }, strict: STRICT, publishedVerificationMatchesLive: verFresh,
  dataHash, finishedAt: new Date().toISOString(),
};

mkdirSync("runs", { recursive: true });
const stamp = run.startedAt.replace(/[:.]/g, "-");
writeFileSync(`runs/e2e-${stamp}.json`, JSON.stringify(run, null, 1));
writeFileSync("runs/e2e-latest-summary.json", JSON.stringify({ base: BASE, file: `runs/e2e-${stamp}.json`, ...run.summary }, null, 1));
// Verification record for /api/c4/verification (published on the next deploy; --strict checks it matches).
let commit = null;
try { commit = execSync("git rev-parse HEAD", { encoding: "utf8" }).trim(); } catch {}
// strict / publishedVerificationMatchesLive describe the run that checks a published record, so they are not published.
const { strict: _s, publishedVerificationMatchesLive: _m, ...published } = run.summary;
writeFileSync("public/data/verification.json", JSON.stringify({
  e2e: { ...published, artifact: `runs/e2e-${stamp}.json` }, dataHash, commit,
  aiAdvice: { model: advice.body?.model ?? null, providerModelName: advice.body?.providerModelName ?? null, fallbackModel: advice.body?.fallbackModel ?? null, modelNote: advice.body?.modelNote ?? null },
  repo: "https://github.com/zaycruz/c4-glucose-guide",
  reproduce: ["scripts/reproduce.sh <Challenge4_Dose_and_Timing dir> [out dir]   # data check, ML engine, benchmark, evidence; diffs every README number (REPRODUCE.md)",
              "npx wrangler deploy && node scripts/warm-advice.mjs && npx wrangler deploy",
              "node scripts/e2e.mjs && npx wrangler deploy && node scripts/e2e.mjs --strict   # this check"],
  seeds: { engine: 20261017, benchmark: 20261017, analysis: 20261017, realWorld: 7, models: "random_state=0" },
}, null, 1));
console.log(JSON.stringify(run.summary, null, 1));
process.exit(failures.length ? 1 : 0);
