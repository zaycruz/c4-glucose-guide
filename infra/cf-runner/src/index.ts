// POST /start {repo, ref}  -> clone, fetch data, run scripts/reproduce.sh in a fresh sandbox
// GET  /status?id=          -> state + log tail;  GET /file?id=&path= -> a file from the sandbox
// GET  /data                -> the dataset tarball (for the sandbox; bearer token required)
// POST /stop?id=            -> destroy the sandbox
// Every route needs "authorization: Bearer <RUNNER_TOKEN>".
import { getSandbox, Sandbox } from "@cloudflare/sandbox";
export { Sandbox };

interface Env { Sandbox: DurableObjectNamespace<Sandbox>; DATA: R2Bucket; RUNNER_TOKEN: string }

const JOB = `set -uo pipefail
W=/workspace/job
mkdir -p $W && cd $W
echo running > $W/state
{
  echo "host: Cloudflare Sandbox ($(nproc) vCPU, $(awk '/MemTotal/{printf "%.0f GiB", $2/1048576}' /proc/meminfo)), $(uname -sr)"
  echo "started: $(date -u +%FT%TZ)"
  git clone --quiet "$REPO" repo && cd repo && git checkout --quiet "$REF" && echo "commit: $(git log -1 --format='%H %s')"
  mkdir -p $W/data && curl -fsS -H "authorization: Bearer $TOKEN" "$DATA_URL" | tar xz -C $W/data && echo "data: $(du -sh $W/data | cut -f1)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$W/bin UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
  export PATH=$W/bin:$PATH UV_CACHE_DIR=$W/cache
  scripts/reproduce.sh $W/data/Challenge4_Dose_and_Timing $W/out
  echo "reproduce exit: $?"
  echo "finished: $(date -u +%FT%TZ)"
} > $W/run.log 2>&1
echo done > $W/state
`;

const json = (o: unknown, status = 200) => new Response(JSON.stringify(o, null, 1), { status, headers: { "content-type": "application/json" } });

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!env.RUNNER_TOKEN || req.headers.get("authorization") !== `Bearer ${env.RUNNER_TOKEN}`) return json({ error: "unauthorized" }, 401);
    const url = new URL(req.url);
    if (url.pathname === "/data") {
      const o = await env.DATA.get("c4-data.tar.gz");
      return o ? new Response(o.body, { headers: { "content-type": "application/gzip" } }) : json({ error: "no data uploaded" }, 404);
    }
    const id = url.searchParams.get("id") ?? "c4-repro";
    const sb = getSandbox(env.Sandbox, id, { keepAlive: true });
    if (url.pathname === "/start" && req.method === "POST") {
      const { repo, ref } = (await req.json()) as { repo: string; ref: string };
      await sb.writeFile("/workspace/job.sh", JOB);
      const p = await sb.startProcess("bash /workspace/job.sh", {
        env: { REPO: repo, REF: ref, DATA_URL: `${url.origin}/data`, TOKEN: env.RUNNER_TOKEN },
      });
      return json({ id, process: p.id });
    }
    if (url.pathname === "/status") {
      const r = await sb.exec("cat /workspace/job/state 2>/dev/null; echo ---; tail -c 6000 /workspace/job/run.log 2>/dev/null; echo ---; cat /workspace/job/out/steps.tsv 2>/dev/null");
      return new Response(r.stdout, { headers: { "content-type": "text/plain" } });
    }
    if (url.pathname === "/file") {
      const f = await sb.readFile(url.searchParams.get("path") ?? "");
      return new Response(f.content, { headers: { "content-type": "text/plain" } });
    }
    if (url.pathname === "/stop" && req.method === "POST") {
      await sb.destroy();
      return json({ stopped: id });
    }
    return json({ error: "not found", routes: ["/start", "/status", "/file", "/data", "/stop"] }, 404);
  },
};
