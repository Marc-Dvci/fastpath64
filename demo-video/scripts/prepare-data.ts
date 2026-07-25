/**
 * Build the video's data file from the committed benchmark CSVs.
 *
 * The video is not allowed to contain hand-typed numbers: every figure it animates is read
 * from the same result files the writeup cites, so a re-run of the benchmarks changes the
 * video the next time it renders.
 */
import {readFileSync, writeFileSync, existsSync} from "node:fs";
import {join, dirname} from "node:path";
import {fileURLToPath} from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");

type Row = Record<string, string>;

function readCsv(path: string): Row[] {
  if (!existsSync(path)) return [];
  const text = readFileSync(path, "utf8").trim();
  const [head, ...lines] = text.split(/\r?\n/);
  const cols = head.split(",");
  // llama-bench wraps every cell in double quotes and never puts a comma inside one,
  // so a plain split plus a dequote is sufficient - and forgetting the dequote turns
  // every numeric field into NaN silently.
  const dequote = (c: string) => c.replace(/^"(.*)"$/, "$1");
  return lines.map((l) => {
    const cells = l.split(",").map(dequote);
    return Object.fromEntries(cols.map((c, i) => [dequote(c), cells[i] ?? ""])) as Row;
  });
}

function caseOf(r: Row): string {
  const p = Number(r.n_prompt ?? 0);
  const g = Number(r.n_gen ?? 0);
  return p && !g ? `pp${p}` : `tg${g}`;
}

function pick(rows: Row[], needle: string, kase: string): number | null {
  for (const r of rows) {
    const name = (r.model_filename ?? "").split("/").pop() ?? "";
    if (name.toLowerCase().includes(needle) && caseOf(r) === kase) {
      const v = Number(r.avg_ts);
      if (Number.isFinite(v)) return v;
    }
  }
  return null;
}

const ab = join(REPO, "results", "raw-ab", "fastpath-ab");
const p0 = join(REPO, "results", "raw");

const stock = readCsv(join(ab, "bench-stock.csv"));
const patched = readCsv(join(ab, "bench-patched.csv"));
const repackOn = readCsv(join(p0, "phase0-n2-stock", "bench-n2-stock.csv"));
const repackOff = readCsv(join(p0, "phase0-n2-no-repack", "bench-n2-no-repack.csv"));

const need = (v: number | null, what: string): number => {
  if (v === null) throw new Error(`missing measurement: ${what} - run the benchmarks first`);
  return v;
};

const data = {
  generatedAt: new Date().toISOString(),
  cpu: "Neoverse-N2",
  features: "sve2 · svei8mm · i8mm · bf16",
  ab: {
    iq4xs: {
      pp512: {stock: need(pick(stock, "iq4_xs", "pp512"), "stock iq4_xs pp512"),
              fast: need(pick(patched, "iq4_xs", "pp512"), "patched iq4_xs pp512")},
      pp2048: {stock: need(pick(stock, "iq4_xs", "pp2048"), "stock iq4_xs pp2048"),
               fast: need(pick(patched, "iq4_xs", "pp2048"), "patched iq4_xs pp2048")},
    },
    q4k: {
      pp512: {stock: need(pick(stock, "q4_k", "pp512"), "stock q4_k pp512"),
              fast: need(pick(patched, "q4_k", "pp512"), "patched q4_k pp512")},
    },
  },
  toggle: {
    q4k: {off: need(pick(repackOff, "q4_k", "pp512"), "repack-off q4_k"),
          on: need(pick(repackOn, "q4_k", "pp512"), "repack-on q4_k")},
    iq4xs: {off: need(pick(repackOff, "iq4_xs", "pp512"), "repack-off iq4_xs"),
            on: need(pick(repackOn, "iq4_xs", "pp512"), "repack-on iq4_xs")},
  },
};

const out = join(HERE, "..", "src", "generated-data.json");
writeFileSync(out, JSON.stringify(data, null, 2) + "\n");
console.log(`wrote ${out}`);
console.log(
  `  iq4_xs pp512: ${data.ab.iq4xs.pp512.stock} -> ${data.ab.iq4xs.pp512.fast} ` +
  `(${(data.ab.iq4xs.pp512.fast / data.ab.iq4xs.pp512.stock).toFixed(2)}x)`
);
