import { useMemo, useState } from "react";
import type { TraceEvent, TraceSummary } from "../types";

interface Props {
  events: TraceEvent[];
  summary: TraceSummary | null;
}

const KIND_COLORS: Record<string, string> = {
  heuristic_apply: "#4a7fc1",
  admit: "#3aa66e",
  reject: "#c46a4a",
  worth_update: "#b58a3e",
  heuristic_birth: "#9c5fc4",
  heuristic_death: "#7a7a7a",
  proof_attempt: "#3aa6a6",
  reflection: "#5b6dc4",
};

export default function TraceLog({ events, summary }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [kindFilter, setKindFilter] = useState<string>("");
  const [workerFilter, setWorkerFilter] = useState<string>("");

  const kinds = useMemo(
    () => Array.from(new Set(events.map((e) => e.kind))).sort(),
    [events]
  );
  const workers = useMemo(
    () => Array.from(new Set(events.map((e) => e.worker))).sort(),
    [events]
  );

  const filtered = useMemo(
    () =>
      events.filter(
        (e) =>
          (!kindFilter || e.kind === kindFilter) &&
          (!workerFilter || e.worker === workerFilter)
      ),
    [events, kindFilter, workerFilter]
  );

  const toggle = (path: string) =>
    setExpanded((s) => {
      const n = new Set(s);
      if (n.has(path)) n.delete(path);
      else n.add(path);
      return n;
    });

  return (
    <div className="trace-log">
      {summary && (
        <div className="trace-summary">
          <div className="summary-item">
            <strong>Total events:</strong> {summary.total}
          </div>
          <div className="summary-row">
            {Object.entries(summary.by_kind)
              .sort((a, b) => b[1] - a[1])
              .map(([k, n]) => (
                <span
                  key={k}
                  className="kind-chip"
                  style={{ backgroundColor: KIND_COLORS[k] || "#888" }}
                  onClick={() => setKindFilter(k === kindFilter ? "" : k)}
                  title="click to filter"
                >
                  {k}: {n}
                </span>
              ))}
          </div>
        </div>
      )}

      <div className="trace-filters">
        <label>
          kind:
          <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
            <option value="">(all)</option>
            {kinds.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label>
          worker:
          <select value={workerFilter} onChange={(e) => setWorkerFilter(e.target.value)}>
            <option value="">(all)</option>
            {workers.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>
        <span className="trace-count">{filtered.length} shown</span>
      </div>

      <table className="trace-table">
        <thead>
          <tr>
            <th>tick</th>
            <th>worker</th>
            <th>kind</th>
            <th>summary</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((e) => (
            <>
              <tr
                key={e.path}
                className="trace-row"
                onClick={() => toggle(e.path)}
              >
                <td className="tick">{e.tick}</td>
                <td>{e.worker}</td>
                <td>
                  <span
                    className="kind-chip"
                    style={{ backgroundColor: KIND_COLORS[e.kind] || "#888" }}
                  >
                    {e.kind}
                  </span>
                </td>
                <td className="summary-cell">{summarize(e)}</td>
              </tr>
              {expanded.has(e.path) && (
                <tr key={e.path + ":expanded"} className="trace-expanded">
                  <td colSpan={4}>
                    <pre>{JSON.stringify(e.payload, null, 2)}</pre>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function summarize(e: TraceEvent): string {
  const p = e.payload as Record<string, unknown>;
  switch (e.kind) {
    case "heuristic_apply":
      return `${p.heuristic} on ${p.target} → ${p.n_candidates} candidates`;
    case "admit":
      return `${p.heuristic} → ${p.candidate} (${p.reason})`;
    case "reject":
      return `${p.heuristic} ✗ ${p.candidate} (${p.reason})`;
    case "worth_update":
      return `${p.heuristic}: ${fmt(p.before)} → ${fmt(p.after)} (${p.cause})`;
    case "heuristic_birth":
      return `${p.name} ← ${p.parent_heuristic}`;
    case "heuristic_death":
      return `${p.name} (${p.successes}/${p.attempts})`;
    case "proof_attempt":
      return `${p.theorem} via ${p.strategy}: ${p.outcome}`;
    case "reflection":
      return `on ${p.target} (${p.outcome})`;
    default:
      return JSON.stringify(p).slice(0, 120);
  }
}

function fmt(v: unknown): string {
  if (typeof v === "number") return v.toFixed(3);
  return String(v);
}
