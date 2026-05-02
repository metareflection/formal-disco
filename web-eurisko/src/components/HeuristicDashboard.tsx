import type { Heuristic } from "../types";

interface Props {
  heuristics: Heuristic[];
}

export default function HeuristicDashboard({ heuristics }: Props) {
  return (
    <div>
      {heuristics.map((h) => (
        <div className="heuristic-card" key={h.name}>
          <h3>
            <span className="name">{h.name}</span>
            <span className="badge">{h.heuristic_kind}</span>
            {h.eurisclo_origin && <span className="badge">{h.eurisclo_origin}</span>}
            {h.born_from_reflection && (
              <span className="badge" style={{ color: "var(--purple)", borderColor: "var(--purple)" }}>
                born
              </span>
            )}
          </h3>
          <div className="meta">
            <span>attempts: {h.attempts}</span>
            <span>successes: {h.successes}</span>
            <span>
              rate: {h.attempts > 0 ? ((h.successes / h.attempts) * 100).toFixed(1) : "0.0"}%
            </span>
            <span>interest: {h.interestingness.toFixed(3)}</span>
            <span>concepts: {h.concepts_created}</span>
            <span style={{ color: "var(--green)" }}>theorems: {h.theorems_proved}</span>
          </div>
          <div className="template">{h.template}</div>
        </div>
      ))}
    </div>
  );
}
