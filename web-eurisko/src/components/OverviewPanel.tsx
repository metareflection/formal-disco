import type { Overview, Heuristic } from "../types";

interface Props {
  overview: Overview;
  heuristics: Heuristic[];
}

export default function OverviewPanel({ overview, heuristics }: Props) {
  const taskTypes = Object.entries(overview.task_status);

  return (
    <>
      <div className="stats-grid">
        <StatCard value={overview.total_concepts} label="Concepts" />
        <StatCard value={overview.proved_count} label="Theorems Proved" color="var(--green)" />
        <StatCard value={overview.conjecture_count} label="Conjectures" color="var(--yellow)" />
        <StatCard value={overview.definition_count} label="Definitions" color="var(--accent)" />
        <StatCard value={overview.total_heuristics} label="Heuristics" color="var(--purple)" />
        <StatCard value={overview.total_tasks} label="Total Tasks" />
      </div>

      <div className="section">
        <h2>Task Queue</h2>
        <table className="task-table">
          <thead>
            <tr>
              <th>Type</th>
              <th>NEW</th>
              <th>DOING</th>
              <th>DONE</th>
              <th>FAILED</th>
              <th>ATTEMPTED</th>
            </tr>
          </thead>
          <tbody>
            {taskTypes.map(([type, counts]) => (
              <tr key={type}>
                <td>{type}</td>
                <td>{counts["NEW"] || 0}</td>
                <td>{counts["DOING"] || 0}</td>
                <td style={{ color: "var(--green)" }}>{counts["DONE"] || 0}</td>
                <td style={{ color: "var(--red)" }}>{counts["FAILED"] || 0}</td>
                <td style={{ color: "var(--yellow)" }}>{counts["ATTEMPTED"] || 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="section">
        <h2>Heuristic Success Rates</h2>
        <div className="bar-chart">
          {heuristics.map((h) => {
            const rate = h.attempts > 0 ? h.successes / h.attempts : 0;
            const maxAttempts = Math.max(...heuristics.map((x) => x.attempts), 1);
            return (
              <div className="bar-row" key={h.name}>
                <div className="label">{h.name}</div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(h.attempts / maxAttempts) * 100}%`,
                      background: `linear-gradient(90deg, var(--green) ${rate * 100}%, var(--border) ${rate * 100}%)`,
                    }}
                  />
                </div>
                <div className="value">
                  {h.successes}/{h.attempts}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

function StatCard({ value, label, color }: { value: number; label: string; color?: string }) {
  return (
    <div className="stat-card">
      <div className="value" style={color ? { color } : undefined}>{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}
