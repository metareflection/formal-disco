import type { Theorem } from "../types";

interface Props {
  theorems: Theorem[];
}

export default function TheoremGallery({ theorems }: Props) {
  if (theorems.length === 0) {
    return (
      <div className="section">
        <h2>Proved Theorems</h2>
        <p style={{ color: "var(--text-muted)" }}>No theorems proved yet. The system is still exploring...</p>
      </div>
    );
  }

  return (
    <div>
      <p style={{ color: "var(--text-muted)", marginBottom: "1rem" }}>
        {theorems.length} theorem{theorems.length !== 1 ? "s" : ""} formally verified in Lean 4
      </p>
      {theorems.map((t) => (
        <div className="theorem-card" key={t.name}>
          <h3>{t.name}</h3>
          <div className="desc">{t.description}</div>
          {t.lean_proof && (
            <pre>{t.lean_proof}</pre>
          )}
          <div className="proof-meta">
            <span>Strategy: {t.proof_strategy}</span>
            {t.origin_heuristic && <span>Heuristic: {t.origin_heuristic}</span>}
            <span>Interest: {t.interestingness.toFixed(2)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
