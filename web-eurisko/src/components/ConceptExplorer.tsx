import { useState, useMemo } from "react";
import type { Concept } from "../types";

interface Props {
  concepts: Concept[];
}

type KindFilter = "all" | "theorem" | "conjecture" | "definition" | "operation";

export default function ConceptExplorer({ concepts }: Props) {
  const [filter, setFilter] = useState<KindFilter>("all");
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    let list = concepts;
    if (filter !== "all") {
      list = list.filter((c) => c.kind === filter);
    }
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.description.toLowerCase().includes(q) ||
          (c.origin_heuristic || "").toLowerCase().includes(q)
      );
    }
    return list.sort((a, b) => b.interestingness - a.interestingness);
  }, [concepts, filter, search]);

  const kindCounts = useMemo(() => {
    const counts: Record<string, number> = { all: concepts.length };
    for (const c of concepts) {
      counts[c.kind] = (counts[c.kind] || 0) + 1;
    }
    return counts;
  }, [concepts]);

  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div>
      <div className="concept-filters">
        {(["all", "theorem", "conjecture", "definition", "operation"] as KindFilter[]).map((k) => (
          <button
            key={k}
            className={filter === k ? "active" : ""}
            onClick={() => setFilter(k)}
          >
            {k} ({kindCounts[k] || 0})
          </button>
        ))}
        <input
          placeholder="Search concepts..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="concept-row" style={{ fontWeight: 600, color: "var(--text-muted)", fontSize: "0.75rem" }}>
        <div>Name</div>
        <div>Kind</div>
        <div>Description</div>
        <div>Interest</div>
      </div>

      {filtered.slice(0, 200).map((c) => (
        <div key={c.path}>
          <div
            className="concept-row"
            style={{ cursor: "pointer" }}
            onClick={() => setExpanded(expanded === c.path ? null : c.path)}
          >
            <div className="name">{c.name}</div>
            <div>
              <span className={`kind-badge kind-${c.kind}`}>{c.kind}</span>
            </div>
            <div style={{ color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {c.description}
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
              {c.interestingness.toFixed(2)}
            </div>
          </div>
          {expanded === c.path && (
            <div style={{ padding: "0.75rem", background: "var(--bg)", borderRadius: "4px", margin: "0.25rem 0 0.5rem", fontSize: "0.8rem" }}>
              {c.lean_statement && (
                <pre style={{ marginBottom: "0.5rem", whiteSpace: "pre-wrap", fontFamily: "var(--mono)" }}>
                  {c.lean_statement}
                </pre>
              )}
              {c.lean_proof && (
                <>
                  <div style={{ color: "var(--green)", marginBottom: "0.25rem" }}>Proof:</div>
                  <pre style={{ marginBottom: "0.5rem", whiteSpace: "pre-wrap", fontFamily: "var(--mono)" }}>
                    {c.lean_proof}
                  </pre>
                </>
              )}
              <div style={{ color: "var(--text-muted)", display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                {c.origin_heuristic && <span>Heuristic: {c.origin_heuristic}</span>}
                {c.proof_strategy && <span>Strategy: {c.proof_strategy}</span>}
                {c.tags.length > 0 && <span>Tags: {c.tags.join(", ")}</span>}
                {c.related_concepts.length > 0 && <span>Related: {c.related_concepts.join(", ")}</span>}
              </div>
            </div>
          )}
        </div>
      ))}

      {filtered.length > 200 && (
        <p style={{ color: "var(--text-muted)", padding: "1rem 0" }}>
          Showing 200 of {filtered.length} concepts
        </p>
      )}
    </div>
  );
}
