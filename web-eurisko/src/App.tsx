import { useEffect, useState } from "react";
import type {
  Overview,
  Concept,
  Heuristic,
  Theorem,
  TraceEvent,
  TraceSummary,
} from "./types";
import OverviewPanel from "./components/OverviewPanel";
import HeuristicDashboard from "./components/HeuristicDashboard";
import TheoremGallery from "./components/TheoremGallery";
import ConceptExplorer from "./components/ConceptExplorer";
import TraceLog from "./components/TraceLog";
import "./style.css";

type Tab = "overview" | "heuristics" | "theorems" | "concepts" | "trace";
type Theme = "dark" | "light";

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem("theme") as Theme) || "dark";
  });
  const [overview, setOverview] = useState<Overview | null>(null);
  const [heuristics, setHeuristics] = useState<Heuristic[]>([]);
  const [theorems, setTheorems] = useState<Theorem[]>([]);
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [traceSummary, setTraceSummary] = useState<TraceSummary | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const refresh = () => {
    fetch("/api/overview").then((r) => r.json()).then(setOverview);
    fetch("/api/heuristics").then((r) => r.json()).then(setHeuristics);
    fetch("/api/theorems").then((r) => r.json()).then(setTheorems);
    fetch("/api/concepts").then((r) => r.json()).then(setConcepts);
    fetch("/api/trace?limit=500").then((r) => r.json()).then(setTrace);
    fetch("/api/trace/summary").then((r) => r.json()).then(setTraceSummary);
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Eurisko Discovery Dashboard</h1>
        <nav>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? "\u2600\uFE0F" : "\uD83C\uDF19"}
          </button>
          {(["overview", "heuristics", "theorems", "concepts", "trace"] as Tab[]).map(
            (t) => (
              <button
                key={t}
                className={tab === t ? "active" : ""}
                onClick={() => setTab(t)}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            )
          )}
        </nav>
      </header>
      <main>
        {tab === "overview" && overview && (
          <OverviewPanel overview={overview} heuristics={heuristics} />
        )}
        {tab === "heuristics" && (
          <HeuristicDashboard heuristics={heuristics} />
        )}
        {tab === "theorems" && <TheoremGallery theorems={theorems} />}
        {tab === "concepts" && <ConceptExplorer concepts={concepts} />}
        {tab === "trace" && <TraceLog events={trace} summary={traceSummary} />}
      </main>
    </div>
  );
}
