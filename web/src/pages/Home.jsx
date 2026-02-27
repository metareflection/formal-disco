import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="page">
      <h1>Reports</h1>
      <div className="card-grid">
        <Link to="/fixer-report" className="card">
          <h2>Fixer Eval Report</h2>
          <p>Compare LLM fixer models on Dafny program repair tasks.</p>
        </Link>
        <Link to="/lemma-report" className="card">
          <h2>Lemma Synthesis Report</h2>
          <p>Compare LLM models on Dafny lemma body synthesis tasks.</p>
        </Link>
      </div>
    </div>
  );
}
