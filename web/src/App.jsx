import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router-dom';
import Home from './pages/Home';
import FixerReport from './pages/FixerReport';
import LemmaReport from './pages/LemmaReport';
import DiversityReport from './pages/DiversityReport';
import DatasetStatsReport from './pages/DatasetStatsReport';

export default function App() {
  return (
    <BrowserRouter>
      <nav className="navbar">
        <Link to="/" className="nav-brand">Formal Disco</Link>
        <div className="nav-links">
          <NavLink to="/fixer-report" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Fixer Report
          </NavLink>
          <NavLink to="/lemma-report" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Lemma Report
          </NavLink>
          <NavLink to="/diversity-report" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Diversity Report
          </NavLink>
          <NavLink to="/dataset-stats" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Dataset Stats
          </NavLink>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/fixer-report" element={<FixerReport />} />
          <Route path="/lemma-report" element={<LemmaReport />} />
          <Route path="/diversity-report" element={<DiversityReport />} />
          <Route path="/dataset-stats" element={<DatasetStatsReport />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
