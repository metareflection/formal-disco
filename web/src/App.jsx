import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router-dom';
import Home from './pages/Home';
import FixerReport from './pages/FixerReport';

export default function App() {
  return (
    <BrowserRouter>
      <nav className="navbar">
        <Link to="/" className="nav-brand">Formal Disco</Link>
        <div className="nav-links">
          <NavLink to="/fixer-report" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Fixer Report
          </NavLink>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/fixer-report" element={<FixerReport />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
