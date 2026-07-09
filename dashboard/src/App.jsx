import { NavLink, Route, Routes } from "react-router-dom";
import Showcase from "./pages/Showcase.jsx";
import Live from "./pages/Live.jsx";

function Nav() {
  return (
    <header className="nav">
      <div className="container nav-inner">
        <div className="brand">
          <span className="dot" />
          <span>Multi-Agent Trading System</span>
          <span className="pill" style={{ marginLeft: 8 }}>paper · keyless</span>
        </div>
        <nav className="nav-links">
          <NavLink to="/" end>Showcase</NavLink>
          <NavLink to="/live">Live Dashboard</NavLink>
          <a href="https://github.com/Chinmay258/Multi-agent-trading-system" target="_blank" rel="noreferrer">GitHub ↗</a>
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="container inner">
        <div className="disclaimer">
          <strong>Disclaimer — educational / demonstration only.</strong> This is not financial,
          investment, or trading advice and not a solicitation to buy or sell any asset. All trading
          shown is <strong>simulated (paper) trading</strong> on public market data with no real money.
          Simulated results do not indicate future performance. No warranty; use at your own risk.
        </div>
        <p style={{ marginTop: 14 }}>
          Built as a portfolio/learning project · live public market data via CCXT (keyless) · paper
          execution · <a href="https://github.com/Chinmay258/Multi-agent-trading-system" target="_blank" rel="noreferrer">source on GitHub</a>.
        </p>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <>
      <Nav />
      <main className="container">
        <Routes>
          <Route path="/" element={<Showcase />} />
          <Route path="/live" element={<Live />} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}
