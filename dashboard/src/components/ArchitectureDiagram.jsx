// Architecture diagram: the seven agents communicating only over the Redis bus.
// Pure SVG so it scales crisply and needs no chart lib.
export default function ArchitectureDiagram() {
  const agents = [
    { x: 40, label: "Market Data", sub: "CCXT (keyless)", color: "#3b82f6" },
    { x: 215, label: "Technical Analysis", sub: "indicators + ML", color: "#3b82f6" },
    { x: 390, label: "Decision", sub: "aggregate + filter", color: "#8b5cf6" },
    { x: 565, label: "Risk", sub: "8 checks + sizing", color: "#f59e0b" },
    { x: 740, label: "Execution", sub: "PaperBroker", color: "#22c55e" },
  ];
  const box = (a, i) => (
    <g key={i}>
      <rect x={a.x} y={150} width={150} height={58} rx={10} fill="#131b2b" stroke={a.color} strokeWidth="1.5" />
      <text x={a.x + 75} y={176} textAnchor="middle" fill="#e6ebf5" fontSize="14" fontWeight="700">{a.label}</text>
      <text x={a.x + 75} y={194} textAnchor="middle" fill="#8a97b0" fontSize="11">{a.sub}</text>
      {/* connector down to the bus */}
      <line x1={a.x + 75} y1={208} x2={a.x + 75} y2={250} stroke="#243049" strokeWidth="1.5" />
      <circle cx={a.x + 75} cy={250} r="3" fill={a.color} />
    </g>
  );
  return (
    <div className="card" style={{ overflowX: "auto" }}>
      <svg viewBox="0 0 920 360" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Architecture diagram">
        {/* pipeline arrows */}
        {[190, 365, 540, 715].map((x, i) => (
          <g key={i}>
            <line x1={x} y1={179} x2={x + 25} y2={179} stroke="#3a496b" strokeWidth="2" markerEnd="url(#arr)" />
          </g>
        ))}
        <defs>
          <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#3a496b" />
          </marker>
        </defs>

        {agents.map(box)}

        {/* Redis bus */}
        <rect x={40} y={262} width={850} height={42} rx={10} fill="url(#busg)" stroke="#243049" />
        <defs>
          <linearGradient id="busg" x1="0" x2="1">
            <stop offset="0" stopColor="#1a2336" />
            <stop offset="1" stopColor="#141b2b" />
          </linearGradient>
        </defs>
        <text x={465} y={288} textAnchor="middle" fill="#cbd5e1" fontSize="13" fontWeight="700" letterSpacing="0.5">
          Redis pub/sub message bus — the only way agents talk
        </text>

        {/* side agents */}
        <rect x={40} y={40} width={150} height={48} rx={10} fill="#0f1521" stroke="#243049" />
        <text x={115} y={62} textAnchor="middle" fill="#e6ebf5" fontSize="13" fontWeight="700">Monitoring</text>
        <text x={115} y={78} textAnchor="middle" fill="#8a97b0" fontSize="11">health · alerts</text>

        <rect x={215} y={40} width={150} height={48} rx={10} fill="#0f1521" stroke="#243049" />
        <text x={290} y={62} textAnchor="middle" fill="#e6ebf5" fontSize="13" fontWeight="700">Sentiment</text>
        <text x={290} y={78} textAnchor="middle" fill="#8a97b0" fontSize="11">disabled stub</text>

        <rect x={740} y={40} width={150} height={48} rx={10} fill="#0f1521" stroke="#243049" />
        <text x={815} y={62} textAnchor="middle" fill="#e6ebf5" fontSize="13" fontWeight="700">FastAPI</text>
        <text x={815} y={78} textAnchor="middle" fill="#8a97b0" fontSize="11">control plane + WS</text>

        {[115, 290, 815].map((x, i) => (
          <line key={i} x1={x} y1={88} x2={x} y2={262} stroke="#243049" strokeWidth="1.2" strokeDasharray="4 4" />
        ))}
      </svg>
    </div>
  );
}
