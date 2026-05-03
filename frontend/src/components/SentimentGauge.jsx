export default function SentimentGauge({ score = 0 }) {
  // score in [-1, 1] -> needle angle on a half-circle gauge
  // Left tip (-1, bearish, red), top (0, neutral), right tip (+1, bullish, green)
  const clamped = Math.max(-1, Math.min(1, score));
  const cx = 110, cy = 110, r = 90;

  // Math angle on the upper half-circle: π (left) -> 0 (right)
  // map score s in [-1, 1] -> theta in [π, 0]:  theta = π * (1 - (s+1)/2) = π * (1 - s)/2
  const scoreToTheta = (s) => Math.PI * (1 - s) / 2;
  const polar = (theta) => ({
    x: cx + r * Math.cos(theta),
    y: cy - r * Math.sin(theta), // SVG y grows downward
  });

  // Build an arc path between two scores
  const arcBetween = (sStart, sEnd, color) => {
    const t1 = scoreToTheta(sStart);
    const t2 = scoreToTheta(sEnd);
    const p1 = polar(t1);
    const p2 = polar(t2);
    // sStart < sEnd means moving rightward across the gauge -> sweep flag = 1 (clockwise in SVG)
    return (
      <path
        d={`M ${p1.x.toFixed(2)} ${p1.y.toFixed(2)} A ${r} ${r} 0 0 1 ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`}
        stroke={color}
        strokeWidth="18"
        fill="none"
        strokeLinecap="round"
      />
    );
  };

  // Needle
  const tip = polar(scoreToTheta(clamped));
  const needleEndX = cx + (r - 8) * Math.cos(scoreToTheta(clamped));
  const needleEndY = cy - (r - 8) * Math.sin(scoreToTheta(clamped));

  const label = clamped > 0.2 ? "Bullish" : clamped < -0.2 ? "Bearish" : "Neutral";
  const color = clamped > 0.2 ? "#26a69a" : clamped < -0.2 ? "#ef5350" : "#8b949e";

  return (
    <div style={{ textAlign: "center" }}>
      <svg width="220" height="140" viewBox="0 0 220 140">
        {arcBetween(-1, -0.33, "#ef5350")}
        {arcBetween(-0.33, 0.33, "#f5b945")}
        {arcBetween(0.33, 1, "#26a69a")}
        <line x1={cx} y1={cy} x2={needleEndX} y2={needleEndY} stroke="#e6edf3" strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="6" fill="#e6edf3" />
      </svg>
      <div style={{ fontSize: 28, fontWeight: 700, color, marginTop: -6 }}>
        {clamped >= 0 ? "+" : ""}{clamped.toFixed(2)}
      </div>
      <div style={{ fontSize: 12, color: "#8b949e", textTransform: "uppercase", letterSpacing: 1 }}>
        {label}
      </div>
    </div>
  );
}
