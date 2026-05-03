import React from "react";

function pct(v, decimals = 0) {
  if (v == null || isNaN(v)) return "n/a";
  return (v * 100).toFixed(decimals) + "%";
}

const TOOLTIPS = {
  confidence: "Probability the model assigned to this direction",
  train_acc: "% correct on the 80% training data. High train acc without high val acc suggests overfitting.",
  val_acc: "% correct on the 20% held-back test data. This is the honest accuracy metric.",
  samples: "Number of historical bars used to train the model.",
};

function StatWithTooltip({ label, value, tooltipKey }) {
  const [showTooltip, setShowTooltip] = React.useState(false);
  return (
    <>
      <dt
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        style={{ cursor: "help", position: "relative", display: "inline-block" }}
      >
        {label}
        {showTooltip && (
          <div
            style={{
              position: "absolute",
              bottom: "125%",
              left: "50%",
              transform: "translateX(-50%)",
              padding: "6px 10px",
              background: "rgba(0,0,0,0.95)",
              border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: 4,
              fontSize: 10,
              lineHeight: 1.3,
              color: "#adb5bd",
              maxWidth: "200px",
              whiteSpace: "normal",
              zIndex: 1000,
              pointerEvents: "none",
            }}
          >
            {TOOLTIPS[tooltipKey] || "No info"}
          </div>
        )}
      </dt>
      <dd>{value}</dd>
    </>
  );
}

function Box({ title, p }) {
  if (!p || p.error) {
    return (
      <div className="pred-box" style={{ background: "#161b22", border: "1px solid #30363d" }}>
        <div className="pred-label">{title}</div>
        <div className="pred-meta">{p?.error || "No prediction"}</div>
      </div>
    );
  }
  const cls = p.direction === "up" ? "pred-up" : "pred-down";
  const arrow = p.direction === "up" ? "▲" : "▼";
  return (
    <div className={`pred-box ${cls}`}>
      <div className="pred-label">{title} · {p.horizon_days}d horizon</div>
      <div className="pred-direction">{arrow} {p.direction.toUpperCase()}</div>
      <dl className="pred-stats">
        <StatWithTooltip label="Confidence" value={pct(p.confidence, 1)} tooltipKey="confidence" />
        <StatWithTooltip label="Train acc" value={pct(p.train_acc)} tooltipKey="train_acc" />
        <StatWithTooltip label="Val acc" value={pct(p.val_acc)} tooltipKey="val_acc" />
        <StatWithTooltip label="Samples" value={p.n_samples ?? "—"} tooltipKey="samples" />
      </dl>
    </div>
  );
}

export default function Predictions({ predict }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <h2>Predictions — {predict.ticker}</h2>
      <div className="predictions">
        <Box title="Short-term" p={predict.short_term} />
        <Box title="Long-term" p={predict.long_term} />
      </div>
      <div className="pred-warn">
        <span className="pred-warn-badge">⚠ Prototype</span>
      </div>
    </div>
  );
}
