import { useEffect } from "react";

const TYPE_META = {
  earnings:      { icon: "📊", color: "#58a6ff", label: "Earnings Report" },
  earnings_call: { icon: "📞", color: "#bc8cff", label: "Earnings Call" },
  dividend:      { icon: "💵", color: "#3fb950", label: "Dividend Payment" },
  split:         { icon: "✂",  color: "#e3b341", label: "Stock Split" },
};

function formatLongDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso + "T00:00:00Z");
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  });
}

export default function EventDetailModal({ event, ticker, onClose }) {
  useEffect(() => {
    function onKey(e) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!event) return null;
  const meta = TYPE_META[event.type] || { icon: "•", color: "#8b949e", label: event.type };
  const surprise = (event.eps_actual != null && event.eps_estimate != null)
    ? event.eps_actual - event.eps_estimate
    : null;
  const surprisePct = (surprise != null && event.eps_estimate)
    ? (surprise / Math.abs(event.eps_estimate)) * 100 : null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-icon" style={{ color: meta.color }}>{meta.icon}</span>
          <div>
            <div className="modal-title">{ticker} · {meta.label}</div>
            <div className="modal-subtitle">{formatLongDate(event.date)}</div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="modal-body">
          {event.type === "earnings" && (
            <dl className="modal-stats">
              <div>
                <dt>Reported EPS</dt>
                <dd>{event.eps_actual != null ? event.eps_actual.toFixed(2) : "—"}</dd>
              </div>
              <div>
                <dt>Estimated EPS</dt>
                <dd>{event.eps_estimate != null ? event.eps_estimate.toFixed(2) : "—"}</dd>
              </div>
              {surprise != null && (
                <div>
                  <dt>Surprise</dt>
                  <dd className={surprise >= 0 ? "up" : "down"}>
                    {surprise >= 0 ? "+" : ""}{surprise.toFixed(2)}
                    {surprisePct != null && (
                      <span className="muted-inline">
                        {" "}({surprise >= 0 ? "+" : ""}{surprisePct.toFixed(1)}%)
                      </span>
                    )}
                  </dd>
                </div>
              )}
              {event.eps_actual == null && event.eps_estimate == null && (
                <div className="modal-note">
                  EPS data not yet released by analysts.
                </div>
              )}
            </dl>
          )}
          {event.type === "earnings_call" && (
            <div className="modal-note">
              Conference call where management discusses the quarterly results
              and answers questions from analysts.
            </div>
          )}
          {event.type === "dividend" && (
            <dl className="modal-stats">
              <div>
                <dt>Amount per share</dt>
                <dd className="up">${event.value != null ? event.value.toFixed(4) : "—"}</dd>
              </div>
              <div className="modal-note">
                Cash distributed to shareholders on record. The ex-dividend date
                is when the stock typically opens lower by the dividend amount.
              </div>
            </dl>
          )}
          {event.type === "split" && (
            <dl className="modal-stats">
              <div>
                <dt>Split ratio</dt>
                <dd>{event.value != null ? `${event.value} : 1` : "—"}</dd>
              </div>
              <div className="modal-note">
                Each existing share was multiplied by this factor. Total market
                cap is unchanged; only the per-share price scales down.
              </div>
            </dl>
          )}
        </div>

        <div className="modal-footer muted-inline">
          {event.type === "earnings" || event.type === "earnings_call"
            ? "Source: Yahoo Finance earnings calendar"
            : "Source: Yahoo Finance corporate actions"}
        </div>
      </div>
    </div>
  );
}
