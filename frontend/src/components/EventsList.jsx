import { useState } from "react";
import EventDetailModal from "./EventDetailModal.jsx";

const TYPE_CONFIG = {
  earnings:      { icon: "📊", color: "#58a6ff" },
  earnings_call: { icon: "📞", color: "#bc8cff" },
  dividend:      { icon: "💵", color: "#3fb950" },
  split:         { icon: "✂",  color: "#e3b341" },
};

function EventRow({ ev, onClick }) {
  const cfg = TYPE_CONFIG[ev.type] ?? { icon: "•", color: "#8b949e" };
  return (
    <button className="event-row" onClick={onClick}>
      <span className="event-icon">{cfg.icon}</span>
      <span className="event-date">{ev.date}</span>
      <span className="event-type" style={{ color: cfg.color }}>{ev.label}</span>
      <span className="event-detail">
        {ev.type === "earnings" && ev.eps_actual != null && (
          <>EPS <strong>{ev.eps_actual >= 0 ? "+" : ""}{ev.eps_actual.toFixed(2)}</strong>
          {ev.eps_estimate != null && (
            <span className="muted-inline"> vs est. {ev.eps_estimate.toFixed(2)}</span>
          )}</>
        )}
        {ev.type === "earnings" && ev.eps_actual == null && ev.eps_estimate != null && (
          <span className="muted-inline">Est. EPS {ev.eps_estimate.toFixed(2)}</span>
        )}
        {ev.type === "dividend" && ev.value != null && `$${ev.value.toFixed(4)}`}
        {ev.type === "split" && ev.value != null && `${ev.value}:1`}
      </span>
      <span className="event-chev">›</span>
    </button>
  );
}

export default function EventsList({ events, ticker }) {
  const [showAll, setShowAll] = useState(false);
  const [selected, setSelected] = useState(null);

  if (!events) {
    return (
      <div className="card loading-bar">
        Loading events for this ticker…
      </div>
    );
  }

  const { past = [], upcoming = [] } = events;
  const oneYearAgo = Date.now() / 1000 - 365 * 24 * 3600;
  const recentPast = past.filter((e) => e.ts >= oneYearAgo);
  const olderPast  = past.filter((e) => e.ts <  oneYearAgo);
  const displayedPast = showAll ? past : recentPast;

  return (
    <>
      {upcoming.length > 0 && (
        <div className="card events-card upcoming-card">
          <div className="card-title">Upcoming Events</div>
          <div className="events-list">
            {upcoming.map((ev, i) => (
              <EventRow key={`up-${i}`} ev={ev} onClick={() => setSelected(ev)} />
            ))}
          </div>
        </div>
      )}

      <div className="card events-card">
        <div className="card-title">
          Past Events{!showAll && " · Last 12 Months"}
          <span className="muted-inline" style={{ marginLeft: 8, textTransform: "none" }}>
            ({displayedPast.length} of {past.length})
          </span>
        </div>
        {displayedPast.length === 0 ? (
          <div className="placeholder">
            {past.length > 0
              ? `No events in the last year. ${past.length} older events available.`
              : "No events found."}
          </div>
        ) : (
          <div className="events-list">
            {displayedPast.map((ev, i) => (
              <EventRow key={`past-${i}`} ev={ev} onClick={() => setSelected(ev)} />
            ))}
          </div>
        )}
        {!showAll && olderPast.length > 0 && (
          <button className="show-more-btn" onClick={() => setShowAll(true)}>
            Show {olderPast.length} older event{olderPast.length !== 1 ? "s" : ""} (all-time)
          </button>
        )}
        {showAll && olderPast.length > 0 && (
          <button className="show-more-btn" onClick={() => setShowAll(false)}>
            Show less (last 12 months only)
          </button>
        )}
      </div>

      {selected && (
        <EventDetailModal
          event={selected}
          ticker={ticker}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  );
}
