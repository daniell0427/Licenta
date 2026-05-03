function formatLocalDate(ts, fallback) {
  if (ts) {
    const d = new Date(ts * 1000);
    if (!isNaN(d.getTime())) {
      return d.toLocaleString(undefined, {
        year: "numeric", month: "short", day: "2-digit",
        hour: "2-digit", minute: "2-digit",
      });
    }
  }
  return fallback || "";
}

export default function NewsList({ items, highlightUrl, highlightTitle }) {
  if (!items || items.length === 0) return <div className="placeholder">No news available</div>;
  return (
    <div className="news-list">
      {items.map((it, i) => {
        const isHighlighted =
          (highlightUrl && it.url && it.url === highlightUrl) ||
          (highlightTitle && it.title && it.title === highlightTitle);
        return (
          <div
            className={`news-item ${isHighlighted ? "highlighted" : ""}`}
            key={i}
            data-news-url={it.url || ""}
            data-news-title={it.title || ""}
          >
            {it.thumbnail ? (
              <img className="news-thumb" src={it.thumbnail} alt="" />
            ) : (
              <div className="news-thumb-placeholder" />
            )}
            <div>
              <p className="news-title">
                {it.url ? (
                  <a href={it.url} target="_blank" rel="noreferrer">{it.title}</a>
                ) : (
                  it.title
                )}
              </p>
              <div className="news-meta">
                {it.label && (
                  <span className={`badge badge-${it.label}`}>
                    {it.label} · {(it.confidence * 100).toFixed(0)}%
                  </span>
                )}
                {it.publisher && <span>{it.publisher}</span>}
                <span>{formatLocalDate(it.ts, it.date)}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
