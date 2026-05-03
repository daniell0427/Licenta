// Reusable loading indicators

export function LoadingBar({ children = "Loading…" }) {
  return <div className="loading-bar">{children}</div>;
}

export function Skeleton({ height = 20, width = "100%", style = {} }) {
  return <div className="skeleton" style={{ height, width, ...style }} />;
}

// A card-shaped skeleton placeholder
export function SkeletonCard({ lines = 3, title = true }) {
  return (
    <div className="card">
      {title && <Skeleton height={14} width={140} style={{ marginBottom: 14 }} />}
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} height={14} width={`${85 - i * 10}%`} style={{ marginBottom: 8 }} />
      ))}
    </div>
  );
}

// Skeleton table row — for trending/gainers/losers/predictions
export function SkeletonTable({ rows = 5, cols = 3 }) {
  return (
    <table className="trending-table">
      <tbody>
        {Array.from({ length: rows }).map((_, r) => (
          <tr key={r}>
            {Array.from({ length: cols }).map((_, c) => (
              <td key={c}><Skeleton height={14} width={c === 0 ? 60 : 80} /></td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
