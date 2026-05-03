function fmt(v, decimals = 2) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(decimals);
}
function fmtPct(p) {
  if (p == null || isNaN(p)) return "—";
  const sign = p >= 0 ? "+" : "";
  return `${sign}${p.toFixed(2)}%`;
}

export default function PeriodSummary({ candles, quote, chartType = "line" }) {
  // Range stats:
  //  - LINE mode: timeframe IS the visible window → use whole candles array
  //    ("1Y" = 1 year of daily bars, range = full year)
  //  - CANDLE mode: timeframe is the CANDLE SIZE, the chart spans years of
  //    history → use ONLY the last candle (it represents the timeframe duration)
  //    e.g. "1M" candle = last candle covers the last month → its open/high/low/close
  //    are the month's stats. Computing from candles[0] would give a multi-year change.
  let range = null;
  if (candles && candles.length >= 1) {
    if (chartType === "candle") {
      const last = candles[candles.length - 1];
      const change = last.close - last.open;
      const changePct = last.open ? (change / last.open) * 100 : 0;
      range = { high: last.high, low: last.low, change, changePct, last };
    } else if (candles.length >= 2) {
      let high = -Infinity, low = Infinity;
      for (const c of candles) {
        if (c.high > high) high = c.high;
        if (c.low  < low)  low  = c.low;
      }
      const first = candles[0];
      const last = candles[candles.length - 1];
      const change = last.close - first.close;
      const changePct = (change / first.close) * 100;
      range = { high, low, change, changePct, last };
    }
  }

  // Sessions come from the live quote — only shown when relevant
  const showSession = quote && (
    quote.pre_market != null || quote.post_market != null || quote.session !== "regular"
  );
  const prevClose = quote?.previous_close;
  const regularChg = prevClose && quote?.regular_price
    ? ((quote.regular_price - prevClose) / prevClose) * 100 : null;
  const preChg = prevClose && quote?.pre_market
    ? ((quote.pre_market - prevClose) / prevClose) * 100 : null;
  const postChg = quote?.regular_price && quote?.post_market
    ? ((quote.post_market - quote.regular_price) / quote.regular_price) * 100 : null;

  if (!range && !showSession) return null;

  return (
    <div className="period-summary">
      {range && (
        <div className="ps-row">
          <div className="ps-cell">
            <div className="ps-label">Range change</div>
            <div className={`ps-value ${range.change >= 0 ? "up" : "down"}`}>
              {range.change >= 0 ? "+" : ""}{fmt(range.change)} ({fmtPct(range.changePct)})
            </div>
          </div>
          <div className="ps-cell">
            <div className="ps-label">Range high</div>
            <div className="ps-value">${fmt(range.high)}</div>
          </div>
          <div className="ps-cell">
            <div className="ps-label">Range low</div>
            <div className="ps-value">${fmt(range.low)}</div>
          </div>
          <div className="ps-cell">
            <div className="ps-label">Last close</div>
            <div className={`ps-value ${range.change >= 0 ? "up" : "down"}`}>
              ${fmt(range.last.close)}
            </div>
          </div>
        </div>
      )}

      {showSession && (
        <div className="ps-row session-row">
          {preChg != null && (
            <div className="ps-cell">
              <div className="ps-label">
                <span className="session-badge pre">Pre-Market</span>
                {quote.pre_market != null && <> · ${fmt(quote.pre_market)}</>}
              </div>
              <div className={`ps-value ${preChg >= 0 ? "up" : "down"}`}>{fmtPct(preChg)}</div>
            </div>
          )}
          {regularChg != null && (
            <div className="ps-cell">
              <div className="ps-label">
                Regular session
                {quote.regular_price != null && <> · ${fmt(quote.regular_price)}</>}
              </div>
              <div className={`ps-value ${regularChg >= 0 ? "up" : "down"}`}>{fmtPct(regularChg)}</div>
            </div>
          )}
          {postChg != null && (
            <div className="ps-cell">
              <div className="ps-label">
                <span className="session-badge post">After Hours</span>
                {quote.post_market != null && <> · ${fmt(quote.post_market)}</>}
              </div>
              <div className={`ps-value ${postChg >= 0 ? "up" : "down"}`}>{fmtPct(postChg)}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
