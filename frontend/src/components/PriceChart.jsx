import { useEffect, useRef, useState } from "react";
import { createChart, CrosshairMode } from "lightweight-charts";

function fmtPrice(v) {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(2);
}
function fmtVol(v) {
  if (v == null || isNaN(v)) return "—";
  if (v >= 1_000_000_000) return (v / 1_000_000_000).toFixed(2) + "B";
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(2) + "M";
  if (v >= 1_000) return (v / 1_000).toFixed(2) + "K";
  return v.toFixed(0);
}
function fmtCrosshairTime(t, intraday) {
  // lightweight-charts gives `t` as a UNIX timestamp (number) for time-axis charts
  if (typeof t !== "number") return "";
  const d = new Date(t * 1000);
  if (isNaN(d.getTime())) return "";
  return intraday
    ? d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
}

export default function PriceChart({ candles, chartType = "candle", fit = true, intraday = false }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  // last-bar fallback when cursor is not on the chart
  const lastBar = candles && candles.length ? candles[candles.length - 1] : null;
  const [cross, setCross] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { color: "#0e1117" }, textColor: "#c9d1d9" },
      grid: {
        vertLines: { color: "#21262d" },
        horzLines: { color: "#21262d" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        horzLine: { visible: true, labelVisible: false },
      },
      rightPriceScale: { borderColor: "#30363d" },
      timeScale: {
        borderColor: "#30363d",
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: 460,
    });

    let priceSeries;
    if (chartType === "line") {
      priceSeries = chart.addAreaSeries({
        lineColor: "#58a6ff",
        topColor: "rgba(88, 166, 255, 0.35)",
        bottomColor: "rgba(88, 166, 255, 0.02)",
        lineWidth: 2,
      });
    } else {
      priceSeries = chart.addCandlestickSeries({
        upColor: "#26a69a",
        downColor: "#ef5350",
        borderUpColor: "#26a69a",
        borderDownColor: "#ef5350",
        wickUpColor: "#26a69a",
        wickDownColor: "#ef5350",
      });
    }

    priceSeries.applyOptions({
      autoscaleInfoProvider: (original) => {
        const r = original();
        if (r && r.priceRange) {
          r.priceRange.minValue = Math.max(0, r.priceRange.minValue);
        }
        return r;
      },
    });

    const volSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      color: "#3b4a5a",
    });
    volSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    // Track which raw candle corresponds to which time so we can show full OHLCV
    // for line charts (where the series only carries a value).
    const candleByTime = new Map();

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData || !param.seriesData.size) {
        setCross(null);
        return;
      }
      const data = param.seriesData.get(priceSeries);
      const volData = param.seriesData.get(volSeries);
      const raw = candleByTime.get(param.time);
      if (!data && !raw) {
        setCross(null);
        return;
      }
      setCross({
        time: param.time,
        open: data?.open ?? raw?.open ?? data?.value,
        high: data?.high ?? raw?.high,
        low: data?.low ?? raw?.low,
        close: data?.close ?? raw?.close ?? data?.value,
        volume: volData?.value ?? raw?.volume,
      });
    });

    chartRef.current = { chart, priceSeries, volSeries, candleByTime };

    const onResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      chartRef.current = null;
      setCross(null);
    };
  }, [chartType]);

  useEffect(() => {
    if (!chartRef.current || !candles?.length) return;
    const { priceSeries, volSeries, chart, candleByTime } = chartRef.current;
    candleByTime.clear();
    candles.forEach((c) => candleByTime.set(c.time, c));

    if (chartType === "line") {
      priceSeries.setData(candles.map((c) => ({ time: c.time, value: c.close })));
    } else {
      priceSeries.setData(candles.map((c) => ({
        time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
      })));
    }
    volSeries.setData(candles.map((c) => ({
      time: c.time,
      value: c.volume,
      color: c.close >= c.open ? "rgba(38, 166, 154, 0.5)" : "rgba(239, 83, 80, 0.5)",
    })));
    if (fit) {
      chart.timeScale().fitContent();
    } else {
      chart.timeScale().scrollToRealTime();
    }
  }, [candles, chartType, fit]);

  const display = cross || (lastBar ? {
    time: lastBar.time,
    open: lastBar.open, high: lastBar.high, low: lastBar.low,
    close: lastBar.close, volume: lastBar.volume,
  } : null);

  return (
    <>
      {display && (
        <div className="chart-readout">
          <span className="chart-readout-time">{fmtCrosshairTime(display.time, intraday)}</span>
          <span><span className="ro-label">O</span> {fmtPrice(display.open)}</span>
          <span><span className="ro-label">H</span> {fmtPrice(display.high)}</span>
          <span><span className="ro-label">L</span> {fmtPrice(display.low)}</span>
          <span>
            <span className="ro-label">C</span>{" "}
            <strong className={display.close >= display.open ? "up" : "down"}>
              {fmtPrice(display.close)}
            </strong>
          </span>
          <span><span className="ro-label">Vol</span> {fmtVol(display.volume)}</span>
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", height: 460 }} />
    </>
  );
}
