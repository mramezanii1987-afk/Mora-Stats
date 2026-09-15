import { useEffect, useMemo, useState } from "react";
import type { ChartGroup, ChartSpec, ChartView, MosaicCell } from "../types";

/*
  Charts are SVG drawn from numbers the engine computed, including the jitter,
  so the picture is identical on every machine and in every rerun.

  Each analysis offers more than one honest view of the same data and lets the
  reader switch between them. Switching animates, because the point of the
  animation is to show that it is the same observations being redescribed
  rather than a different dataset. Nothing animates on its own, and the whole
  thing holds still when the system asks for reduced motion.

  Hovering or tabbing to any element writes its numbers into the line under the
  chart instead of a floating tooltip, which keeps the figures readable with a
  keyboard and legible when the chart is saved.
*/

const W = 560;
const H = 300;
const PAD = { top: 18, right: 20, bottom: 46, left: 62 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

const VIEW_LABEL: Record<ChartView, string> = {
  intervals: "Means and intervals",
  distribution: "Spread",
  points: "Every case",
  fit: "Fitted line",
  share: "Shares",
  residual: "Departures",
};

export default function Chart({ spec }: { spec: ChartSpec }) {
  const [view, setView] = useState<ChartView>(spec.default_view ?? "points");
  const [readout, setReadout] = useState<string | null>(null);

  useEffect(() => {
    setView(spec.default_view ?? "points");
    setReadout(null);
  }, [spec.title, spec.default_view]);

  if (spec.kind === "none") {
    return (
      <p className="text-[var(--color-muted)] max-w-[52ch]">{spec.reason}</p>
    );
  }

  const views = spec.views ?? [];

  return (
    <figure className="m-0">
      {views.length > 1 && (
        <div role="tablist" aria-label="Chart view" className="inline-flex rounded border border-[var(--color-rule)] overflow-hidden mb-3">
          {views.map((option) => (
            <button
              key={option}
              role="tab"
              aria-selected={view === option}
              onClick={() => setView(option)}
              className={`px-2.5 py-1 text-[11px] border-r border-[var(--color-rule)] last:border-r-0 ${
                view === option
                  ? "bg-[var(--color-accent)] text-white"
                  : "hover:bg-[var(--color-sunk)] text-[var(--color-muted)]"
              }`}
            >
              {VIEW_LABEL[option]}
            </button>
          ))}
        </div>
      )}

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[600px]" role="img" aria-label={spec.title}>
        {spec.kind === "groups" && <Groups spec={spec} view={view} onRead={setReadout} />}
        {spec.kind === "scatter" && <Scatter spec={spec} view={view} onRead={setReadout} />}
        {spec.kind === "mosaic" && <Mosaic spec={spec} view={view} onRead={setReadout} />}
      </svg>

      <figcaption className="mt-1 text-[11px] text-[var(--color-muted)] max-w-[62ch] min-h-[2.4em]">
        {readout ?? caption(spec, view)}
        {spec.thinned && " Drawing shows a sample of the cases; every figure uses all of them."}
      </figcaption>
    </figure>
  );
}

function caption(spec: ChartSpec, view: ChartView): string {
  if (spec.kind === "groups") {
    if (view === "intervals") return "Dots are group means, bars are 95% intervals around them.";
    if (view === "distribution") return "Boxes span the middle half of each group, the line inside is the median, whiskers reach the furthest case within one and a half interquartile ranges.";
    return "One dot per case, spread sideways so they do not overlap. The line is the group mean.";
  }
  if (spec.kind === "scatter") {
    return view === "fit"
      ? "The band is the 95% interval around the fitted line, so it is narrowest where the data is densest."
      : "One dot per case, with no line fitted.";
  }
  return view === "share"
    ? "Bar widths are the share of each row."
    : "Bars show how far each cell sits from what independence would give, in standard residuals.";
}

function scale(min: number, max: number) {
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad;
  const hi = max + pad;
  return {
    y: (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * PLOT_H,
    ticks: [lo, lo + (hi - lo) / 2, hi],
  };
}

const MOTION = "transition-all duration-500 ease-out motion-reduce:transition-none";

function Axis({ ticks, y, label }: { ticks: number[]; y: (v: number) => number; label?: string }) {
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} y1={y(t)} x2={W - PAD.right} y2={y(t)} stroke="var(--color-rule)" strokeWidth="0.5" />
          <text x={PAD.left - 8} y={y(t) + 3} textAnchor="end" fontSize="10" fill="var(--color-muted)" className="tabular">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {label && (
        <text x={16} y={PAD.top + PLOT_H / 2} fontSize="11" fill="var(--color-muted)" textAnchor="middle"
          transform={`rotate(-90 16 ${PAD.top + PLOT_H / 2})`}>
          {label}
        </text>
      )}
    </g>
  );
}

function Groups({ spec, view, onRead }: { spec: ChartSpec; view: ChartView; onRead: (t: string | null) => void }) {
  const groups = spec.groups ?? [];
  const bounds = useMemo(() => {
    const all = groups.flatMap((g) => [g.ci_low, g.ci_high, g.whisker_low, g.whisker_high, ...g.values]);
    return { min: Math.min(...all), max: Math.max(...all) };
  }, [groups]);
  const { y, ticks } = scale(bounds.min, bounds.max);
  const step = PLOT_W / groups.length;

  const describe = (g: ChartGroup) =>
    view === "intervals"
      ? `${g.label}: mean ${g.mean.toFixed(2)} ${spec.unit ?? ""}, 95% interval ${g.ci_low.toFixed(2)} to ${g.ci_high.toFixed(2)}, n = ${g.n}`
      : `${g.label}: median ${g.median.toFixed(2)}, middle half from ${g.q1.toFixed(2)} to ${g.q3.toFixed(2)}, n = ${g.n}`;

  return (
    <g>
      <Axis ticks={ticks} y={y} label={spec.y_label} />
      {groups.map((g, i) => {
        const cx = PAD.left + step * (i + 0.5);
        const width = Math.min(step * 0.46, 54);
        return (
          <g
            key={g.label}
            tabIndex={0}
            role="group"
            aria-label={describe(g)}
            onMouseEnter={() => onRead(describe(g))}
            onMouseLeave={() => onRead(null)}
            onFocus={() => onRead(describe(g))}
            onBlur={() => onRead(null)}
          >
            <rect x={cx - step / 2} y={PAD.top} width={step} height={PLOT_H} fill="transparent" />

            {view === "points" &&
              g.values.map((value, index) => (
                <circle
                  key={index}
                  cx={cx + (g.jitter[index] ?? 0) * width}
                  cy={y(value)}
                  r="2.2"
                  fill="var(--color-accent)"
                  fillOpacity="0.45"
                  className={MOTION}
                />
              ))}

            {view === "distribution" && (
              <g className={MOTION}>
                <line x1={cx} y1={y(g.whisker_high)} x2={cx} y2={y(g.q3)} stroke="var(--color-accent)" strokeWidth="1" />
                <line x1={cx} y1={y(g.q1)} x2={cx} y2={y(g.whisker_low)} stroke="var(--color-accent)" strokeWidth="1" />
                <rect x={cx - width / 2} y={y(g.q3)} width={width} height={Math.max(y(g.q1) - y(g.q3), 1)}
                  fill="var(--color-accent-soft)" stroke="var(--color-accent)" strokeWidth="1" className={MOTION} />
                <line x1={cx - width / 2} y1={y(g.median)} x2={cx + width / 2} y2={y(g.median)}
                  stroke="var(--color-accent)" strokeWidth="2" className={MOTION} />
              </g>
            )}

            {(view === "intervals" || view === "points") && (
              <g className={MOTION}>
                {view === "intervals" && (
                  <>
                    <line x1={cx} y1={y(g.ci_low)} x2={cx} y2={y(g.ci_high)} stroke="var(--color-accent)" strokeWidth="1.6" />
                    <line x1={cx - 9} y1={y(g.ci_low)} x2={cx + 9} y2={y(g.ci_low)} stroke="var(--color-accent)" strokeWidth="1.2" />
                    <line x1={cx - 9} y1={y(g.ci_high)} x2={cx + 9} y2={y(g.ci_high)} stroke="var(--color-accent)" strokeWidth="1.2" />
                    <circle cx={cx} cy={y(g.mean)} r="4" fill="var(--color-accent)" />
                  </>
                )}
                {view === "points" && (
                  <line x1={cx - width} y1={y(g.mean)} x2={cx + width} y2={y(g.mean)}
                    stroke="var(--color-ink)" strokeWidth="1.4" />
                )}
              </g>
            )}

            <text x={cx} y={H - PAD.bottom + 18} textAnchor="middle" fontSize="11" fill="var(--color-ink)">
              {g.label}
            </text>
            <text x={cx} y={H - PAD.bottom + 31} textAnchor="middle" fontSize="10" fill="var(--color-faint)" className="tabular">
              n = {g.n}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Scatter({ spec, view, onRead }: { spec: ChartSpec; view: ChartView; onRead: (t: string | null) => void }) {
  const points = spec.points ?? [];
  const band = spec.band ?? [];
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const lows = band.map((b) => b.low);
  const highs = band.map((b) => b.high);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const { y, ticks } = scale(Math.min(...ys, ...lows), Math.max(...ys, ...highs));
  const x = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin || 1)) * PLOT_W;

  const area =
    band.length > 1
      ? `${band.map((b) => `${x(b.x)},${y(b.high)}`).join(" ")} ${[...band]
          .reverse()
          .map((b) => `${x(b.x)},${y(b.low)}`)
          .join(" ")}`
      : "";

  return (
    <g>
      <Axis ticks={ticks} y={y} label={spec.y_label} />
      {view === "fit" && area && (
        <polygon points={area} fill="var(--color-accent)" fillOpacity="0.14" className={MOTION} />
      )}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={x(p.x)}
          cy={y(p.y)}
          r="2.6"
          fill="var(--color-accent)"
          fillOpacity="0.5"
          tabIndex={-1}
          onMouseEnter={() => onRead(`${spec.x_label} ${p.x.toFixed(2)}, ${spec.y_label} ${p.y.toFixed(2)}`)}
          onMouseLeave={() => onRead(null)}
        />
      ))}
      {view === "fit" && spec.fit && (
        <line x1={x(spec.fit.x1)} y1={y(spec.fit.y1)} x2={x(spec.fit.x2)} y2={y(spec.fit.y2)}
          stroke="var(--color-ink)" strokeWidth="1.3" className={MOTION} />
      )}
      <text x={PAD.left + PLOT_W / 2} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--color-muted)">
        {spec.x_label}
      </text>
    </g>
  );
}

function Mosaic({ spec, view, onRead }: { spec: ChartSpec; view: ChartView; onRead: (t: string | null) => void }) {
  const rows = spec.rows ?? [];
  const columns = spec.columns ?? [];
  const cells = spec.cells ?? [];
  const rowHeight = PLOT_H / (rows.length || 1);
  const maxResidual = Math.max(2, ...cells.map((c) => Math.abs(c.residual)));

  const describe = (cell: MosaicCell) =>
    `${cell.row}, ${cell.column}: ${cell.count} cases, ${(cell.share * 100).toFixed(0)}% of the row` +
    (cell.expected === null ? "" : `, independence would give ${cell.expected.toFixed(1)}, residual ${cell.residual.toFixed(2)}`);

  return (
    <g>
      {rows.map((row, i) => {
        let offset = 0;
        return (
          <g key={row}>
            {columns.map((column) => {
              const cell = cells.find((c) => c.row === row && c.column === column);
              if (!cell) return null;
              const full = view === "share" ? cell.share : Math.abs(cell.residual) / maxResidual;
              const width = Math.max(full * PLOT_W - 2, 0);
              const cx = view === "share" ? PAD.left + offset : PAD.left;
              if (view === "share") offset += full * PLOT_W;
              const strong = Math.abs(cell.residual) > 1.96;
              const bandHeight = (rowHeight - 12) / (view === "share" ? 1 : columns.length);
              const cy =
                view === "share"
                  ? PAD.top + i * rowHeight + 6
                  : PAD.top + i * rowHeight + 6 + columns.indexOf(column) * bandHeight;
              return (
                <g
                  key={column}
                  tabIndex={0}
                  role="img"
                  aria-label={describe(cell)}
                  onMouseEnter={() => onRead(describe(cell))}
                  onMouseLeave={() => onRead(null)}
                  onFocus={() => onRead(describe(cell))}
                  onBlur={() => onRead(null)}
                >
                  <rect
                    x={cx}
                    y={cy}
                    width={width}
                    height={Math.max(bandHeight - 2, 2)}
                    fill={strong ? "var(--color-accent)" : "var(--color-accent-soft)"}
                    className={MOTION}
                  />
                  {width > 58 && (
                    <text x={cx + 8} y={cy + Math.max(bandHeight - 2, 2) / 2 + 3.5} fontSize="10"
                      fill={strong ? "#fff" : "var(--color-ink)"} className="tabular">
                      {column} {view === "share" ? `${(cell.share * 100).toFixed(0)}%` : cell.residual.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={PAD.left - 8} y={PAD.top + i * rowHeight + rowHeight / 2 + 3} textAnchor="end" fontSize="11" fill="var(--color-ink)">
              {row}
            </text>
          </g>
        );
      })}
    </g>
  );
}
