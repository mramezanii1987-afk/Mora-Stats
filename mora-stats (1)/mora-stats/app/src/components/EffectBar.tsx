import type { Estimate, PowerSummary } from "../types";

/*
  The one place this app spends its boldness. It draws the estimate, the
  interval around it, and a marker for the smallest effect this design could
  reliably catch. Seeing the interval straddle that marker is the fastest
  way to understand why a result is not settled, which no row of numbers
  conveys at a glance.
*/

interface Props {
  effect: Estimate;
  power: PowerSummary;
  bounded: boolean;
}

export default function EffectBar({ effect, power, bounded }: Props) {
  const lo = effect.ci_low ?? effect.value;
  const hi = effect.ci_high ?? effect.value;
  const mde = power.mde;
  const marker = mde === null ? null : effect.value < 0 ? -mde : mde;
  const candidates = [lo, hi, effect.value, 0, marker ?? 0];
  const rawMin = Math.min(...candidates);
  const rawMax = Math.max(...candidates);
  const pad = (rawMax - rawMin) * 0.12 || 0.5;
  const min = rawMin - pad;
  const max = rawMax + pad;
  const x = (value: number) => ((value - min) / (max - min)) * 100;
  const zeroInside = lo <= 0 && hi >= 0;
  const format = (value: number) =>
    bounded ? value.toFixed(2).replace(/^(-?)0\./, "$1.") : value.toFixed(2);

  return (
    <figure className="m-0">
      <svg
        viewBox="0 0 100 26"
        preserveAspectRatio="none"
        className="w-full h-[44px]"
        role="img"
        aria-label={`${effect.name} ${format(effect.value)}, interval ${format(lo)} to ${format(hi)}`}
      >
        <line x1="0" y1="13" x2="100" y2="13" stroke="var(--color-rule)" strokeWidth="0.4" />
        {min < 0 && max > 0 && (
          <line x1={x(0)} y1="4" x2={x(0)} y2="22" stroke="var(--color-rule)" strokeWidth="0.8" />
        )}
        {marker !== null && (
          <line
            x1={x(marker)}
            y1="5"
            x2={x(marker)}
            y2="21"
            stroke="var(--color-muted)"
            strokeWidth="0.6"
            strokeDasharray="1.6 1.4"
          />
        )}
        <line x1={x(lo)} y1="13" x2={x(hi)} y2="13" stroke="var(--color-accent)" strokeWidth="1.8" />
        <line x1={x(lo)} y1="9" x2={x(lo)} y2="17" stroke="var(--color-accent)" strokeWidth="1" />
        <line x1={x(hi)} y1="9" x2={x(hi)} y2="17" stroke="var(--color-accent)" strokeWidth="1" />
        <circle cx={x(effect.value)} cy="13" r="2.2" fill="var(--color-accent)" />
      </svg>
      <figcaption className="flex justify-between text-[11px] text-[var(--color-muted)] tabular">
        <span>{format(lo)}</span>
        <span>
          {zeroInside ? "interval includes zero" : "interval excludes zero"}
          {marker !== null && `, dashed line marks ${format(Math.abs(marker))}`}
        </span>
        <span>{format(hi)}</span>
      </figcaption>
    </figure>
  );
}
