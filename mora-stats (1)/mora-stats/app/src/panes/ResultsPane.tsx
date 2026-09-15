import { useRef, useState } from "react";
import Chart from "../components/Chart";
import EffectBar from "../components/EffectBar";
import Table, { toMarkdown, toTsv } from "../components/Table";
import type { RunPayload, Threat } from "../types";

/*
  The reading order is fixed: the figure that carries the result, the tables
  behind it, the sentence to paste into a manuscript, the plain reading of
  what it means, then what could be wrong with it.

  Everything here is copyable and nothing is a screenshot: tables copy as
  cells, the chart saves as SVG, and the whole block copies as one piece for
  people who work in a document rather than in an app.
*/

const BOUNDED = new Set([
  "Hedges' g",
  "Pearson's r",
  "Spearman's rho",
  "rank-biserial r",
  "Cramer's V",
  "omega squared",
  "epsilon squared",
]);

interface Props {
  payload: RunPayload | null;
  error: { message: string; action: string } | null;
  busy: boolean;
}

export default function ResultsPane({ payload, error, busy }: Props) {
  const [copied, setCopied] = useState<string | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);

  if (error) {
    return (
      <section className="h-full p-8">
        <div className="max-w-[54ch] border-l-2 border-[var(--color-serious)] bg-[var(--color-serious-soft)] rounded-r px-4 py-3">
          <p className="font-medium">{error.message}</p>
          <p className="mt-1 text-[var(--color-muted)]">{error.action}</p>
        </div>
      </section>
    );
  }

  if (!payload) {
    return (
      <section className="h-full flex items-center justify-center p-10">
        <p className="max-w-[38ch] text-center text-[var(--color-muted)] leading-relaxed">
          Pick variables on the left and the result appears here, with the tables, the
          sentence for your manuscript, and a plain reading of what it means.
        </p>
      </section>
    );
  }

  const { result, interpretation, chart } = payload;
  const effect = result.effect;
  const bounded = effect ? BOUNDED.has(effect.name) : false;
  const headline = effect
    ? bounded
      ? effect.value.toFixed(2).replace(/^(-?)0\./, "$1.")
      : effect.value.toFixed(2)
    : "—";

  const flash = (label: string) => {
    setCopied(label);
    window.setTimeout(() => setCopied(null), 1600);
  };

  const copyEverything = async () => {
    const parts = [
      result.label,
      "",
      ...interpretation.tables.map(toTsv),
      "",
      interpretation.apa,
      "",
      interpretation.sentences.join(" "),
    ];
    if (interpretation.threats.length) {
      parts.push("", "Checks");
      interpretation.threats.forEach((t) => parts.push(`- ${t.message} ${t.action}`));
    }
    await navigator.clipboard.writeText(parts.join("\n"));
    flash("all");
  };

  const copyAsMarkdown = async () => {
    await navigator.clipboard.writeText(
      [
        `## ${result.label}`,
        "",
        ...interpretation.tables.map(toMarkdown),
        "",
        interpretation.apa,
      ].join("\n\n"),
    );
    flash("markdown");
  };

  const saveChart = () => {
    const svg = chartRef.current?.querySelector("svg");
    if (!svg) return;
    const clone = svg.cloneNode(true) as SVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    // Resolve the theme variables so the saved file stands on its own.
    const style = getComputedStyle(document.documentElement);
    clone.querySelectorAll<SVGElement>("*").forEach((node) => {
      (["stroke", "fill"] as const).forEach((attribute) => {
        const value = node.getAttribute(attribute);
        if (value?.startsWith("var(")) {
          const name = value.slice(4, -1).trim();
          node.setAttribute(attribute, style.getPropertyValue(name).trim() || "#000");
        }
      });
    });
    const blob = new Blob([clone.outerHTML], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${result.analysis}-${result.variables.join("-")}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="h-full overflow-y-auto group/results">
      <header className="sticky top-0 z-10 bg-[var(--color-surface)] px-7 pt-5 pb-3 border-b border-[var(--color-rule)]">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h2 className="text-[15px] font-semibold leading-tight">{result.label}</h2>
            <p className="text-[11px] text-[var(--color-muted)] mt-0.5 tabular">
              {result.variables.join(", ")} · n = {result.n}
              {result.n_missing > 0 && `, ${result.n_missing} dropped`}
              {busy && " · updating"}
            </p>
          </div>
          <div className="flex gap-3 shrink-0 pt-0.5">
            <button
              onClick={copyEverything}
              className="px-2.5 py-1 rounded border border-[var(--color-rule)] hover:bg-[var(--color-sunk)] text-[11px]"
            >
              {copied === "all" ? "Copied" : "Copy results"}
            </button>
            <button
              onClick={copyAsMarkdown}
              className="px-2.5 py-1 rounded border border-[var(--color-rule)] hover:bg-[var(--color-sunk)] text-[11px]"
            >
              {copied === "markdown" ? "Copied" : "Markdown"}
            </button>
          </div>
        </div>
      </header>

      <div className="px-7 py-6 space-y-8 max-w-[860px]">
        {effect && (
          <section className="flex flex-wrap items-end gap-x-10 gap-y-4">
            <div>
              <div className="text-[40px] leading-none font-semibold tabular tracking-tight">
                {headline}
              </div>
              <div className="mt-1.5 text-[12px] text-[var(--color-muted)]">{effect.name}</div>
            </div>
            <div className="flex-1 min-w-[240px]">
              <EffectBar effect={effect} power={result.power} bounded={bounded} />
            </div>
          </section>
        )}

        <section>
          {interpretation.tables.map((table) => (
            <Table key={table.key} table={table} />
          ))}
        </section>

        <Block title="Sentence for your manuscript">
          <p className="prose-block">{interpretation.apa}</p>
        </Block>

        <Block title="What this means">
          <p className="prose-block">{interpretation.sentences.join(" ")}</p>
        </Block>

        {interpretation.threats.length > 0 && (
          <Block title={`Checks on this result (${interpretation.threats.length})`}>
            <ul className="space-y-2.5">
              {interpretation.threats.map((threat, index) => (
                <ThreatRow key={`${threat.code}-${index}`} threat={threat} />
              ))}
            </ul>
          </Block>
        )}

        {interpretation.multiplicity && (
          <Block title="Multiple comparisons">
            <p className="max-w-[64ch] text-[var(--color-muted)] leading-relaxed">
              {interpretation.multiplicity}
            </p>
          </Block>
        )}

        <Block
          title="Chart"
          action={
            chart.kind !== "none" ? (
              <button
                onClick={saveChart}
                className="text-[11px] text-[var(--color-accent)] underline underline-offset-2"
              >
                Save as SVG
              </button>
            ) : undefined
          }
        >
          <div ref={chartRef}>
            <Chart spec={chart} />
          </div>
        </Block>
      </div>
    </section>
  );
}

function Block({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="flex items-baseline justify-between border-b border-[var(--color-rule)] pb-1 mb-3">
        <h3 className="block-heading">{title}</h3>
        {action}
      </div>
      {children}
    </section>
  );
}

function ThreatRow({ threat }: { threat: Threat }) {
  const tone =
    threat.severity === "serious"
      ? "border-[var(--color-serious)] bg-[var(--color-serious-soft)]"
      : threat.severity === "warn"
        ? "border-[var(--color-warn)] bg-[var(--color-warn-soft)]"
        : "border-[var(--color-rule)] bg-[var(--color-sunk)]";
  return (
    <li className={`border-l-2 pl-3.5 py-2 pr-3 rounded-r ${tone}`}>
      <p className="max-w-[62ch] leading-relaxed">{threat.message}</p>
      <p className="max-w-[62ch] mt-1 text-[var(--color-muted)] leading-relaxed">
        {threat.action}
      </p>
    </li>
  );
}
