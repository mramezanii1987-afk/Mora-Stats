import { useCallback, useEffect, useState } from "react";
import AnalysisPane from "./panes/AnalysisPane";
import DataPane from "./panes/DataPane";
import ResultsPane from "./panes/ResultsPane";
import { EngineFailure, engine } from "./api";
import type { Dataset, DesignType, RunPayload, Suggestion, VarType } from "./types";

const DESIGNS: { value: DesignType; label: string; note: string }[] = [
  {
    value: "experimental",
    label: "Experimental",
    note: "People were assigned at random, so differences can be read as effects.",
  },
  {
    value: "quasi_experimental",
    label: "Quasi-experimental",
    note: "Groups existed already, so differences carry whatever else separates them.",
  },
  {
    value: "correlational",
    label: "Correlational",
    note: "Nothing was assigned, so findings are associations.",
  },
];

export default function App() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [design, setDesign] = useState<DesignType>("correlational");
  const [path, setPath] = useState("sample/teaching-trial.csv");
  const [selected, setSelected] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [alpha, setAlpha] = useState(0.05);
  const [payload, setPayload] = useState<RunPayload | null>(null);
  const [error, setError] = useState<{ message: string; action: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const fail = (caught: unknown) => {
    if (caught instanceof EngineFailure) {
      setError({ message: caught.message, action: caught.action });
    } else {
      setError({
        message: "Something went wrong in the interface.",
        action: "Reload the window. The dataset on disk is untouched.",
      });
    }
  };

  const openFile = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await engine.newProject(design, alpha);
      const opened = await engine.open(path, 1);
      setDataset(opened);
      setSelected([]);
      setSuggestions([]);
      setActive(null);
      setPayload(null);
    } catch (caught) {
      fail(caught);
    } finally {
      setBusy(false);
    }
  }, [design, alpha, path]);

  // Live results. Any change to the variables, the test or the threshold
  // recomputes, so there is never an apply step to forget.
  useEffect(() => {
    if (!dataset || selected.length === 0) {
      setSuggestions([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const { suggestions: list } = await engine.suggest(selected);
        if (cancelled) return;
        setSuggestions(list);
        const stillFits = list.find((s) => s.test === active && s.eligible);
        const next = stillFits?.test ?? list.find((s) => s.recommended)?.test ?? null;
        setActive(next ?? null);
      } catch (caught) {
        if (!cancelled) fail(caught);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, selected]);

  useEffect(() => {
    if (!dataset || !active || selected.length === 0) {
      setPayload(null);
      return;
    }
    let cancelled = false;
    setBusy(true);
    (async () => {
      try {
        const next = await engine.run(active, selected, alpha);
        if (!cancelled) {
          setPayload(next);
          setError(null);
        }
      } catch (caught) {
        if (!cancelled) {
          setPayload(null);
          fail(caught);
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dataset, active, selected, alpha]);

  const toggle = (name: string) =>
    setSelected((current) =>
      current.includes(name)
        ? current.filter((n) => n !== name)
        : [...current, name].slice(-2),
    );

  const retype = async (name: string, type: VarType) => {
    try {
      await engine.setVariable(name, { type });
      const refreshed = await engine.open(path, 1);
      setDataset(refreshed);
    } catch (caught) {
      fail(caught);
    }
  };

  const ledger = payload?.ledger;

  return (
    <div className="h-full flex flex-col bg-[var(--color-surface)]">
      <header className="flex items-center gap-4 px-4 h-12 border-b border-[var(--color-rule)] bg-[var(--color-paper)]">
        <span className="font-semibold tracking-tight">MoRa Stats</span>

        <label className="sr-only" htmlFor="path">
          File to open
        </label>
        <input
          id="path"
          value={path}
          onChange={(event) => setPath(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && openFile()}
          className="w-[24rem] px-2 py-1 rounded border border-[var(--color-rule)] bg-[var(--color-surface)] font-[family-name:var(--font-mono)] text-[11px]"
        />
        <button
          onClick={openFile}
          className="px-2.5 py-1 rounded border border-[var(--color-rule)] hover:bg-[var(--color-sunk)]"
        >
          Open
        </button>

        <label className="sr-only" htmlFor="design">
          Study design
        </label>
        <select
          id="design"
          value={design}
          onChange={(event) => setDesign(event.target.value as DesignType)}
          className="px-2 py-1 rounded border border-[var(--color-rule)] bg-[var(--color-surface)]"
          title={DESIGNS.find((d) => d.value === design)?.note}
        >
          {DESIGNS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <div className="ml-auto flex items-center gap-4">
          {ledger && ledger.project_count >= 3 && (
            <span className="text-[11px] text-[var(--color-warn)] tabular">
              {ledger.project_count} tests run, {Math.round(ledger.family_wise_rate * 100)}%
              chance of a false positive among them
            </span>
          )}
          <button
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            className="text-[11px] text-[var(--color-muted)]"
          >
            {theme === "light" ? "Dark" : "Light"}
          </button>
        </div>
      </header>

      <p className="px-4 py-1.5 text-[11px] text-[var(--color-muted)] border-b border-[var(--color-rule)] bg-[var(--color-paper)]">
        {DESIGNS.find((d) => d.value === design)?.note}
      </p>

      <main className="flex-1 min-h-0 grid grid-cols-[minmax(280px,1fr)_minmax(260px,0.85fr)_minmax(440px,1.5fr)]">
        <DataPane dataset={dataset} selected={selected} onToggle={toggle} onRetype={retype} />
        <AnalysisPane
          selected={selected}
          suggestions={suggestions}
          active={active}
          alpha={alpha}
          busy={busy}
          onChoose={setActive}
          onAlpha={setAlpha}
          onClear={() => setSelected([])}
        />
        <ResultsPane payload={payload} error={error} busy={busy} />
      </main>
    </div>
  );
}
