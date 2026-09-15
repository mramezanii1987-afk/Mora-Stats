import type { Suggestion } from "../types";

/*
  No modal dialogs and no confirm step. Picking variables refreshes the list
  of tests that fit, choosing one runs it, and changing an option reruns it.
  A test that does not fit stays visible with the reason attached, because
  hiding it just moves the confusion somewhere else.
*/

interface Props {
  selected: string[];
  suggestions: Suggestion[];
  active: string | null;
  alpha: number;
  busy: boolean;
  onChoose: (test: string) => void;
  onAlpha: (alpha: number) => void;
  onClear: () => void;
}

export default function AnalysisPane({
  selected,
  suggestions,
  active,
  alpha,
  busy,
  onChoose,
  onAlpha,
  onClear,
}: Props) {
  return (
    <section className="h-full flex flex-col min-h-0 border-x border-[var(--color-rule)]">
      <header className="px-4 py-3 border-b border-[var(--color-rule)] flex items-baseline justify-between">
        <h2 className="text-[13px] font-semibold">Analysis</h2>
        {busy && <span className="text-[11px] text-[var(--color-muted)]">working</span>}
      </header>

      <div className="px-4 py-3 border-b border-[var(--color-rule)]">
        {selected.length === 0 ? (
          <p className="text-[var(--color-muted)]">
            Pick one or two variables on the left. The first is the outcome, the second
            splits or pairs with it.
          </p>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            {selected.map((name, index) => (
              <span
                key={name}
                className="px-2 py-1 rounded border border-[var(--color-rule)] bg-[var(--color-sunk)]"
              >
                <span className="text-[10px] text-[var(--color-muted)] mr-1 tabular">
                  {index === 0 ? "outcome" : "with"}
                </span>
                {name}
              </span>
            ))}
            <button
              onClick={onClear}
              className="text-[11px] text-[var(--color-muted)] underline underline-offset-2"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <ul className="flex-1 min-h-0 overflow-y-auto">
        {suggestions.map((suggestion) => {
          const isActive = suggestion.test === active;
          return (
            <li key={suggestion.test}>
              <button
                disabled={!suggestion.eligible}
                onClick={() => onChoose(suggestion.test)}
                className={`w-full text-left px-4 py-2.5 border-b border-[var(--color-rule)] ${
                  isActive ? "bg-[var(--color-accent-soft)]" : ""
                } ${
                  suggestion.eligible
                    ? "cursor-pointer hover:bg-[var(--color-sunk)]"
                    : "opacity-55 cursor-not-allowed"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium">{suggestion.label}</span>
                  {suggestion.recommended && (
                    <span className="text-[10px] px-1.5 py-[1px] rounded-full border border-[var(--color-accent)] text-[var(--color-accent)]">
                      suggested
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-[var(--color-muted)] mt-0.5 max-w-[46ch]">
                  {suggestion.reason}
                </p>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="px-4 py-3 border-t border-[var(--color-rule)] flex items-center gap-3">
        <label htmlFor="alpha" className="text-[11px] text-[var(--color-muted)]">
          Threshold
        </label>
        <select
          id="alpha"
          value={alpha}
          onChange={(event) => onAlpha(Number(event.target.value))}
          className="bg-transparent border border-[var(--color-rule)] rounded px-1.5 py-0.5 tabular"
        >
          {[0.1, 0.05, 0.01, 0.001].map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </div>
    </section>
  );
}
