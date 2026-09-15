import type { Dataset, VarType, VariableSummary } from "../types";

/*
  The data pane is a variable list first and a grid second. People pick
  variables from the list far more often than they read individual cells, so
  the list gets the attention and the grid sits underneath it.
*/

const TYPE_LABEL: Record<VarType, string> = {
  continuous: "continuous",
  ordinal: "ordinal",
  nominal: "nominal",
};

interface Props {
  dataset: Dataset | null;
  selected: string[];
  onToggle: (name: string) => void;
  onRetype: (name: string, type: VarType) => void;
}

export default function DataPane({ dataset, selected, onToggle, onRetype }: Props) {
  if (!dataset) {
    return (
      <section className="h-full flex items-center justify-center p-8 text-center">
        <p className="max-w-[34ch] text-[var(--color-muted)]">
          Open a .csv or .xlsx file to start. Nothing is uploaded and nothing leaves this
          machine.
        </p>
      </section>
    );
  }

  const summaries = new Map(dataset.summary.map((s) => [s.name, s]));

  return (
    <section className="h-full flex flex-col min-h-0">
      <header className="px-4 py-3 border-b border-[var(--color-rule)] flex items-baseline justify-between">
        <h2 className="text-[13px] font-semibold">Variables</h2>
        <span className="text-[11px] text-[var(--color-muted)] tabular">
          {dataset.n_rows} rows, {dataset.columns.length} columns
        </span>
      </header>

      <ul className="overflow-y-auto flex-1 min-h-0">
        {dataset.variables.map((variable) => {
          const summary = summaries.get(variable.name) as VariableSummary | undefined;
          const isSelected = selected.includes(variable.name);
          const role = selected.indexOf(variable.name);
          return (
            <li key={variable.name}>
              <div
                role="button"
                tabIndex={0}
                onClick={() => onToggle(variable.name)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onToggle(variable.name);
                  }
                }}
                className={`w-full text-left px-4 py-2 border-b border-[var(--color-rule)] cursor-pointer ${
                  isSelected ? "bg-[var(--color-accent-soft)]" : "hover:bg-[var(--color-sunk)]"
                }`}
              >
                <div className="flex items-center gap-2">
                  {isSelected && (
                    <span className="text-[10px] text-[var(--color-accent)] font-semibold tabular">
                      {role + 1}
                    </span>
                  )}
                  <span className="font-medium truncate">{variable.label ?? variable.name}</span>
                  <span className="ml-auto font-[family-name:var(--font-mono)] text-[10px] text-[var(--color-accent)]">
                    {summary?.sparkline}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-1 text-[11px] text-[var(--color-muted)]">
                  <select
                    aria-label={`Type of ${variable.name}`}
                    value={variable.type}
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => onRetype(variable.name, event.target.value as VarType)}
                    className="bg-transparent border border-[var(--color-rule)] rounded px-1 py-[1px] text-[11px]"
                  >
                    {Object.keys(TYPE_LABEL).map((type) => (
                      <option key={type} value={type}>
                        {TYPE_LABEL[type as VarType]}
                      </option>
                    ))}
                  </select>
                  <span className="tabular">
                    {summary?.missing ? `${summary.missing} missing` : "complete"}
                  </span>
                  {summary?.mean !== undefined && (
                    <span className="tabular">M {summary.mean.toFixed(2)}</span>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="border-t border-[var(--color-rule)] overflow-auto max-h-[38%]">
        <table className="w-full text-[11px] tabular border-collapse">
          <thead className="sticky top-0 bg-[var(--color-surface)]">
            <tr>
              {dataset.columns.map((column) => (
                <th
                  key={column}
                  className="text-left font-medium px-3 py-1.5 border-b border-[var(--color-rule)] text-[var(--color-muted)] whitespace-nowrap"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataset.preview.map((row, index) => (
              <tr key={index} className="even:bg-[var(--color-sunk)]">
                {dataset.columns.map((column) => (
                  <td key={column} className="px-3 py-1 whitespace-nowrap">
                    {formatCell(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {dataset.truncated && (
          <p className="px-3 py-2 text-[11px] text-[var(--color-faint)]">
            Showing the first 200 rows. Every analysis uses all {dataset.n_rows}.
          </p>
        )}
      </div>
    </section>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return String(value);
}
