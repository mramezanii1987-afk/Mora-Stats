import { useState } from "react";
import type { ResultTable } from "../types";

/*
  A journal table, not a spreadsheet dump. Rules above and below the header
  and one under the last row, nothing vertical, figures on tabular numerals
  so decimal points line up down the column, and a note underneath in the
  place a reader already looks for it.

  Copy produces tab separated text, which pastes into Word as a real table
  and into Excel as real cells. The same table also copies as Markdown for
  anyone drafting somewhere else.
*/

export default function Table({ table }: { table: ResultTable }) {
  const [copied, setCopied] = useState<string | null>(null);

  const copy = async (kind: "cells" | "markdown") => {
    const text = kind === "cells" ? toTsv(table) : toMarkdown(table);
    await navigator.clipboard.writeText(text);
    setCopied(kind);
    window.setTimeout(() => setCopied(null), 1500);
  };

  return (
    <figure className="m-0 mb-6">
      <figcaption className="flex items-baseline justify-between gap-4 mb-1.5">
        <span className="text-[12px] font-semibold">{table.title}</span>
        <span className="flex gap-3 opacity-0 focus-within:opacity-100 group-hover/results:opacity-100 transition-opacity">
          <button
            onClick={() => copy("cells")}
            className="text-[11px] text-[var(--color-accent)] underline underline-offset-2"
          >
            {copied === "cells" ? "Copied" : "Copy table"}
          </button>
          <button
            onClick={() => copy("markdown")}
            className="text-[11px] text-[var(--color-muted)] underline underline-offset-2"
          >
            {copied === "markdown" ? "Copied" : "Markdown"}
          </button>
        </span>
      </figcaption>

      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse tabular text-[12.5px]">
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  className={`font-medium py-1.5 pr-6 last:pr-0 border-y border-[var(--color-ink)] whitespace-nowrap ${
                    column.align === "left" ? "text-left" : "text-right"
                  }`}
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, index) => (
              <tr key={index} className="border-b border-[var(--color-rule)] last:border-[var(--color-ink)]">
                {table.columns.map((column) => {
                  const flagged = row[`${column.key}__flag`] === true;
                  const isEmphasis = column.key === table.emphasis;
                  return (
                    <td
                      key={column.key}
                      className={`py-[7px] pr-6 last:pr-0 whitespace-nowrap ${
                        column.align === "left" ? "text-left" : "text-right"
                      } ${isEmphasis ? "font-semibold text-[var(--color-accent)]" : ""} ${
                        flagged ? "text-[var(--color-accent)] font-medium" : ""
                      }`}
                    >
                      {String(row[column.key] ?? "")}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {table.note && (
        <p className="mt-1.5 text-[11px] leading-snug text-[var(--color-muted)] max-w-[62ch]">
          <span className="italic">Note.</span> {table.note}
        </p>
      )}
    </figure>
  );
}

export function toTsv(table: ResultTable): string {
  const header = table.columns.map((c) => c.label).join("\t");
  const body = table.rows.map((row) =>
    table.columns.map((c) => String(row[c.key] ?? "")).join("\t"),
  );
  return [table.title, header, ...body].join("\n");
}

export function toMarkdown(table: ResultTable): string {
  const header = `| ${table.columns.map((c) => c.label).join(" | ")} |`;
  const divider = `| ${table.columns
    .map((c) => (c.align === "left" ? ":---" : "---:"))
    .join(" | ")} |`;
  const body = table.rows.map(
    (row) => `| ${table.columns.map((c) => String(row[c.key] ?? "")).join(" | ")} |`,
  );
  const note = table.note ? [``, `*Note.* ${table.note}`] : [];
  return [`**${table.title}**`, ``, header, divider, ...body, ...note].join("\n");
}
