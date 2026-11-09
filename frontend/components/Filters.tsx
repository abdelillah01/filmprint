"use client";

import { useId, useState } from "react";
import type { CatalogMeta, Filters as FiltersState } from "../lib/api";
import { EMPTY_FILTERS, activeFilterCount } from "../lib/profile";
import { prettyGenre } from "../lib/title";

interface FiltersProps {
  meta: CatalogMeta;
  value: FiltersState;
  onChange: (next: FiltersState) => void;
}

const RATINGS = [6, 7, 8];

function toYear(raw: string): number | undefined {
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 1800 && n <= 2100 ? n : undefined;
}

const field =
  "h-9 rounded-lg border border-line bg-card px-3 text-sm text-ink outline-none transition-colors duration-150 ease-smooth focus:border-line-strong placeholder:text-ink-muted";

/** Keeps what the user is typing as text; reports a number only once it is a valid year. */
function YearInput({
  label,
  placeholder,
  value,
  onChange,
}: {
  label: string;
  placeholder: string;
  value: number | undefined;
  onChange: (year: number | undefined) => void;
}) {
  const [text, setText] = useState(value == null ? "" : String(value));
  // Follow external changes (for example "Clear filters") without an effect: if the typed text
  // no longer matches the value we were given, show the value.
  const shown = toYear(text) === value ? text : value == null ? "" : String(value);

  return (
    <input
      className={`${field} w-24 font-mono tabular-nums`}
      inputMode="numeric"
      placeholder={placeholder}
      aria-label={label}
      value={shown}
      onChange={(e) => {
        setText(e.target.value);
        onChange(toYear(e.target.value));
      }}
    />
  );
}

export default function Filters({ meta, value, onChange }: FiltersProps) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const count = activeFilterCount(value);

  // off -> include -> exclude -> off
  const cycleGenre = (g: string) => {
    if (value.genres.includes(g)) {
      onChange({
        ...value,
        genres: value.genres.filter((x) => x !== g),
        exclude_genres: [...value.exclude_genres, g],
      });
    } else if (value.exclude_genres.includes(g)) {
      onChange({ ...value, exclude_genres: value.exclude_genres.filter((x) => x !== g) });
    } else {
      onChange({ ...value, genres: [...value.genres, g] });
    }
  };

  return (
    <div className="max-w-3xl">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={panelId}
        className="inline-flex h-8 items-center gap-2 text-sm text-ink-2 transition-colors duration-150 ease-smooth hover:text-ink"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform duration-150 ease-smooth ${open ? "rotate-90" : ""}`}
          aria-hidden
        >
          <path d="m9 18 6-6-6-6" />
        </svg>
        Filters
        {count > 0 && (
          <span className="font-mono text-xs tabular-nums text-accent">{count} active</span>
        )}
      </button>

      {open && (
        <div
          id={panelId}
          className="mt-2 flex flex-col gap-5 rounded-xl border border-line bg-card p-5"
        >
          <div>
            <div className="mb-2 flex items-baseline justify-between gap-4">
              <span className="text-sm text-ink">Genres</span>
              <span className="text-xs text-ink-muted">
                Click to include, again to exclude, again to clear
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {meta.genres.map((g) => {
                const included = value.genres.includes(g);
                const excluded = value.exclude_genres.includes(g);
                return (
                  <button
                    key={g}
                    type="button"
                    onClick={() => cycleGenre(g)}
                    aria-label={`${prettyGenre(g)}: ${
                      included ? "included" : excluded ? "excluded" : "not set"
                    }`}
                    className={`h-8 rounded-lg border px-3 text-sm transition-[color,background-color,border-color] duration-150 ease-smooth ${
                      included
                        ? "border-accent bg-raised text-ink"
                        : excluded
                          ? "border-line bg-bg text-ink-muted line-through"
                          : "border-line bg-card text-ink-2 hover:border-line-strong hover:text-ink"
                    }`}
                  >
                    {included && <span className="mr-1 text-accent">+</span>}
                    {excluded && <span className="mr-1 no-underline">&minus;</span>}
                    {prettyGenre(g)}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-5">
            <div>
              <span className="mb-2 block text-sm text-ink">Year</span>
              <div className="flex items-center gap-2">
                <YearInput
                  label="From year"
                  placeholder={String(meta.year_min ?? "")}
                  value={value.year_min}
                  onChange={(year_min) => onChange({ ...value, year_min })}
                />
                <span className="text-ink-muted">to</span>
                <YearInput
                  label="To year"
                  placeholder={String(meta.year_max ?? "")}
                  value={value.year_max}
                  onChange={(year_max) => onChange({ ...value, year_max })}
                />
              </div>
            </div>

            <div>
              <label htmlFor={`${panelId}-rating`} className="mb-2 block text-sm text-ink">
                Minimum IMDb rating
              </label>
              <select
                id={`${panelId}-rating`}
                className={`${field} font-mono tabular-nums`}
                value={value.min_rating ?? ""}
                onChange={(e) =>
                  onChange({
                    ...value,
                    min_rating: e.target.value === "" ? undefined : Number(e.target.value),
                  })
                }
              >
                <option value="">Any</option>
                {RATINGS.map((r) => (
                  <option key={r} value={r}>
                    {r}+
                  </option>
                ))}
              </select>
            </div>

            {count > 0 && (
              <button
                type="button"
                onClick={() => onChange(EMPTY_FILTERS)}
                className="h-9 text-sm text-ink-muted underline-offset-4 transition-colors duration-150 ease-smooth hover:text-ink hover:underline"
              >
                Clear filters
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
