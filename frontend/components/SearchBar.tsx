"use client";

import { useEffect, useId, useRef, useState } from "react";
import { searchTitles } from "../lib/api";
import { parseTitle } from "../lib/title";

interface SearchBarProps {
  /** Called with a catalog title (from a suggestion) or the raw text the user typed. */
  onAdd: (title: string) => void;
  /** Slimmer variant for the navigation bar. */
  compact?: boolean;
}

const iconProps = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function SearchIcon() {
  return (
    <svg {...iconProps} className="ml-4 shrink-0 text-ink-muted" aria-hidden>
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function ClearIcon() {
  return (
    <svg {...iconProps} aria-hidden>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

export default function SearchBar({ onAdd, compact = false }: SearchBarProps) {
  const [text, setText] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const listId = useId();
  const wrapRef = useRef<HTMLDivElement>(null);

  // Debounced autocomplete; stale requests are aborted.
  useEffect(() => {
    const q = text.trim();
    if (q.length < 2) return;
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const titles = await searchTitles(q, ctrl.signal);
        setSuggestions(titles);
        setActive(-1);
      } catch {
        /* aborted or backend down: just show no suggestions */
      }
    }, 150);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [text]);

  const add = (title: string) => {
    const t = title.trim();
    if (!t) return;
    onAdd(t);
    setText("");
    setSuggestions([]);
    setOpen(false);
    setActive(-1);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" && visible.length) {
      e.preventDefault();
      setOpen(true);
      setActive((i) => (i + 1) % visible.length);
    } else if (e.key === "ArrowUp" && visible.length) {
      e.preventDefault();
      setActive((i) => (i <= 0 ? visible.length - 1 : i - 1));
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    add(open && active >= 0 && visible[active] ? visible[active] : text);
  };

  // Too-short input shows nothing, whatever the last response was.
  const visible = text.trim().length >= 2 ? suggestions : [];
  const showList = open && visible.length > 0;

  return (
    <div
      ref={wrapRef}
      className={`relative ${compact ? "w-full" : "max-w-2xl"}`}
      onBlur={(e) => {
        if (!wrapRef.current?.contains(e.relatedTarget as Node)) setOpen(false);
      }}
    >
      <form
        onSubmit={submit}
        className={`flex items-center rounded-xl border border-line bg-card transition-colors duration-150 ease-smooth focus-within:border-line-strong ${compact ? "h-10" : "h-12"}`}
      >
        <SearchIcon />
        <input
          className="h-full min-w-0 flex-1 bg-transparent px-3 text-[15px] text-ink outline-none placeholder:text-ink-muted"
          placeholder="Add a film you like"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded={showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
          aria-label="Add a film you like"
          autoComplete="off"
        />
        {text && (
          <button
            type="button"
            onClick={() => {
              setText("");
              setSuggestions([]);
            }}
            className="mr-1 flex h-8 w-8 items-center justify-center rounded-lg text-ink-muted transition-colors duration-150 ease-smooth hover:text-ink"
            aria-label="Clear search"
          >
            <ClearIcon />
          </button>
        )}
        <button
          type="submit"
          className={`shrink-0 rounded-md bg-accent text-sm font-medium text-accent-ink transition-[transform,opacity] duration-150 ease-smooth hover:opacity-90 active:scale-[0.98] ${compact ? "m-1 h-8 px-4" : "m-1.5 h-9 px-5"}`}
        >
          Add
        </button>
      </form>

      {showList && (
        <ul
          id={listId}
          role="listbox"
          className="absolute left-0 right-0 top-[calc(100%+4px)] z-10 overflow-hidden rounded-xl border border-line bg-card py-1"
        >
          {visible.map((s, i) => {
            const { title, year } = parseTitle(s);
            return (
              <li
                key={s}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={i === active}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => add(s)}
                onMouseEnter={() => setActive(i)}
                className={`flex cursor-pointer items-baseline justify-between gap-3 px-4 py-2 text-sm ${
                  i === active ? "bg-raised text-ink" : "text-ink-2"
                }`}
              >
                <span className="truncate">{title}</span>
                {year && (
                  <span className="shrink-0 font-mono text-xs tabular-nums text-ink-muted">
                    {year}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
