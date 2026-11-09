"use client";

import { useState } from "react";

interface SearchBarProps {
  onSearch: (title: string) => void;
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

export default function SearchBar({ onSearch }: SearchBarProps) {
  const [title, setTitle] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;
    onSearch(title.trim());
  };

  return (
    <form
      onSubmit={submit}
      className="flex h-12 max-w-2xl items-center rounded-xl border border-line bg-card transition-colors duration-150 ease-smooth focus-within:border-line-strong"
    >
      <SearchIcon />
      <input
        className="h-full min-w-0 flex-1 bg-transparent px-3 text-[15px] text-ink outline-none placeholder:text-ink-muted"
        placeholder="Search a film"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        aria-label="Movie title"
      />
      {title && (
        <button
          type="button"
          onClick={() => setTitle("")}
          className="mr-1 flex h-8 w-8 items-center justify-center rounded-lg text-ink-muted transition-colors duration-150 ease-smooth hover:text-ink"
          aria-label="Clear search"
        >
          <ClearIcon />
        </button>
      )}
      <button
        type="submit"
        className="m-1.5 h-9 shrink-0 rounded-md bg-accent px-5 text-sm font-medium text-accent-ink transition-[transform,opacity] duration-150 ease-smooth hover:opacity-90 active:scale-[0.98]"
      >
        Find
      </button>
    </form>
  );
}
