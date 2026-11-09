"use client";

import { useEffect, useId, useRef, useState } from "react";
import Image from "next/image";
import SearchBar from "./SearchBar";

export type View = "discover" | "for-you";

interface NavbarProps {
  view: View;
  /** "For you" only makes sense once there is a profile. */
  hasProfile: boolean;
  /** Number of liked films, shown next to "Your taste". */
  tasteCount: number;
  onNavigate: (view: View) => void;
  onHome: () => void;
  onAdd: (title: string) => void;
  /** Contents of the "Your taste" menu. */
  taste: React.ReactNode;
}

const REPO_URL = "https://github.com/abdelillah01/filmprint";

const iconProps = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

function Logo() {
  return <Image src="/logo.png" alt="" width={28} height={28} className="shrink-0" priority />;
}

function HeartIcon() {
  return (
    <svg {...iconProps}>
      <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg {...iconProps}>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

const item =
  "inline-flex h-9 items-center gap-2 rounded-lg px-3 text-sm transition-[color,background-color] duration-150 ease-smooth";

function NavButton({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-current={active ? "page" : undefined}
      className={`${item} disabled:cursor-not-allowed disabled:opacity-40 ${
        active ? "bg-raised text-ink" : "text-ink-2 hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

export default function Navbar({
  view,
  hasProfile,
  tasteCount,
  onNavigate,
  onHome,
  onAdd,
  taste,
}: NavbarProps) {
  const [open, setOpen] = useState(false);
  const menuId = useId();
  const menuRef = useRef<HTMLDivElement>(null);

  // Close the menu on an outside click or Escape.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-3 px-6 py-3 md:px-8">
        <h1 className="order-1 text-base font-medium leading-none">
          <button
            type="button"
            onClick={onHome}
            className="flex items-center gap-2 rounded-md py-1 text-ink"
            aria-label="Filmprint, home"
          >
            <Logo />
            <span className="hidden sm:inline">Filmprint</span>
          </button>
        </h1>

        <nav aria-label="Main" className="order-2 flex items-center gap-1 md:order-3">
          <NavButton active={view === "discover"} onClick={() => onNavigate("discover")}>
            Discover
          </NavButton>
          <NavButton
            active={view === "for-you"}
            disabled={!hasProfile}
            onClick={() => onNavigate("for-you")}
          >
            For you
          </NavButton>

          <div ref={menuRef} className="sm:relative">
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              aria-controls={menuId}
              aria-label={
                tasteCount > 0 ? `Your taste, ${tasteCount} liked` : "Your taste"
              }
              className={`${item} ${open ? "bg-raised text-ink" : "text-ink-2 hover:text-ink"}`}
            >
              <HeartIcon />
              <span className="hidden sm:inline">Your taste</span>
              {tasteCount > 0 && (
                <span className="font-mono text-xs tabular-nums text-accent">{tasteCount}</span>
              )}
            </button>
            {open && (
              <div
                id={menuId}
                role="region"
                aria-label="Your taste"
                // On a phone the panel spans the screen under the navbar (the header is the positioned
                // ancestor); from `sm` it hangs under the button.
                className="absolute inset-x-3 top-full z-40 mt-2 rounded-xl border border-line-strong bg-card p-4 sm:inset-x-auto sm:right-0 sm:w-[22rem]"
              >
                {taste}
              </div>
            )}
          </div>

          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Source code on GitHub"
            title="Source code on GitHub"
            className={`${item} px-2 text-ink-2 hover:text-ink`}
          >
            <GithubIcon />
          </a>
        </nav>

        <div className="order-3 w-full md:order-2 md:w-auto md:flex-1">
          <div className="md:mx-auto md:max-w-xl">
            <SearchBar onAdd={onAdd} compact />
          </div>
        </div>
      </div>
    </header>
  );
}
