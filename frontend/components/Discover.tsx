"use client";

import { useEffect, useState } from "react";
import { fetchPopular, type Movie } from "../lib/api";
import { LIMITS } from "../lib/profile";
import { parseTitle, prettyGenre } from "../lib/title";

interface DiscoverProps {
  /** Genres offered as quick filters for the picker. */
  genres: string[];
  /** Titles currently in the user's likes; their tiles show as selected. */
  liked: string[];
  /** Add the film to the likes, or remove it if it is already there. */
  onToggle: (title: string) => void;
  /** Remove every liked film. */
  onClear: () => void;
  /** Go to the recommendations. */
  onSeeResults: () => void;
}

const COUNT = 18;

/** A slow, decorative strip of posters. Not interactive; hidden from assistive tech. */
function PosterStrip({ movies }: { movies: Movie[] }) {
  const posters = movies.filter((m) => m.poster);
  if (posters.length < 4) return <div className="h-40" aria-hidden />;
  return (
    <div
      aria-hidden
      className="overflow-hidden [mask-image:linear-gradient(to_right,transparent,#000_8%,#000_92%,transparent)]"
    >
      <div className="marquee-track">
        {[...posters, ...posters].map((m, i) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={i}
            src={m.poster ?? ""}
            alt=""
            className="mr-3 aspect-[2/3] h-40 shrink-0 rounded-lg border border-line object-cover"
          />
        ))}
      </div>
    </div>
  );
}

function Tile({
  movie,
  selected,
  disabled,
  onToggle,
}: {
  movie: Movie;
  selected: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  const { title, year } = parseTitle(movie.title);
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={selected}
      disabled={disabled && !selected}
      className="group text-left disabled:cursor-not-allowed disabled:opacity-40"
    >
      <span
        className={`relative block overflow-hidden rounded-xl border transition-colors duration-150 ease-smooth ${
          selected ? "border-accent" : "border-line group-hover:border-line-strong"
        }`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={movie.poster ?? ""}
          alt={`Poster for ${title}`}
          loading="lazy"
          className="aspect-[2/3] w-full object-cover"
        />
        {selected && (
          <span className="absolute bottom-2 right-2 flex h-6 w-6 items-center justify-center rounded-md bg-accent text-accent-ink">
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M20 6 9 17l-5-5" />
            </svg>
          </span>
        )}
      </span>
      <span className="mt-2 block truncate text-sm text-ink">{title}</span>
      <span className="block font-mono text-xs tabular-nums text-ink-muted">{year}</span>
    </button>
  );
}

/** The Discover page: browse well-known films by genre and pick the ones you love. */
export default function Discover({
  genres,
  liked,
  onToggle,
  onClear,
  onSeeResults,
}: DiscoverProps) {
  const [genre, setGenre] = useState<string | null>(null);
  const [byGenre, setByGenre] = useState<Record<string, Movie[]>>({});
  const [failed, setFailed] = useState(false);

  const key = genre ?? "";
  // Results are cached per genre; the effect only runs a request for a genre we have not loaded.
  useEffect(() => {
    if (byGenre[key]) return;
    let cancelled = false;
    fetchPopular(genre ?? undefined, COUNT)
      .then((r) => {
        if (!cancelled) setByGenre((m) => ({ ...m, [key]: r.movies }));
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [key, genre, byGenre]);

  const movies = byGenre[key];
  const likedLower = liked.map((t) => t.toLowerCase());
  const atLimit = liked.length >= LIMITS.likes;

  return (
    <section aria-label="Discover">
      <div className="mb-8">
        <PosterStrip movies={byGenre[""] ?? []} />
      </div>

      <h2 className="text-2xl font-medium leading-8 text-ink">Pick a few films you love</h2>
      <p className="mt-2 max-w-xl text-base text-ink-2">
        Choose as many as you like and we will find ten more in the same spirit. You can also
        search for a film in the bar above.
      </p>

      <div className="mt-6 flex flex-wrap gap-2" role="group" aria-label="Genre">
        {[null, ...genres].map((g) => {
          const active = g === genre;
          return (
            <button
              key={g ?? "all"}
              type="button"
              onClick={() => setGenre(g)}
              aria-pressed={active}
              className={`h-8 rounded-lg border px-3 text-sm transition-[color,background-color,border-color] duration-150 ease-smooth ${
                active
                  ? "border-accent bg-raised text-ink"
                  : "border-line bg-card text-ink-2 hover:border-line-strong hover:text-ink"
              }`}
            >
              {g ? prettyGenre(g) : "Popular"}
            </button>
          );
        })}
      </div>

      <div className="mt-6">
        {failed && !movies ? (
          <p className="text-sm text-ink-muted">
            Can&apos;t load suggestions right now. Search for a film in the bar above to start.
          </p>
        ) : !movies ? (
          <div className="grid grid-cols-3 gap-4 sm:grid-cols-4 md:grid-cols-6" aria-hidden>
            {[...Array(12)].map((_, i) => (
              <div key={i} className="skeleton aspect-[2/3] rounded-xl" />
            ))}
          </div>
        ) : movies.length === 0 ? (
          <p className="text-sm text-ink-muted">No films to suggest for this genre.</p>
        ) : (
          <div className="grid grid-cols-3 gap-x-4 gap-y-5 sm:grid-cols-4 md:grid-cols-6">
            {movies.map((m) => (
              <Tile
                key={m.tmdb_id ?? m.title}
                movie={m}
                selected={likedLower.includes(m.title.toLowerCase())}
                disabled={atLimit}
                onToggle={() => onToggle(m.title)}
              />
            ))}
          </div>
        )}
      </div>

      {liked.length > 0 && (
        <div className="sticky bottom-4 z-10 mt-8 flex max-w-2xl items-center justify-between gap-4 rounded-xl border border-line-strong bg-card px-4 py-3">
          <p className="text-sm text-ink-2">
            <span className="font-mono tabular-nums text-ink">{liked.length}</span> selected
            {atLimit && <span className="text-ink-muted"> (the most we can use)</span>}
          </p>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onClear}
              className="text-sm text-ink-muted underline-offset-4 transition-colors duration-150 ease-smooth hover:text-ink hover:underline"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={onSeeResults}
              className="h-9 rounded-md bg-accent px-5 text-sm font-medium text-accent-ink transition-[transform,opacity] duration-150 ease-smooth hover:opacity-90 active:scale-[0.98]"
            >
              Get recommendations
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
