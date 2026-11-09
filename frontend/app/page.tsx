"use client";

import { useState } from "react";
import { motion, type Variants } from "framer-motion";
import SearchBar from "../components/SearchBar";
import MovieCard from "../components/MovieCard";
import LoadingSkeleton from "../components/LoadingSkeleton";
import recommendMovies, { describeError, type Movie } from "../lib/api";
import { parseTitle } from "../lib/title";

const FEATURED = [
  "Inception", "The Dark Knight", "Toy Story", "Interstellar",
  "Pulp Fiction", "The Matrix", "Forrest Gump", "Fight Club",
];

const EASE = [0.22, 1, 0.36, 1] as const;

// One entrance on load: a short fade and rise for the main content.
const pageEnter: Variants = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: EASE } },
};

// Cards share the same fade and rise; only the first six are staggered.
const cardEnter: Variants = {
  hidden: { opacity: 0, y: 8 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: EASE, delay: Math.min(i, 5) * 0.04 },
  }),
};

const GRID = "grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5";

export default function Home() {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [matched, setMatched] = useState<string | null>(null);
  // Set only when the backend corrected the query (typo / near-miss), not on every search.
  const [correctedFrom, setCorrectedFrom] = useState<string | null>(null);

  const handleSearch = async (title: string) => {
    setError("");
    setMovies([]);
    setQuery(title);
    setMatched(null);
    setCorrectedFrom(null);
    setLoading(true);

    try {
      const { matched: m, query: corrected, recommendations } = await recommendMovies(title);
      setMatched(m);
      setCorrectedFrom(corrected ?? null);
      setMovies(recommendations);
    } catch (err) {
      setError(describeError(err, title));
    } finally {
      setLoading(false);
    }
  };

  const parsed = parseTitle(matched || query);
  const shownTitle = parsed.year ? `${parsed.title} (${parsed.year})` : parsed.title;
  const hasResults = !loading && movies.length > 0;
  const isEmpty = !loading && movies.length === 0 && !error && !query;

  return (
    <main className="min-h-screen">
      <motion.div
        initial="hidden"
        animate="show"
        variants={pageEnter}
        className="mx-auto w-full max-w-6xl px-6 py-12 md:px-8"
      >
        <header className="mb-8">
          <div className="flex items-center gap-2">
            <ClapperIcon />
            <h1 className="text-lg font-medium leading-none">Movie Recommender</h1>
          </div>
          <p className="mt-3 text-base text-ink-2">Type a film you like and get ten similar ones</p>
        </header>

        <div className="mb-12">
          <SearchBar onSearch={handleSearch} />
        </div>

        {loading && (
          <div className={GRID}>
            {[...Array(10)].map((_, i) => (
              <LoadingSkeleton key={i} />
            ))}
          </div>
        )}

        {!loading && error && (
          <div
            role="alert"
            className="flex max-w-2xl items-start gap-3 rounded-xl border border-line bg-card p-4 text-sm text-ink-2"
          >
            <span aria-hidden className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-danger" />
            {error}
          </div>
        )}

        {hasResults && (
          <section>
            <div className="mb-6">
              <p className="text-sm text-ink-2">
                Because you watched <span className="font-medium text-ink">{shownTitle}</span>
                <span className="ml-2 font-mono text-xs tabular-nums text-ink-muted">
                  {movies.length} results
                </span>
              </p>
              {correctedFrom && (
                <p className="mt-1 text-sm text-ink-muted">
                  No exact match for &ldquo;{correctedFrom}&rdquo;, showing results for
                  &ldquo;{matched}&rdquo;
                </p>
              )}
            </div>

            <div key={matched ?? query} className={GRID}>
              {movies.map((m, i) => (
                <motion.div
                  key={m.tmdb_id ?? m.title}
                  custom={i}
                  variants={cardEnter}
                  initial="hidden"
                  animate="show"
                >
                  <MovieCard
                    rank={i + 1}
                    title={m.title}
                    poster={m.poster}
                    vote_average={m.vote_average}
                    release_date={m.release_date}
                  />
                </motion.div>
              ))}
            </div>
          </section>
        )}

        {isEmpty && (
          <section>
            <p className="mb-3 text-sm text-ink-muted">Try one of these</p>
            <div className="flex flex-wrap gap-2">
              {FEATURED.map((film) => (
                <button
                  key={film}
                  onClick={() => handleSearch(film)}
                  className="h-8 rounded-lg border border-line bg-card px-3 text-sm text-ink-2 transition-[color,background-color,border-color,transform] duration-150 ease-smooth hover:border-line-strong hover:bg-raised hover:text-ink active:scale-[0.98]"
                >
                  {film}
                </button>
              ))}
            </div>
          </section>
        )}

        {!loading && query && movies.length === 0 && !error && (
          <p className="text-sm text-ink-muted">
            No results for &ldquo;{query}&rdquo;. Try a different title.
          </p>
        )}
      </motion.div>
    </main>
  );
}

function ClapperIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0 text-ink"
      aria-hidden
    >
      <path d="M4 11v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8H4Z" />
      <path d="m4 11-.88-2.87a2 2 0 0 1 1.33-2.5l11.48-3.5a2 2 0 0 1 2.5 1.32l.87 2.87L4 11Z" />
      <path d="m6.6 4.99 3.38 4.2" />
      <path d="m11.86 3.38 3.38 4.2" />
    </svg>
  );
}
