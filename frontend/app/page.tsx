"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { motion, type Variants } from "framer-motion";
import Navbar, { type View } from "../components/Navbar";
import MovieCard from "../components/MovieCard";
import LoadingSkeleton from "../components/LoadingSkeleton";
import ProfileBar from "../components/ProfileBar";
import Filters from "../components/Filters";
import StatsPanel from "../components/StatsPanel";
import Discover from "../components/Discover";
import {
  ApiError,
  describeError,
  fetchMeta,
  recommend,
  type CatalogMeta,
  type RecommendResponse,
} from "../lib/api";
import {
  EMPTY_FILTERS,
  EMPTY_PROFILE,
  activeFilterCount,
  addDislike,
  addLike,
  clearSeen,
  markSeen,
  profileStore,
  removeDislike,
  removeLike,
  sameTitles,
  type Profile,
} from "../lib/profile";
import { parseTitle } from "../lib/title";

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

// Narrower than before: the stats sidebar takes the right-hand column from `lg` up.
const GRID = "grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4";

const noopSubscribe = () => () => {};

/** "A", "A and B", "A, B and C" */
function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export default function Home() {
  // The taste profile lives in the browser (localStorage) and is read through an external
  // store, so server render and first paint use the empty profile and nothing mismatches.
  const profile = useSyncExternalStore(
    profileStore.subscribe,
    profileStore.getSnapshot,
    profileStore.getServerSnapshot,
  );
  const hydrated = useSyncExternalStore(noopSubscribe, () => true, () => false);

  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [meta, setMeta] = useState<CatalogMeta | null>(null);
  // Which page the user chose in the navbar. null = automatic: Discover until there is a
  // profile, then recommendations.
  const [view, setView] = useState<View | null>(null);
  // Set when we replace the profile with the backend's canonical titles, so that update
  // does not trigger a second identical request.
  const skipNext = useRef(false);

  /** Every profile change goes through here so an empty profile also clears the results. */
  const update = (fn: (p: Profile) => Profile) => {
    const next = fn(profileStore.getSnapshot());
    profileStore.set(next);
    if (next.likes.length === 0) {
      setResult(null);
      setError("");
      setBusy(false);
    }
  };

  useEffect(() => {
    fetchMeta()
      .then(setMeta)
      .catch(() => setMeta(null));
  }, []);

  // Re-query whenever the profile changes; cancel anything still in flight.
  useEffect(() => {
    if (skipNext.current) {
      skipNext.current = false;
      return;
    }
    if (profile.likes.length === 0) return;

    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setBusy(true);
      try {
        const res = await recommend(profile, ctrl.signal);
        if (ctrl.signal.aborted) return;
        setResult(res);
        setError("");
        setBusy(false);
        // Adopt the catalog's spelling of each title ("The Matrix" -> "Matrix, The (1999)").
        if (!sameTitles(res.likes, profile.likes) || !sameTitles(res.dislikes, profile.dislikes)) {
          skipNext.current = true;
          profileStore.set({ ...profileStore.getSnapshot(), likes: res.likes, dislikes: res.dislikes });
        }
      } catch (err) {
        if (ctrl.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) {
          return;
        }
        setBusy(false);
        if (err instanceof ApiError && err.kind === "not_found" && err.unresolved.length) {
          // Drop the films we could not find so the rest of the profile still works.
          setNotice(describeError(err));
          const missing = err.unresolved.map((t) => t.toLowerCase());
          update((p) => ({ ...p, likes: p.likes.filter((t) => !missing.includes(t.toLowerCase())) }));
        } else {
          setError(describeError(err));
        }
      }
    }, 250);

    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [profile]);

  const add = (title: string) => {
    setNotice("");
    setView("for-you");
    update((p) => addLike(p, title));
  };

  // Picking on the Discover page keeps you there (automatic mode would jump to the results).
  const toggleLike = (title: string) => {
    setView("discover");
    update((p) =>
      p.likes.some((t) => t.toLowerCase() === title.toLowerCase())
        ? removeLike(p, title)
        : addLike(p, title),
    );
  };

  const goHome = () => {
    setView(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const hasProfile = profile.likes.length > 0;
  const hasResults = !!result && result.recommendations.length > 0;
  const showSkeleton = busy && !result;
  const filtersActive = activeFilterCount(profile.filters) > 0;
  const activeView: View = hasProfile ? (view ?? "for-you") : "discover";

  return (
    <>
      <Navbar
        view={activeView}
        hasProfile={hasProfile}
        tasteCount={profile.likes.length}
        onNavigate={setView}
        onHome={goHome}
        onAdd={add}
        taste={
          <ProfileBar
            likes={profile.likes}
            dislikes={profile.dislikes}
            hiddenCount={profile.exclude.length}
            onRemoveLike={(t) => update((p) => removeLike(p, t))}
            onRemoveDislike={(t) => update((p) => removeDislike(p, t))}
            onShowSeen={() => update(clearSeen)}
            onReset={() => {
              setNotice("");
              setView(null);
              update(() => EMPTY_PROFILE);
            }}
          />
        }
      />

      <main className="min-h-screen">
        <motion.div
          initial="hidden"
          animate="show"
          variants={pageEnter}
          className="mx-auto w-full max-w-7xl px-6 pb-12 pt-8 md:px-8"
        >
          {notice && (
            <div
              role="status"
              className="mb-6 flex max-w-2xl items-start justify-between gap-4 rounded-xl border border-line bg-card p-4 text-sm text-ink-2"
            >
              <span>{notice}</span>
              <button
                type="button"
                onClick={() => setNotice("")}
                className="shrink-0 text-ink-muted transition-colors duration-150 ease-smooth hover:text-ink"
              >
                Dismiss
              </button>
            </div>
          )}

          {activeView === "discover" ? (
            hydrated && (
              <Discover
                genres={meta?.genres ?? []}
                liked={profile.likes}
                onToggle={toggleLike}
                onClear={() => {
                  setView("discover");
                  update((p) => ({ ...p, likes: [] }));
                }}
                onSeeResults={() => setView("for-you")}
              />
            )
          ) : (
            <>
              {meta && (
                <div className="mb-8">
                  <Filters
                    meta={meta}
                    value={profile.filters}
                    onChange={(filters) => update((p) => ({ ...p, filters }))}
                  />
                </div>
              )}

              {showSkeleton && (
                <ResultsLayout aside={<div className="skeleton h-80 rounded-xl" aria-hidden />}>
                  <div className={GRID}>
                    {[...Array(10)].map((_, i) => (
                      <LoadingSkeleton key={i} />
                    ))}
                  </div>
                </ResultsLayout>
              )}

              {!busy && error && (
                <div
                  role="alert"
                  className="flex max-w-2xl items-start gap-3 rounded-xl border border-line bg-card p-4 text-sm text-ink-2"
                >
                  <span aria-hidden className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-danger" />
                  {error}
                </div>
              )}

              {result && !error && (
                <ResultsLayout
                  aside={
                    hasResults ? (
                      <StatsPanel liked={result.liked} picks={result.recommendations} />
                    ) : null
                  }
                >
                  <div className="mb-6">
                    <h2 className="text-sm text-ink-2">
                      Because you like{" "}
                      <span className="font-medium text-ink">
                        {joinNames(result.likes.map((t) => parseTitle(t).title))}
                      </span>
                      <span className="ml-2 font-mono text-xs tabular-nums text-ink-muted">
                        {result.recommendations.length} results
                      </span>
                    </h2>
                    {result.corrections.map((c) => (
                      <p key={c.query} className="mt-1 text-sm text-ink-muted">
                        No exact match for &ldquo;{c.query}&rdquo;, using &ldquo;
                        {parseTitle(c.matched).title}&rdquo; instead
                      </p>
                    ))}
                  </div>

                  {hasResults ? (
                    <div
                      className={`${GRID} transition-opacity duration-150 ease-smooth ${
                        busy ? "opacity-60" : ""
                      }`}
                    >
                      {result.recommendations.map((m, i) => (
                        <motion.div
                          key={m.tmdb_id ?? m.title}
                          custom={i}
                          variants={cardEnter}
                          initial="hidden"
                          animate="show"
                        >
                          <MovieCard
                            rank={i + 1}
                            movie={m}
                            onLike={() => add(m.title)}
                            onDislike={() => update((p) => addDislike(p, m.title))}
                            onSeen={() => update((p) => markSeen(p, m.title))}
                          />
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="max-w-2xl text-sm text-ink-muted">
                      {filtersActive ? (
                        <>
                          No films match these filters.{" "}
                          <button
                            type="button"
                            onClick={() => update((p) => ({ ...p, filters: EMPTY_FILTERS }))}
                            className="text-ink-2 underline underline-offset-4 hover:text-ink"
                          >
                            Clear filters
                          </button>
                        </>
                      ) : (
                        "Nothing left to recommend. Remove a film from your lists or show the hidden ones."
                      )}
                    </div>
                  )}
                </ResultsLayout>
              )}
            </>
          )}

          {hasProfile && (
            <footer className="mt-16 max-w-2xl text-xs leading-5 text-ink-muted">
              Ratings from IMDb: a snapshot of IMDb&apos;s non-commercial dataset, information
              courtesy of IMDb (imdb.com). Letterboxd links open the film&apos;s page there. Posters
              and film data from TMDB; this product uses the TMDB API but is not endorsed or
              certified by TMDB.
            </footer>
          )}
        </motion.div>
      </main>
    </>
  );
}

/** Results on the left; the stats sidebar on the right from `lg`, below the posters on small screens. */
function ResultsLayout({
  aside,
  children,
}: {
  aside: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-start lg:gap-8 xl:grid-cols-[minmax(0,1fr)_20rem]">
      <div>{children}</div>
      {aside && <aside className="mt-8 lg:sticky lg:top-24 lg:mt-0">{aside}</aside>}
    </div>
  );
}
