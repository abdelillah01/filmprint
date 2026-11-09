import type { Movie } from "./api";

export interface Summary {
  count: number;
  imdbAvg: number | null;
  imdbCount: number;
  yearMedian: number | null;
  yearMin: number | null;
  yearMax: number | null;
  runtimeAvg: number | null;
  runtimeTotal: number;
  topGenres: { genre: string; count: number }[];
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : Math.round((s[mid - 1] + s[mid]) / 2);
}

const present = (xs: (number | null | undefined)[]): number[] =>
  xs.filter((x): x is number => typeof x === "number" && Number.isFinite(x));

/** Aggregate facts about a list of films; every field is null/0 when the data is missing. */
export function summarize(movies: Movie[], genreLimit = 3): Summary {
  const imdb = present(movies.map((m) => m.imdb_rating));
  const years = present(movies.map((m) => parseInt((m.release_date ?? "").slice(0, 4), 10)));
  const runtimes = present(movies.map((m) => m.runtime));

  const genreCounts = new Map<string, number>();
  for (const m of movies) {
    for (const g of m.genres ?? []) genreCounts.set(g, (genreCounts.get(g) ?? 0) + 1);
  }
  const topGenres = [...genreCounts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, genreLimit)
    .map(([genre, count]) => ({ genre, count }));

  return {
    count: movies.length,
    imdbAvg: mean(imdb),
    imdbCount: imdb.length,
    yearMedian: median(years),
    yearMin: years.length ? Math.min(...years) : null,
    yearMax: years.length ? Math.max(...years) : null,
    runtimeAvg: mean(runtimes),
    runtimeTotal: runtimes.reduce((a, b) => a + b, 0),
    topGenres,
  };
}

/** 112 -> "1h 52m", 45 -> "45m" */
export function formatRuntime(minutes: number | null | undefined): string | null {
  if (minutes == null || !Number.isFinite(minutes) || minutes <= 0) return null;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  if (h === 0) return `${m}m`;
  return m === 0 ? `${h}h` : `${h}h ${m}m`;
}

/** 1234 -> "1.2k", 1500000 -> "1.5M" */
export function formatVotes(n: number | null | undefined): string | null {
  if (n == null || !Number.isFinite(n)) return null;
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}
