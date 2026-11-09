export interface Movie {
  title: string;
  poster: string | null;
  release_date: string | null;
  vote_average: number | null;
  tmdb_id: number | null;
  genres: string[];
  /** e.g. "tt0114709"; used to link to IMDb. */
  imdb_id: string | null;
  /** From IMDb's non-commercial dataset; null when that file isn't installed. */
  imdb_rating: number | null;
  imdb_votes: number | null;
  tmdb_votes: number | null;
  /** Minutes. */
  runtime: number | null;
}

export interface Filters {
  /** Keep films that have at least one of these genres. */
  genres: string[];
  exclude_genres: string[];
  year_min?: number;
  year_max?: number;
  min_rating?: number;
}

export interface RecommendRequest {
  likes: string[];
  dislikes: string[];
  /** Films already seen; never recommended. */
  exclude: string[];
  filters: Filters;
  n?: number;
}

export interface Correction {
  query: string;
  matched: string;
}

export interface RecommendResponse {
  matched: string;
  likes: string[];
  dislikes: string[];
  ignored: string[];
  /** Titles the backend had to fuzzily correct (typos, near-misses). */
  corrections: Correction[];
  /** Details of the matched liked films, for comparing taste against the results. */
  liked: Movie[];
  recommendations: Movie[];
}

export interface CatalogMeta {
  genres: string[];
  year_min: number | null;
  year_max: number | null;
  n_movies: number;
}

/** Why a request failed, so the UI can say something accurate. */
export type ApiErrorKind =
  | "not_found" // 404: a liked title isn't in the catalog
  | "bad_request" // 400: the backend rejected the input
  | "rate_limited" // 429: too many requests
  | "unavailable" // 503: backend is up but the model isn't loaded
  | "server" // other 5xx
  | "network"; // the backend could not be reached at all

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  /** For not_found: the liked titles that could not be matched. */
  readonly unresolved: string[];

  constructor(kind: ApiErrorKind, message: string, status?: number, unresolved: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.unresolved = unresolved;
  }
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

function kindForStatus(status: number): ApiErrorKind {
  if (status === 404) return "not_found";
  if (status === 429) return "rate_limited";
  if (status === 503) return "unavailable";
  if (status >= 500) return "server";
  return "bad_request";
}

/** Human-readable message for an error from the API client. */
export function describeError(err: unknown): string {
  const kind = err instanceof ApiError ? err.kind : "network";
  switch (kind) {
    case "not_found": {
      const titles = err instanceof ApiError ? err.unresolved : [];
      return titles.length
        ? `We couldn't find ${titles.map((t) => `“${t}”`).join(", ")} in the catalog. Check the spelling or pick a suggestion.`
        : "We couldn't find that film. Check the spelling or pick a suggestion.";
    }
    case "bad_request":
      return "That search wasn't valid. Try a different title or loosen the filters.";
    case "rate_limited":
      return "Too many requests in a short time. Wait a moment and try again.";
    case "unavailable":
      return "The recommendation model isn't available right now. Please try again shortly.";
    case "server":
      return "Something went wrong on the server. Please try again.";
    default:
      return "Can't reach the recommendation server. Is the backend running?";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("network", "Could not reach the backend");
  }

  if (!response.ok) {
    let unresolved: string[] = [];
    if (response.status === 404) {
      try {
        const body = await response.json();
        if (Array.isArray(body.unresolved)) unresolved = body.unresolved;
      } catch {
        /* body wasn't JSON; the status is enough */
      }
    }
    throw new ApiError(
      kindForStatus(response.status),
      `Backend returned ${response.status}`,
      response.status,
      unresolved,
    );
  }
  return (await response.json()) as T;
}

/** Drop empty filter fields and fix an inverted year range before sending. */
export function cleanFilters(filters: Filters): Filters {
  const out: Filters = { genres: filters.genres, exclude_genres: filters.exclude_genres };
  let { year_min, year_max } = filters;
  if (year_min != null && year_max != null && year_min > year_max) {
    [year_min, year_max] = [year_max, year_min];
  }
  if (year_min != null) out.year_min = year_min;
  if (year_max != null) out.year_max = year_max;
  if (filters.min_rating != null) out.min_rating = filters.min_rating;
  return out;
}

export async function recommend(
  req: RecommendRequest,
  signal?: AbortSignal,
): Promise<RecommendResponse> {
  const data = await request<RecommendResponse>("/api/v1/recommend/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n: 10, ...req, filters: cleanFilters(req.filters) }),
    signal,
  });
  if (!Array.isArray(data.recommendations)) {
    throw new ApiError("server", "Invalid data format from backend");
  }
  return data;
}

export async function searchTitles(q: string, signal?: AbortSignal): Promise<string[]> {
  const data = await request<{ titles: string[] }>(
    `/api/v1/titles/?q=${encodeURIComponent(q)}`,
    { signal },
  );
  return data.titles ?? [];
}

export async function fetchMeta(): Promise<CatalogMeta> {
  return request<CatalogMeta>("/api/v1/meta/");
}

/** Well-known films with posters, optionally within one genre: the cold-start picker. */
export async function fetchPopular(
  genre?: string,
  n = 18,
  signal?: AbortSignal,
): Promise<{ genre: string | null; movies: Movie[] }> {
  const params = new URLSearchParams({ n: String(n) });
  if (genre) params.set("genre", genre);
  return request(`/api/v1/popular/?${params}`, { signal });
}
