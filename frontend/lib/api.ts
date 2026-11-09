export interface Movie {
  title: string;
  poster: string | null;
  release_date: string | null;
  vote_average: number | null;
  tmdb_id: number | null;
}

export interface RecommendResponse {
  matched: string;
  /** Only present when the backend had to correct the query (e.g. a typo). */
  query?: string;
  recommendations: Movie[];
}

/** Why a request failed, so the UI can say something accurate. */
export type ApiErrorKind =
  | "not_found" // 404: the title isn't in the catalog
  | "bad_request" // 400: the backend rejected the input
  | "rate_limited" // 429: too many requests
  | "unavailable" // 503: backend is up but the model isn't loaded
  | "server" // other 5xx
  | "network"; // the backend could not be reached at all

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 404) return "not_found";
  if (status === 429) return "rate_limited";
  if (status === 503) return "unavailable";
  if (status >= 500) return "server";
  return "bad_request";
}

/** Human-readable message for an error from `recommendMovies`. */
export function describeError(err: unknown, title: string): string {
  const kind = err instanceof ApiError ? err.kind : "network";
  switch (kind) {
    case "not_found":
      return `We couldn't find a movie matching “${title}”. Check the spelling or try another title.`;
    case "bad_request":
      return "That search wasn't valid. Try a different title.";
    case "rate_limited":
      return "Too many searches in a short time. Wait a moment and try again.";
    case "unavailable":
      return "The recommendation model isn't available right now. Please try again shortly.";
    case "server":
      return "Something went wrong on the server. Please try again.";
    default:
      return "Can't reach the recommendation server. Is the backend running?";
  }
}

export default async function recommendMovies(title: string): Promise<RecommendResponse> {
  if (!title) return { matched: "", recommendations: [] };

  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  let response: Response;
  try {
    response = await fetch(`${apiBase}/api/v1/recommend/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
  } catch {
    throw new ApiError("network", "Could not reach the backend");
  }

  if (!response.ok) {
    throw new ApiError(
      kindForStatus(response.status),
      `Backend returned ${response.status}`,
      response.status,
    );
  }

  const data: RecommendResponse = await response.json();
  if (!Array.isArray(data.recommendations)) {
    throw new ApiError("server", "Invalid data format from backend");
  }
  return data;
}
