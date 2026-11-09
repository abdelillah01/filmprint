import type { Filters } from "./api";

/** The user's taste, kept in the browser only. Nothing here is sent anywhere except as the
 * body of a recommend request. */
export interface Profile {
  likes: string[];
  dislikes: string[];
  /** Films marked "seen": hidden from results. */
  exclude: string[];
  filters: Filters;
}

export const LIMITS = { likes: 10, dislikes: 10, exclude: 200 } as const;

export const EMPTY_FILTERS: Filters = { genres: [], exclude_genres: [] };
export const EMPTY_PROFILE: Profile = {
  likes: [],
  dislikes: [],
  exclude: [],
  filters: EMPTY_FILTERS,
};

const STORAGE_KEY = "movie-recommender.profile.v1";

const same = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();
const without = (list: string[], title: string) => list.filter((t) => !same(t, title));
const push = (list: string[], title: string, max: number) =>
  [...without(list, title), title].slice(-max);

/** Liking a film also removes it from dislikes and from the hidden list. */
export const addLike = (p: Profile, title: string): Profile => ({
  ...p,
  likes: push(p.likes, title, LIMITS.likes),
  dislikes: without(p.dislikes, title),
  exclude: without(p.exclude, title),
});

export const addDislike = (p: Profile, title: string): Profile => ({
  ...p,
  likes: without(p.likes, title),
  dislikes: push(p.dislikes, title, LIMITS.dislikes),
});

export const markSeen = (p: Profile, title: string): Profile => ({
  ...p,
  exclude: push(p.exclude, title, LIMITS.exclude),
});

export const removeLike = (p: Profile, title: string): Profile => ({
  ...p,
  likes: without(p.likes, title),
});

export const removeDislike = (p: Profile, title: string): Profile => ({
  ...p,
  dislikes: without(p.dislikes, title),
});

export const clearSeen = (p: Profile): Profile => ({ ...p, exclude: [] });

export function activeFilterCount(f: Filters): number {
  return (
    f.genres.length +
    f.exclude_genres.length +
    (f.year_min != null ? 1 : 0) +
    (f.year_max != null ? 1 : 0) +
    (f.min_rating != null ? 1 : 0)
  );
}

const isStrings = (v: unknown): v is string[] =>
  Array.isArray(v) && v.every((x) => typeof x === "string");

/** Read the saved profile; any problem (private mode, bad JSON) just means "start empty". */
export function loadProfile(): Profile {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY_PROFILE;
    const data = JSON.parse(raw);
    const f = data.filters ?? {};
    return {
      likes: isStrings(data.likes) ? data.likes.slice(0, LIMITS.likes) : [],
      dislikes: isStrings(data.dislikes) ? data.dislikes.slice(0, LIMITS.dislikes) : [],
      exclude: isStrings(data.exclude) ? data.exclude.slice(0, LIMITS.exclude) : [],
      filters: {
        genres: isStrings(f.genres) ? f.genres : [],
        exclude_genres: isStrings(f.exclude_genres) ? f.exclude_genres : [],
        year_min: typeof f.year_min === "number" ? f.year_min : undefined,
        year_max: typeof f.year_max === "number" ? f.year_max : undefined,
        min_rating: typeof f.min_rating === "number" ? f.min_rating : undefined,
      },
    };
  } catch {
    return EMPTY_PROFILE;
  }
}

export function saveProfile(p: Profile): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable: the profile just won't persist */
  }
}

export function sameTitles(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((t, i) => t === b[i]);
}

// ---------------------------------------------------------------------------
// A tiny external store so React reads the saved profile via useSyncExternalStore:
// the server render and first client render use the empty profile (no hydration
// mismatch), then the real one is picked up. Every change is persisted immediately.
// ---------------------------------------------------------------------------

let current: Profile = EMPTY_PROFILE;
let loaded = false;
const listeners = new Set<() => void>();

export const profileStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
  getSnapshot(): Profile {
    if (!loaded && typeof window !== "undefined") {
      current = loadProfile();
      loaded = true;
    }
    return current;
  },
  getServerSnapshot(): Profile {
    return EMPTY_PROFILE;
  },
  set(next: Profile): void {
    current = next;
    loaded = true;
    saveProfile(next);
    listeners.forEach((l) => l());
  },
};
