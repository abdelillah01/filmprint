import type { Movie } from "../lib/api";
import { formatRuntime, formatVotes } from "../lib/stats";
import { parseTitle } from "../lib/title";

interface MovieCardProps {
  rank: number;
  movie: Movie;
  onLike?: () => void;
  onDislike?: () => void;
  onSeen?: () => void;
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
  "aria-hidden": true,
};

function ThumbUp() {
  return (
    <svg {...iconProps}>
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
    </svg>
  );
}

function ThumbDown() {
  return (
    <svg {...iconProps}>
      <path d="M17 14V2" />
      <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
    </svg>
  );
}

function Check() {
  return (
    <svg {...iconProps}>
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function FilmIcon() {
  return (
    <svg {...iconProps} width={32} height={32} className="text-ink-muted">
      <rect x="2" y="2" width="20" height="20" rx="2" />
      <line x1="7" y1="2" x2="7" y2="22" />
      <line x1="17" y1="2" x2="17" y2="22" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <line x1="2" y1="7" x2="7" y2="7" />
      <line x1="17" y1="7" x2="22" y2="7" />
      <line x1="17" y1="17" x2="22" y2="17" />
      <line x1="2" y1="17" x2="7" y2="17" />
    </svg>
  );
}

function ActionButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  if (!onClick) return null;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="flex h-7 w-7 items-center justify-center rounded-md text-ink-muted transition-[color,background-color,transform] duration-150 ease-smooth hover:bg-raised hover:text-ink active:scale-[0.97]"
    >
      {children}
    </button>
  );
}

const linkClass =
  "text-ink-muted underline-offset-4 transition-colors duration-150 ease-smooth hover:text-ink hover:underline";

export default function MovieCard({ rank, movie, onLike, onDislike, onSeen }: MovieCardProps) {
  const { title, year: parsedYear } = parseTitle(movie.title);
  const year = parsedYear || (movie.release_date ? movie.release_date.slice(0, 4) : null);
  const runtime = formatRuntime(movie.runtime);
  const votes = formatVotes(movie.imdb_votes);

  return (
    <article className="overflow-hidden rounded-xl border border-line bg-card transition-colors duration-150 ease-smooth hover:border-line-strong">
      {movie.poster ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={movie.poster}
          alt={`Poster for ${title}`}
          loading="lazy"
          className="aspect-[2/3] w-full object-cover"
        />
      ) : (
        <div className="flex aspect-[2/3] w-full items-center justify-center bg-raised">
          <FilmIcon />
        </div>
      )}

      <div className="p-3">
        <div className="flex h-7 items-center justify-between">
          <p className="font-mono text-xs tabular-nums text-ink-muted">
            {String(rank).padStart(2, "0")}
          </p>
          <div className="-mr-1 flex items-center gap-0.5">
            <ActionButton label={`I like ${title}`} onClick={onLike}>
              <ThumbUp />
            </ActionButton>
            <ActionButton label={`Not for me: ${title}`} onClick={onDislike}>
              <ThumbDown />
            </ActionButton>
            <ActionButton label={`Already seen ${title}`} onClick={onSeen}>
              <Check />
            </ActionButton>
          </div>
        </div>

        <h3 className="line-clamp-2 min-h-10 text-sm font-medium leading-5 text-ink">{title}</h3>

        <p className="mt-1 h-4 font-mono text-xs tabular-nums text-ink-2">
          {[year, runtime].filter(Boolean).join(" · ")}
        </p>

        <p
          className="mt-1 h-4 font-mono text-xs tabular-nums text-ink-2"
          title={
            movie.imdb_rating != null && movie.imdb_votes != null
              ? `${movie.imdb_rating.toFixed(1)} on IMDb from ${movie.imdb_votes.toLocaleString("en")} votes`
              : undefined
          }
        >
          {movie.imdb_rating != null ? (
            <>
              <span className="text-ink-muted">IMDb</span> {movie.imdb_rating.toFixed(1)}
              {votes && <span className="text-ink-muted"> ({votes})</span>}
            </>
          ) : (
            <span className="text-ink-muted">No IMDb rating</span>
          )}
        </p>

        <p className="mt-2 flex h-4 gap-3 text-xs">
          {movie.imdb_id && (
            <a
              href={`https://www.imdb.com/title/${movie.imdb_id}/`}
              target="_blank"
              rel="noopener noreferrer"
              className={linkClass}
              aria-label={`${title} on IMDb`}
            >
              IMDb
            </a>
          )}
          {movie.tmdb_id != null && (
            <a
              href={`https://letterboxd.com/tmdb/${movie.tmdb_id}/`}
              target="_blank"
              rel="noopener noreferrer"
              className={linkClass}
              aria-label={`${title} on Letterboxd`}
            >
              Letterboxd
            </a>
          )}
        </p>
      </div>
    </article>
  );
}
