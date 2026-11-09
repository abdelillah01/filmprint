import { parseTitle } from "../lib/title";

interface MovieCardProps {
  rank: number;
  title: string;
  poster: string | null;
  vote_average?: number | null;
  release_date?: string | null;
}

function FilmIcon() {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-ink-muted"
      aria-hidden
    >
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

export default function MovieCard({
  rank,
  title: rawTitle,
  poster,
  vote_average,
  release_date,
}: MovieCardProps) {
  const { title, year: parsedYear } = parseTitle(rawTitle);
  const year = parsedYear || (release_date ? release_date.slice(0, 4) : null);
  const rating = vote_average != null ? Number(vote_average).toFixed(1) : null;

  return (
    <article className="overflow-hidden rounded-xl border border-line bg-card transition-colors duration-150 ease-smooth hover:border-line-strong">
      {poster ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={poster}
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
        <p className="font-mono text-xs tabular-nums text-ink-muted">
          {String(rank).padStart(2, "0")}
        </p>
        <h3 className="mt-1 line-clamp-2 min-h-10 text-sm font-medium leading-5 text-ink">
          {title}
        </h3>
        <p className="mt-1 font-mono text-xs tabular-nums text-ink-2">
          {[year, rating].filter(Boolean).join(" · ")}
        </p>
      </div>
    </article>
  );
}
