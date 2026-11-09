import type { Movie } from "../lib/api";
import { formatRuntime, summarize, type Summary } from "../lib/stats";
import { prettyGenre } from "../lib/title";

interface StatsPanelProps {
  /** The films the user likes (their taste). */
  liked: Movie[];
  /** The films currently recommended. */
  picks: Movie[];
}

const dash = <span className="text-ink-muted">&mdash;</span>;

function genreLine(s: Summary) {
  if (!s.topGenres.length) return dash;
  return s.topGenres.map((g) => `${prettyGenre(g.genre)} ${g.count}`).join(" · ");
}

/** Always-visible comparison of the user's films with the current picks. */
export default function StatsPanel({ liked, picks }: StatsPanelProps) {
  const a = summarize(liked);
  const b = summarize(picks);
  const hasImdb = a.imdbCount > 0 || b.imdbCount > 0;
  const partialImdb = hasImdb && (a.imdbCount < a.count || b.imdbCount < b.count);

  const rows: { label: string; you: React.ReactNode; picks: React.ReactNode }[] = [
    { label: "Films", you: a.count, picks: b.count },
    ...(hasImdb
      ? [
          {
            label: "IMDb average",
            you: a.imdbAvg == null ? dash : a.imdbAvg.toFixed(1),
            picks: b.imdbAvg == null ? dash : b.imdbAvg.toFixed(1),
          },
        ]
      : []),
    { label: "Median year", you: a.yearMedian ?? dash, picks: b.yearMedian ?? dash },
    {
      label: "Avg runtime",
      you: formatRuntime(a.runtimeAvg) ?? dash,
      picks: formatRuntime(b.runtimeAvg) ?? dash,
    },
    {
      label: "Watch time",
      you: formatRuntime(a.runtimeTotal) ?? dash,
      picks: formatRuntime(b.runtimeTotal) ?? dash,
    },
  ];

  return (
    <section aria-label="Stats" className="rounded-xl border border-line bg-card">
      <h2 className="border-b border-line px-4 py-3 text-sm font-medium text-ink">Stats</h2>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-ink-muted">
            <th scope="col" className="px-4 pb-1 pt-3 text-left font-normal">
              <span className="sr-only">Measure</span>
            </th>
            <th scope="col" className="px-2 pb-1 pt-3 text-right font-normal">
              Yours
            </th>
            <th scope="col" className="px-4 pb-1 pt-3 text-right font-normal">
              Picks
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <th scope="row" className="whitespace-nowrap px-4 py-2 text-left font-normal text-ink-2">
                {r.label}
              </th>
              <td className="px-2 py-2 text-right font-mono tabular-nums text-ink">{r.you}</td>
              <td className="px-4 py-2 text-right font-mono tabular-nums text-ink">{r.picks}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <dl className="space-y-3 border-t border-line px-4 py-3 text-sm">
        <div>
          <dt className="text-ink-muted">Your top genres</dt>
          <dd className="mt-1 text-ink">{genreLine(a)}</dd>
        </div>
        <div>
          <dt className="text-ink-muted">Picks top genres</dt>
          <dd className="mt-1 text-ink">{genreLine(b)}</dd>
        </div>
      </dl>

      {partialImdb && (
        <p className="border-t border-line px-4 py-3 text-xs text-ink-muted">
          IMDb average uses{" "}
          <span className="font-mono tabular-nums">
            {a.imdbCount}/{a.count}
          </span>{" "}
          of your films and{" "}
          <span className="font-mono tabular-nums">
            {b.imdbCount}/{b.count}
          </span>{" "}
          of the picks.
        </p>
      )}
    </section>
  );
}
