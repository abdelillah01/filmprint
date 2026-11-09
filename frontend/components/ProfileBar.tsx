import { parseTitle } from "../lib/title";

interface ProfileBarProps {
  likes: string[];
  dislikes: string[];
  hiddenCount: number;
  onRemoveLike: (title: string) => void;
  onRemoveDislike: (title: string) => void;
  onShowSeen: () => void;
  onReset: () => void;
}

function Chip({ title, onRemove }: { title: string; onRemove: () => void }) {
  const { title: shown, year } = parseTitle(title);
  return (
    <span className="inline-flex h-8 items-center gap-1 rounded-lg border border-line bg-card pl-3 pr-1 text-sm text-ink">
      <span className="max-w-[16rem] truncate">{shown}</span>
      {year && <span className="font-mono text-xs tabular-nums text-ink-muted">{year}</span>}
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${shown}`}
        className="flex h-6 w-6 items-center justify-center rounded-md text-ink-muted transition-colors duration-150 ease-smooth hover:bg-raised hover:text-ink"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          aria-hidden
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </span>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="mb-2 text-sm text-ink-muted">{label}</p>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

const linkButton =
  "text-sm text-ink-muted underline-offset-4 transition-colors duration-150 ease-smooth hover:text-ink hover:underline";

export default function ProfileBar({
  likes,
  dislikes,
  hiddenCount,
  onRemoveLike,
  onRemoveDislike,
  onShowSeen,
  onReset,
}: ProfileBarProps) {
  const empty = likes.length === 0 && dislikes.length === 0 && hiddenCount === 0;
  return (
    <div className="flex flex-col gap-4">
      {empty && (
        <p className="text-sm text-ink-muted">
          Nothing yet. Add films from the search box or the Discover page.
        </p>
      )}
      {likes.length > 0 && (
        <Row label="Films you like">
          {likes.map((t) => (
            <Chip key={t} title={t} onRemove={() => onRemoveLike(t)} />
          ))}
        </Row>
      )}
      {dislikes.length > 0 && (
        <Row label="Not for me">
          {dislikes.map((t) => (
            <Chip key={t} title={t} onRemove={() => onRemoveDislike(t)} />
          ))}
        </Row>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {hiddenCount > 0 && (
          <span className="text-sm text-ink-muted">
            <span className="font-mono tabular-nums">{hiddenCount}</span> hidden as seen
            <button type="button" onClick={onShowSeen} className={`ml-2 ${linkButton}`}>
              Show again
            </button>
          </span>
        )}
        <button type="button" onClick={onReset} className={linkButton}>
          Start over
        </button>
      </div>
    </div>
  );
}
