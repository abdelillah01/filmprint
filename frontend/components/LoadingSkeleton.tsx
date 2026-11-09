export default function LoadingSkeleton() {
  // Mirrors MovieCard: rank row 28, title 40, then year, rating and links lines of 16px.
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-card" aria-hidden>
      <div className="skeleton aspect-[2/3] w-full" />
      <div className="p-3">
        <div className="flex h-7 items-center">
          <div className="skeleton h-4 w-6 rounded" />
        </div>
        <div className="skeleton h-5 w-3/4 rounded" />
        <div className="h-5" />
        <div className="skeleton mt-1 h-4 w-1/3 rounded" />
        <div className="skeleton mt-1 h-4 w-2/3 rounded" />
        <div className="skeleton mt-2 h-4 w-1/2 rounded" />
      </div>
    </div>
  );
}
