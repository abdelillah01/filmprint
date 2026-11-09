export default function LoadingSkeleton() {
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-card" aria-hidden>
      <div className="skeleton aspect-[2/3] w-full" />
      <div className="p-3">
        <div className="skeleton h-4 w-6 rounded" />
        <div className="skeleton mt-1 h-5 w-3/4 rounded" />
        <div className="skeleton mt-1 h-4 w-1/3 rounded" />
        <div className="h-5" />
      </div>
    </div>
  );
}
