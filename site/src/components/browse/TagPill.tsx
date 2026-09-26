import { Link } from 'react-router'
import { cn } from '@/lib/utils'

/** The one way a tag is shown anywhere on the site: a small outlined pill linking to the tag's page. */
export function TagPill({ tag, count, className }: { tag: string; count?: number; className?: string }) {
  return (
    <Link
      to={`/tags/${tag}`}
      className={cn(
        'inline-flex h-6 items-center gap-1 rounded-full border px-2.5 font-sans text-xs text-foreground/80 transition-colors hover:border-foreground/30 hover:bg-muted hover:text-foreground',
        className,
      )}
    >
      <span className="text-muted-foreground">#</span>
      {tag}
      {count !== undefined && <span className="text-muted-foreground tabular-nums">{count}</span>}
    </Link>
  )
}
