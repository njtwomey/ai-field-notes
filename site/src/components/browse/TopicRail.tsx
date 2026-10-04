import { Layers } from 'lucide-react'
import { Link } from 'react-router'
import { TopicIcon } from '@/components/layout/category-icon'
import { browseUrl, notes, taxonomy, type BrowseParams, type CategoryNode } from '@/lib/content'
import { cn } from '@/lib/utils'
import { inCategory } from './filters'

/**
 * Topics with note counts, organized by the master 8-Part hierarchy.
 * Top-level pillars (subjects) sit directly under each Part.
 * Expanding a subject reveals its child categories (topics).
 */
export function TopicRail({ params, onNavigate }: { params: BrowseParams; onNavigate?: () => void }) {
  const count = (path: string) => notes.filter((n) => inCategory(n, path)).length
  const item = (active: boolean) =>
    cn(
      'flex items-center gap-2 rounded-md px-2 py-1 text-xs transition-colors hover:bg-muted',
      active ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground',
    )

  return (
    <nav aria-label="Topics" className="space-y-4">
      <Link
        to={browseUrl({ ...params, c: undefined, part: undefined })}
        onClick={onNavigate}
        className={cn(item(!params.c && !params.part), 'py-1.5 text-sm')}
      >
        <Layers className="size-4 shrink-0" aria-hidden />
        <span className="flex-1 font-medium">All notes</span>
        <span className="text-xs text-muted-foreground tabular-nums">{notes.length}</span>
      </Link>

      <div className="space-y-4 pt-1">
        {taxonomy.map((part, idx) => {
          const isPartActive = params.part === part.path && !params.c
          const partNotesCount = count(part.path)

          return (
            <section key={part.path} className="space-y-1">
              <Link
                to={browseUrl({ ...params, part: part.path, c: undefined })}
                onClick={onNavigate}
                className={cn(
                  'group flex items-center justify-between gap-1 rounded-md px-2 py-1 text-[11px] font-semibold tracking-wider uppercase transition-colors select-none',
                  isPartActive
                    ? 'bg-muted text-foreground'
                    : 'text-muted-foreground/80 hover:bg-muted/60 hover:text-foreground',
                )}
                title={`Filter by ${part.roman ?? `Part ${idx + 1}`}: ${part.title}`}
              >
                <span className="truncate">
                  {part.roman ? `${part.roman} · ` : ''}
                  {part.title}
                </span>
                <span className="font-mono text-[10px] text-muted-foreground/70 tabular-nums group-hover:text-foreground">
                  {partNotesCount}
                </span>
              </Link>

              {/* Top-level pillars (Subjects) directly under the Part */}
              <div className="ml-2 space-y-0.5 border-l border-border/40 pl-1.5">
                {part.children.map((subject) => {
                  const open = params.c === subject.path || params.c?.startsWith(`${subject.path}/`)
                  const n = count(subject.path)
                  return (
                    <div key={subject.path}>
                      <Link
                        to={browseUrl({ ...params, c: subject.path, part: undefined })}
                        onClick={onNavigate}
                        className={cn(item(params.c === subject.path), n === 0 && 'opacity-40')}
                      >
                        <TopicIcon icon={subject.icon} className="size-3.5 shrink-0" aria-hidden />
                        <span className="flex-1 truncate">{subject.title}</span>
                        <span className="text-[10px] tabular-nums">{n}</span>
                      </Link>
                      {open && subject.children.length > 0 && (
                        <Subtopics
                          nodes={subject.children}
                          params={params}
                          onNavigate={onNavigate}
                          count={count}
                          item={item}
                        />
                      )}
                    </div>
                  )
                })}
              </div>
            </section>
          )
        })}
      </div>
    </nav>
  )
}

/** Subtopics at any depth, shown once their parent subject or topic is open. */
function Subtopics({
  nodes,
  params,
  onNavigate,
  count,
  item,
}: {
  nodes: CategoryNode[]
  params: BrowseParams
  onNavigate?: () => void
  count: (path: string) => number
  item: (active: boolean) => string
}) {
  return (
    <div className="my-0.5 ml-3 space-y-0.5 border-l border-border/40 pl-2">
      {nodes.map((sub) => {
        const open = params.c === sub.path || params.c?.startsWith(`${sub.path}/`)
        const n = count(sub.path)
        return (
          <div key={sub.path}>
            <Link
              to={browseUrl({ ...params, c: sub.path, part: undefined })}
              onClick={onNavigate}
              className={cn(item(params.c === sub.path), 'py-0.5 text-xs', n === 0 && 'opacity-50')}
            >
              <span className="flex-1 truncate">{sub.title}</span>
              <span className="text-[10px] tabular-nums">{n}</span>
            </Link>
            {open && sub.children.length > 0 && (
              <Subtopics nodes={sub.children} params={params} onNavigate={onNavigate} count={count} item={item} />
            )}
          </div>
        )
      })}
    </div>
  )
}
