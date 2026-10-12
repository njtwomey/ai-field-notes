import { Layers } from 'lucide-react'
import { useRef } from 'react'
import { Link } from 'react-router'
import { TopicIcon } from '@/components/layout/category-icon'
import { useScrollToCurrent } from '@/hooks/use-scroll-into-panel'
import { browseUrl, notes, taxonomy, type BrowseParams, type CategoryNode } from '@/lib/content'
import { useHideStubs } from '@/lib/stubs'
import { cn } from '@/lib/utils'
import { inCategory } from './filters'

/**
 * The topic index: every part, and below them only the branch that holds the current selection, opened down to it.
 * The current entry is centred in the rail's scroll box whenever the selection changes, so a deep topic opens in view.
 */
export function TopicRail({ params, onNavigate }: { params: BrowseParams; onNavigate?: () => void }) {
  const hideStubs = useHideStubs()
  const shown = hideStubs ? notes.filter((n) => n.status !== 'stub') : notes
  const count = (path: string) => shown.filter((n) => inCategory(n, path)).length
  const selected = params.c
  const navRef = useRef<HTMLElement>(null)
  useScrollToCurrent(navRef, selected)

  return (
    <nav ref={navRef} aria-label="Topics" className="space-y-3">
      <Link
        to={browseUrl({ ...params, c: undefined })}
        onClick={onNavigate}
        aria-current={!selected ? 'page' : undefined}
        className={cn(rowClass(!selected), 'py-1.5 text-sm')}
      >
        <Layers className="size-4 shrink-0" aria-hidden />
        <span className="flex-1 font-medium">All parts</span>
        <span className="text-xs text-muted-foreground tabular-nums">{shown.length}</span>
      </Link>

      <div className="space-y-1 pt-1">
        {taxonomy.map((part) => {
          const active = selected === part.path
          const open = active || onPathTo(selected, part.path)
          return (
            <section key={part.path} className="space-y-1">
              <Link
                to={browseUrl({ ...params, c: part.path })}
                onClick={onNavigate}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'group flex items-center justify-between gap-1 rounded-md px-2 py-1 text-[11px] font-semibold tracking-wider uppercase transition-colors select-none',
                  active || open
                    ? 'bg-muted text-foreground'
                    : 'text-muted-foreground/80 hover:bg-muted/60 hover:text-foreground',
                )}
              >
                <span className="truncate">
                  {part.roman ? `${part.roman} · ` : ''}
                  {part.title}
                </span>
                <span className="font-mono text-[10px] text-muted-foreground/70 tabular-nums group-hover:text-foreground">
                  {count(part.path)}
                </span>
              </Link>
              {open && part.children.length > 0 && (
                <Subtopics nodes={part.children} params={params} onNavigate={onNavigate} count={count} depth={1} />
              )}
            </section>
          )
        })}
      </div>
    </nav>
  )
}

/** Whether `path` is a strict ancestor of the selection. */
const onPathTo = (selected: string | undefined, path: string) => !!selected?.startsWith(`${path}/`)

const rowClass = (active: boolean) =>
  cn(
    'flex items-center gap-2 rounded-md px-2 py-1 text-xs transition-colors hover:bg-muted',
    active ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground',
  )

/** The topics under an open part or topic; only the one on the path to the selection opens further. */
function Subtopics({
  nodes,
  params,
  onNavigate,
  count,
  depth,
}: {
  nodes: CategoryNode[]
  params: BrowseParams
  onNavigate?: () => void
  count: (path: string) => number
  depth: number
}) {
  return (
    <div className={cn('space-y-0.5 border-l border-border/40', depth === 1 ? 'ml-2 pl-1.5' : 'my-0.5 ml-3 pl-2')}>
      {nodes.map((sub) => {
        const active = params.c === sub.path
        const open = active || onPathTo(params.c, sub.path)
        const n = count(sub.path)
        return (
          <div key={sub.path}>
            <Link
              to={browseUrl({ ...params, c: sub.path })}
              onClick={onNavigate}
              aria-current={active ? 'page' : undefined}
              className={cn(
                rowClass(active),
                depth > 1 && 'py-0.5',
                open && !active && 'text-foreground',
                n === 0 && 'opacity-50',
              )}
            >
              {depth === 1 && <TopicIcon icon={sub.icon} className="size-3.5 shrink-0" aria-hidden />}
              <span className="flex-1 truncate">{sub.title}</span>
              <span className="text-[10px] tabular-nums">{n}</span>
            </Link>
            {open && sub.children.length > 0 && (
              <Subtopics nodes={sub.children} params={params} onNavigate={onNavigate} count={count} depth={depth + 1} />
            )}
          </div>
        )
      })}
    </div>
  )
}
