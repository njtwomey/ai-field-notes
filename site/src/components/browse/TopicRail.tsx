import { Layers } from 'lucide-react'
import { Link } from 'react-router'
import { TopicIcon } from '@/components/layout/category-icon'
import { browseUrl, notes, taxonomy, type BrowseParams, type CategoryNode } from '@/lib/content'
import { cn } from '@/lib/utils'
import { inCategory } from './filters'

/**
 * Topics with note counts. The selected topic expands to its subtopics. Links keep the other browse filters.
 */
export function TopicRail({ params, onNavigate }: { params: BrowseParams; onNavigate?: () => void }) {
  const count = (path: string) => notes.filter((n) => inCategory(n, path)).length
  const item = (active: boolean) =>
    cn(
      'flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-muted',
      active ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground',
    )

  return (
    <nav aria-label="Topics" className="space-y-0.5">
      <Link to={browseUrl({ ...params, c: undefined })} onClick={onNavigate} className={item(!params.c)}>
        <Layers className="size-4" aria-hidden />
        <span className="flex-1">All topics</span>
        <span className="text-xs tabular-nums">{notes.length}</span>
      </Link>
      {taxonomy.map((topic) => {
        const open = params.c === topic.path || params.c?.startsWith(`${topic.path}/`)
        const n = count(topic.path)
        return (
          <div key={topic.path}>
            <Link
              to={browseUrl({ ...params, c: topic.path })}
              onClick={onNavigate}
              className={cn(item(params.c === topic.path), n === 0 && 'opacity-50')}
            >
              <TopicIcon icon={topic.icon} className="size-4" aria-hidden />
              <span className="flex-1">{topic.title}</span>
              <span className="text-xs tabular-nums">{n}</span>
            </Link>
            {open && topic.children.length > 0 && (
              <Subtopics nodes={topic.children} params={params} onNavigate={onNavigate} count={count} item={item} />
            )}
          </div>
        )
      })}
    </nav>
  )
}

/** Subtopics at any depth, shown once their top-level topic is open. */
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
    <div className="my-0.5 ml-4 space-y-0.5 border-l pl-2">
      {nodes.map((sub) => {
        return (
          <div key={sub.path}>
            <Link
              to={browseUrl({ ...params, c: sub.path })}
              onClick={onNavigate}
              className={cn(item(params.c === sub.path), 'py-1 text-xs', count(sub.path) === 0 && 'opacity-50')}
            >
              <span className="flex-1">{sub.title}</span>
              <span className="tabular-nums">{count(sub.path)}</span>
            </Link>
            {sub.children.length > 0 && (
              <Subtopics nodes={sub.children} params={params} onNavigate={onNavigate} count={count} item={item} />
            )}
          </div>
        )
      })}
    </div>
  )
}
