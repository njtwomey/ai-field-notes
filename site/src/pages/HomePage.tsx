import { ArrowRight, ChevronDown, Search, Waypoints } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'
import { TopicIcon } from '@/components/layout/category-icon'
import { openSearch } from '@/components/layout/SearchCommand'
import { Kbd } from '@/components/ui/kbd'
import { browseUrl, groups, notes, notesInCategory, references, taxonomy, type CategoryNode } from '@/lib/content'
import { cn } from '@/lib/utils'

/** Landing page: search, then every topic with its subtopics, in the groups of content/groups.yaml. */
export function HomePage() {
  return (
    <main className="px-4 pt-14 pb-20 lg:px-8">
      <section className="mx-auto max-w-2xl space-y-6 text-center">
        <h1 className="font-prose text-5xl font-bold">AI Field Notes</h1>
        <p className="font-prose text-lg text-muted-foreground">
          Machine learning notes from my research career, digitised and made interactive.
        </p>
        <button
          type="button"
          onClick={openSearch}
          className="flex h-12 w-full items-center gap-3 rounded-xl border bg-muted/40 px-4 text-left text-muted-foreground transition-colors hover:bg-muted"
        >
          <Search className="size-5" aria-hidden />
          <span className="flex-1">Search concepts, techniques and tests…</span>
          <Kbd>⌘K</Kbd>
        </button>
        <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-sm text-muted-foreground">
          <span>
            {notes.length} notes · {taxonomy.length} topics · {Object.keys(references).length} references
          </span>
          <Link to={browseUrl()} className="inline-flex items-center gap-1 text-foreground hover:underline">
            Browse all <ArrowRight className="size-3.5" aria-hidden />
          </Link>
          <Link
            to={browseUrl({ view: 'map' })}
            className="inline-flex items-center gap-1 text-foreground hover:underline"
          >
            <Waypoints className="size-3.5" aria-hidden /> Open the map
          </Link>
        </div>
      </section>

      <div className="mt-16 space-y-12">
        {groups.map((group) => (
          <section key={group.title} aria-labelledby={groupId(group.title)}>
            <div className="mb-4 flex items-baseline gap-3">
              <h2 id={groupId(group.title)} className="font-prose text-xl font-bold">
                {group.title}
              </h2>
              <span className="text-xs text-muted-foreground tabular-nums">
                {group.topics.reduce((sum, t) => sum + notesInCategory(t).length, 0)} notes
              </span>
              <div className="h-px flex-1 self-center bg-border" />
            </div>
            <div className="grid items-start gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
              {group.topics.map((path) => (
                <TopicTile key={path} topic={taxonomy.find((t) => t.path === path)!} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  )
}

const groupId = (title: string) => `group-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`

/** Subtopics shown before a tile is expanded. Every collapsed tile reserves exactly this many rows, so all match. */
const TILE_ROWS = 6

function TopicTile({ topic }: { topic: CategoryNode }) {
  const [open, setOpen] = useState(false)
  const extra = topic.children.length - TILE_ROWS
  const shown = open ? topic.children : topic.children.slice(0, TILE_ROWS)
  // Invisible rows pad short tiles to the same height as full ones.
  const padding = Math.max(0, TILE_ROWS - shown.length)
  return (
    <div className="flex flex-col rounded-xl border p-5">
      <Link to={browseUrl({ c: topic.path })} className="group mb-3 flex items-center gap-2.5">
        <span className="flex size-8 items-center justify-center rounded-lg bg-muted">
          <TopicIcon icon={topic.icon} className="size-4" aria-hidden />
        </span>
        <span className="flex-1 font-medium group-hover:underline">{topic.title}</span>
        <span className="text-xs text-muted-foreground tabular-nums">{notesInCategory(topic.path).length}</span>
      </Link>
      <ul className="space-y-1">
        {shown.map((child) => {
          const count = notesInCategory(child.path).length
          return (
            <li key={child.path}>
              <Link
                to={browseUrl({ c: child.path })}
                className={cn(
                  '-mx-2 flex items-center gap-2 rounded-md px-2 py-1 text-sm hover:bg-muted',
                  count === 0 && 'text-muted-foreground',
                )}
              >
                <span className="flex-1 truncate">{child.title}</span>
                <span className="text-xs text-muted-foreground tabular-nums">{count}</span>
              </Link>
            </li>
          )
        })}
        {Array.from({ length: padding }, (_, k) => (
          <li key={`pad-${k}`} aria-hidden className="invisible px-2 py-1 text-sm">
            &nbsp;
          </li>
        ))}
      </ul>
      {extra > 0 ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="-mx-2 mt-2 flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <ChevronDown className={cn('size-3.5 transition-transform', open && 'rotate-180')} aria-hidden />
          {open ? 'Show fewer' : `Show all ${topic.children.length}`}
        </button>
      ) : (
        <div aria-hidden className="invisible mt-2 px-2 py-1 text-xs">
          &nbsp;
        </div>
      )}
    </div>
  )
}
