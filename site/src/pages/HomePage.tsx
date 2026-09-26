import { ArrowRight, Search, Waypoints } from 'lucide-react'
import { Link } from 'react-router'
import { categoryIcon } from '@/components/layout/category-icon'
import { kindIcons } from '@/components/layout/kind-icon'
import { openSearch } from '@/components/layout/SearchCommand'
import { Kbd } from '@/components/ui/kbd'
import { browseUrl, kindLabels, notes, notesInCategory, noteUrl, references, taxonomy, topicOf } from '@/lib/content'
import { cn } from '@/lib/utils'

/** Landing page: search, then every topic with its subtopics, then what changed recently. */
export function HomePage() {
  const recent = [...notes].sort((a, b) => b.updated.localeCompare(a.updated)).slice(0, 6)
  return (
    <main className="mx-auto max-w-6xl px-4 pt-14 pb-20 lg:px-8">
      <section className="mx-auto max-w-2xl space-y-6 text-center">
        <h1 className="font-prose text-5xl font-bold">AI Field Notes</h1>
        <p className="font-prose text-lg text-muted-foreground">
          Notes on AI and machine learning gathered over a research career, digitised and curated in one place.
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

      <section className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {taxonomy.map((topic) => {
          const Icon = categoryIcon(topic.icon)
          return (
            <div key={topic.path} className="flex flex-col rounded-xl border p-5">
              <Link to={browseUrl({ c: topic.path })} className="group mb-3 flex items-center gap-2.5">
                <span className="flex size-8 items-center justify-center rounded-lg bg-muted">
                  <Icon className="size-4" aria-hidden />
                </span>
                <span className="flex-1 font-medium group-hover:underline">{topic.title}</span>
                <span className="text-xs text-muted-foreground tabular-nums">{notesInCategory(topic.path).length}</span>
              </Link>
              <ul className="space-y-1">
                {topic.children.map((child) => {
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
              </ul>
            </div>
          )
        })}
      </section>

      <section className="mt-16">
        <h2 className="mb-3 text-xs font-medium tracking-wide text-muted-foreground uppercase">Recently updated</h2>
        <ul className="divide-y rounded-xl border">
          {recent.map((n) => {
            const Kind = kindIcons[n.kind]
            return (
              <li key={n.slug}>
                <Link to={noteUrl(n.slug)} className="flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-muted/50">
                  <Kind className="size-4 shrink-0 text-muted-foreground" aria-label={kindLabels[n.kind]} />
                  <span className="font-medium">{n.title}</span>
                  <span className="hidden truncate text-muted-foreground sm:inline">{topicOf(n).title}</span>
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums">{n.updated}</span>
                </Link>
              </li>
            )
          })}
        </ul>
      </section>
    </main>
  )
}
