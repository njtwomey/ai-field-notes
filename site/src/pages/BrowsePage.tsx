import { List, Search, Waypoints } from 'lucide-react'
import { useMemo } from 'react'
import { Navigate, useParams, useSearchParams } from 'react-router'
import { ConceptMap } from '@/components/browse/ConceptMap'
import { filterNotes } from '@/components/browse/filters'
import { NoteCard } from '@/components/browse/NoteCard'
import { TopicRail } from '@/components/browse/TopicRail'
import { kindIcons } from '@/components/layout/kind-icon'
import { Input } from '@/components/ui/input'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  browseUrl,
  category,
  categoryOrder,
  categoryTrail,
  kindLabels,
  notes,
  type BrowseParams,
  type NoteKind,
} from '@/lib/content'
import { noteKinds } from '@/lib/content-schema'
import { cn } from '@/lib/utils'

/**
 * The explorer: topics on the left, notes as cards (or as a map) on the right, filtered by topic, kind and text. All
 * state lives in the URL, so every view can be linked.
 */
export function BrowsePage() {
  const [search, setSearch] = useSearchParams()
  const params: BrowseParams = {
    c: search.get('c') ?? undefined,
    kind: (search.get('kind') as NoteKind | null) ?? undefined,
    q: search.get('q') ?? undefined,
    view: search.get('view') === 'map' ? 'map' : 'list',
  }
  const set = (patch: Partial<BrowseParams>) => {
    const next = new URLSearchParams(search)
    for (const [k, v] of Object.entries(patch)) {
      if (v) next.set(k, v)
      else next.delete(k)
    }
    setSearch(next, { replace: true })
  }

  const matches = useMemo(() => filterNotes(params), [params.c, params.kind, params.q]) // eslint-disable-line react-hooks/exhaustive-deps
  const kindCounts = useMemo(() => {
    const scoped = filterNotes({ c: params.c, q: params.q })
    return Object.fromEntries(noteKinds.map((k) => [k, scoped.filter((n) => n.kind === k).length]))
  }, [params.c, params.q])

  // Group cards under their category so a topic's structure is visible without reading.
  const groups = useMemo(() => {
    const byCategory = new Map<string, typeof matches>()
    for (const n of matches) byCategory.set(n.category, [...(byCategory.get(n.category) ?? []), n])
    return [...byCategory.entries()].sort(([a], [b]) => categoryOrder(a) - categoryOrder(b))
  }, [matches])

  const filtered = !!(params.c || params.kind || params.q)
  const scopeTitle = params.c ? category(params.c)?.title : 'All notes'

  return (
    <main className="px-4 py-8 lg:px-8">
      <div className="grid gap-8 lg:grid-cols-[240px_minmax(0,1fr)]">
        <aside className="hidden lg:block">
          <div className="sticky top-20">
            <TopicRail params={params} />
          </div>
        </aside>

        <div className="min-w-0 space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h1 className="font-prose text-3xl font-bold">{scopeTitle}</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                {matches.length} of {notes.length} notes
              </p>
            </div>
            <ToggleGroup
              value={[params.view ?? 'list']}
              onValueChange={(v) => v[0] && set({ view: v[0] === 'map' ? 'map' : undefined })}
              variant="outline"
              size="sm"
              spacing={0}
            >
              <ToggleGroupItem value="list" className="px-3">
                <List /> List
              </ToggleGroupItem>
              <ToggleGroupItem value="map" className="px-3">
                <Waypoints /> Map
              </ToggleGroupItem>
            </ToggleGroup>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="relative w-full sm:w-72">
              <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={params.q ?? ''}
                onChange={(e) => set({ q: e.target.value })}
                placeholder="Filter by title, alias or tag"
                className="pl-8"
              />
            </div>
            <div className="flex flex-wrap gap-1.5">
              <KindChip active={!params.kind} onClick={() => set({ kind: undefined })} label="All" />
              {noteKinds.map((k) => (
                <KindChip
                  key={k}
                  kind={k}
                  active={params.kind === k}
                  count={kindCounts[k]}
                  onClick={() => set({ kind: params.kind === k ? undefined : k })}
                  label={kindLabels[k]}
                />
              ))}
            </div>
          </div>

          {/* Topics as chips on small screens, where the rail is hidden. */}
          <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 lg:hidden">
            <TopicRail params={params} />
          </div>

          {params.view === 'map' ? (
            <div className="rounded-lg border">
              <ConceptMap highlight={filtered ? new Set(matches.map((n) => n.slug)) : new Set()} />
            </div>
          ) : matches.length === 0 ? (
            <p className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              No notes match. Clear a filter or try another word.
            </p>
          ) : (
            <div className="space-y-8">
              {groups.map(([path, list]) => (
                <section key={path} className="space-y-3">
                  <h2 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                    {groupTitle(path, params.c)}
                  </h2>
                  <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                    {list.map((n) => (
                      <NoteCard key={n.slug} note={n} />
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

function KindChip({
  kind,
  label,
  count,
  active,
  onClick,
}: {
  kind?: NoteKind
  label: string
  count?: number
  active: boolean
  onClick: () => void
}) {
  const Icon = kind ? kindIcons[kind] : undefined
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={count === 0 && !active}
      aria-pressed={active}
      className={cn(
        'inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs transition-colors disabled:opacity-40',
        active ? 'border-foreground bg-foreground text-background' : 'hover:bg-muted',
      )}
    >
      {Icon && <Icon className="size-3.5" aria-hidden />}
      {label}
      {count !== undefined && <span className="tabular-nums opacity-70">{count}</span>}
    </button>
  )
}

/**
 * A group's heading: its category trail below the selected topic, which the rail already shows. A group that is the
 * selected category itself keeps its own title.
 */
function groupTitle(path: string, selected?: string): string {
  const trail = categoryTrail(path)
  const below = selected ? trail.filter((c) => !`${selected}/`.startsWith(`${c.path}/`)) : trail
  return (below.length ? below : trail.slice(-1)).map((c) => c.title).join(' › ')
}

/** Old category URLs (/c/…) open the explorer filtered to that category. */
export function CategoryRedirect() {
  const path = useParams()['*'] ?? ''
  return <Navigate to={browseUrl({ c: path })} replace />
}
