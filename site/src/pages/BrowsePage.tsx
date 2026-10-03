import { ChevronDown, FolderTree, List, Search, Waypoints } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, Navigate, useParams, useSearchParams } from 'react-router'
import { CategoryTrail } from '@/components/note/TaxonomyTrail'
import { ConceptMap } from '@/components/browse/ConceptMap'
import { filterNotes, inCategory } from '@/components/browse/filters'
import { NoteCard } from '@/components/browse/NoteCard'
import { TopicRail } from '@/components/browse/TopicRail'
import { kindIcons } from '@/components/layout/kind-icon'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  browseUrl,
  category,
  categoryOrder,
  categoryTrail,
  indexFirst,
  indexOfCategory,
  kindLabels,
  notes,
  taxonomy,
  type BrowseParams,
  type CategoryNode,
  type NoteKind,
} from '@/lib/content'
import { noteKinds } from '@/lib/content-schema'
import { cn } from '@/lib/utils'

/**
 * The explorer: topics on the left, notes as cards (or as a map) on the right, filtered by topic, kind and text. All
 * state lives in the URL, so every view can be linked. Below `lg` the topic rail moves into a sheet opened from one
 * button, so the notes start right under the filters instead of below every topic.
 */
export function BrowsePage() {
  const [search, setSearch] = useSearchParams()
  const params: BrowseParams = {
    c: search.get('c') ?? undefined,
    part: search.get('part') ?? undefined,
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

  const matches = useMemo(() => filterNotes(params), [params.c, params.part, params.kind, params.q]) // eslint-disable-line react-hooks/exhaustive-deps
  const kindCounts = useMemo(() => {
    const scoped = filterNotes({ c: params.c, part: params.part, q: params.q })
    return Object.fromEntries(noteKinds.map((k) => [k, scoped.filter((n) => n.kind === k).length]))
  }, [params.c, params.part, params.q])

  // Group cards under their category so a topic's structure is visible without reading.
  const categoryGroups = useMemo(() => {
    // An index note leads the group of the category it introduces, which may be an ancestor of its own folder.
    const inView = (path: string) => !params.c || path === params.c || path.startsWith(`${params.c}/`)
    const byCategory = new Map<string, typeof matches>()
    for (const n of matches) {
      const introduces = indexOfCategory(n.slug)
      const home = introduces && inView(introduces) ? introduces : n.category
      byCategory.set(home, [...(byCategory.get(home) ?? []), n])
    }
    return [...byCategory.entries()]
      .sort(([a], [b]) => categoryOrder(a) - categoryOrder(b))
      .map(([path, list]) => [path, indexFirst(list, path)] as const)
  }, [matches, params.c])

  const selectedPart = params.part
    ? taxonomy.find(
        (p) =>
          p.path === params.part ||
          String(p.num) === params.part ||
          `part-${p.num}` === params.part ||
          p.title.toLowerCase() === params.part?.toLowerCase(),
      )
    : undefined

  // When browsing all notes or an entire part, group the categories under the 8 Parts
  const partSections = useMemo(() => {
    if (params.c) return null
    const relevantParts = selectedPart ? [selectedPart] : taxonomy
    const sections: {
      part: CategoryNode
      categories: typeof categoryGroups
      totalNotes: number
    }[] = []

    for (const p of relevantParts) {
      const catsInPart = categoryGroups.filter(([catPath]) =>
        catPath === p.path || catPath.startsWith(`${p.path}/`),
      )
      if (catsInPart.length > 0) {
        const totalNotes = catsInPart.reduce((sum, [, list]) => sum + list.length, 0)
        sections.push({ part: p, categories: catsInPart, totalNotes })
      }
    }
    return sections
  }, [categoryGroups, params.c, selectedPart])

  const filtered = !!(params.c || params.part || params.kind || params.q)
  const scopeTitle = params.c
    ? (category(params.c)?.title ?? params.c)
    : selectedPart
      ? `${selectedPart.roman ? `${selectedPart.roman}: ` : ''}${selectedPart.title}`
      : 'All notes'

  return (
    <main className="px-4 py-8 lg:px-8">
      <div className="grid gap-8 lg:grid-cols-[250px_minmax(0,1fr)]">
        <aside className="hidden lg:block">
          <div className="sticky top-20 max-h-[calc(100svh-6rem)] overflow-y-auto overscroll-contain pr-2 pb-6">
            <TopicRail params={params} />
          </div>
        </aside>

        <div className="min-w-0 space-y-6">
          {/* The trail stays under the site header while the list scrolls, so the reader always sees where they are. */}
          {params.c && (
            <div className="sticky top-14 z-10 -mx-2 border-b bg-background/95 px-2 py-2 backdrop-blur">
              <CategoryTrail path={params.c} className="-ml-1" />
            </div>
          )}
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-prose text-3xl font-bold">{scopeTitle}</h1>
                {params.part && (
                  <button
                    type="button"
                    onClick={() => set({ part: undefined })}
                    className="text-xs text-muted-foreground hover:text-foreground underline"
                  >
                    (show all parts)
                  </button>
                )}
              </div>
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
            <div className="w-full lg:hidden">
              <TopicPicker params={params} />
            </div>
            <div className="relative w-full sm:w-72">
              <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={params.q ?? ''}
                onChange={(e) => set({ q: e.target.value })}
                placeholder="Filter by title, alias or tag"
                className="pl-8"
              />
            </div>
            {/* One scrolling row on phones; wrapped from sm up. */}
            <div className="-mx-4 flex w-[calc(100%+2rem)] gap-1.5 overflow-x-auto px-4 pb-1 sm:mx-0 sm:w-auto sm:flex-wrap sm:overflow-visible sm:px-0 sm:pb-0">
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

          {params.view === 'map' ? (
            <div className="rounded-lg border">
              <ConceptMap highlight={filtered ? new Set(matches.map((n) => n.slug)) : new Set()} />
            </div>
          ) : matches.length === 0 ? (
            <p className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              No notes match. Clear a filter or try another word.
            </p>
          ) : partSections ? (
            <div className="space-y-12">
              {partSections.map(({ part, categories, totalNotes }) => (
                <section key={part.title} className="space-y-6">
                  <div className="flex items-baseline justify-between border-b border-border/70 pb-2">
                    <div className="flex items-baseline gap-2.5">
                      <h2 className="font-prose text-xl font-bold tracking-tight">
                        {part.roman ? `${part.roman}: ` : ''}
                        {part.title}
                      </h2>
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {totalNotes} {totalNotes === 1 ? 'note' : 'notes'}
                      </span>
                    </div>
                    {!params.part && (
                      <Link
                        to={browseUrl({ ...params, part: part.path, c: undefined })}
                        className="text-xs text-muted-foreground hover:text-foreground hover:underline"
                      >
                        Focus this part →
                      </Link>
                    )}
                  </div>
                  <div className="space-y-8">
                    {categories.map(([path, list]) => (
                      <section key={path} className="space-y-3">
                        <h3 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                          {groupTitle(path, part.path)}
                        </h3>
                        <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                          {list.map((n) => (
                            <NoteCard key={n.slug} note={n} lead={category(path)?.index === n.slug} />
                          ))}
                        </div>
                      </section>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          ) : (
            <div className="space-y-8">
              {categoryGroups.map(([path, list]) => (
                <section key={path} className="space-y-3">
                  <h2 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                    {groupTitle(path, params.c)}
                  </h2>
                  <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                    {list.map((n) => (
                      <NoteCard key={n.slug} note={n} lead={category(path)?.index === n.slug} />
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
        'inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3 text-xs transition-colors disabled:opacity-40',
        active ? 'border-foreground bg-foreground text-background' : 'hover:bg-muted',
      )}
    >
      {Icon && <Icon className="size-3.5" aria-hidden />}
      {label}
      {count !== undefined && <span className="tabular-nums opacity-70">{count}</span>}
    </button>
  )
}

/** Below `lg`: the selected topic as one button, opening the topic rail in a sheet that closes on a choice. */
function TopicPicker({ params }: { params: BrowseParams }) {
  const [open, setOpen] = useState(false)
  const selectedPart = params.part
    ? taxonomy.find(
        (p) =>
          p.path === params.part ||
          String(p.num) === params.part ||
          `part-${p.num}` === params.part ||
          p.title.toLowerCase() === params.part?.toLowerCase(),
      )
    : undefined
  const title = params.c
    ? (category(params.c)?.title ?? params.c)
    : selectedPart
      ? `${selectedPart.roman ? `${selectedPart.roman}: ` : ''}${selectedPart.title}`
      : 'All notes'
  const count = notes.filter((n) => {
    if (params.c && !inCategory(n, params.c)) return false
    if (selectedPart && !inCategory(n, selectedPart.path)) return false
    return true
  }).length

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger render={<Button variant="outline" className="w-full justify-between" />}>
        <span className="flex min-w-0 items-center gap-2">
          <FolderTree className="size-4 text-muted-foreground" aria-hidden />
          <span className="truncate">{title}</span>
          <span className="text-xs text-muted-foreground tabular-nums">{count}</span>
        </span>
        <ChevronDown className="size-4 opacity-60" aria-hidden />
      </SheetTrigger>
      <SheetContent side="left" className="w-80 overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Topics</SheetTitle>
        </SheetHeader>
        <div className="px-4 pb-6">
          <TopicRail params={params} onNavigate={() => setOpen(false)} />
        </div>
      </SheetContent>
    </Sheet>
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
