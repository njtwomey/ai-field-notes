import { ChevronDown, FolderTree, Search } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { Navigate, useParams, useSearchParams } from 'react-router'
import { CategoryTrail } from '@/components/note/TaxonomyTrail'
import { filterNotes, inCategory } from '@/components/browse/filters'
import { NoteRow, PartsIndex, TopicContents } from '@/components/browse/Outline'
import { StubToggle } from '@/components/browse/StubToggle'
import { TopicRail } from '@/components/browse/TopicRail'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import {
  browseUrl,
  category,
  categoryOrder,
  categoryTrail,
  indexFirst,
  indexOfCategory,
  notes,
  taxonomy,
  type BrowseParams,
  type CategoryNode,
  type NoteKind,
  type NoteMeta,
} from '@/lib/content'
import { useHideStubs, withoutStubs } from '@/lib/stubs'

/**
 * The explorer, one level at a time. With nothing selected it lists the parts; with a part or topic selected it lists
 * what sits directly under it: the topic's introducing note, its subtopics, then its own notes. Typing a filter or
 * picking a kind switches to a flat list of the matching notes anywhere under the selection. All state lives in the
 * URL, so every view can be linked. Below `lg` the topic rail moves into a sheet opened from one button.
 */
export function BrowsePage() {
  const [search, setSearch] = useSearchParams()
  const params: BrowseParams = {
    c: search.get('c') ?? undefined,
    kind: (search.get('kind') as NoteKind | null) ?? undefined,
    q: search.get('q') ?? undefined,
  }
  const set = (patch: Partial<BrowseParams>) => {
    const next = new URLSearchParams(search)
    for (const [k, v] of Object.entries(patch)) {
      if (v) next.set(k, v)
      else next.delete(k)
    }
    setSearch(next, { replace: true })
  }

  const hideStubs = useHideStubs()
  const shown = withoutStubs(notes, hideStubs)
  const count = (path: string) => shown.filter((n) => inCategory(n, path)).length
  const node = params.c ? category(params.c) : undefined
  const filtering = !!(params.q || params.kind)

  const matches = filterNotes({ c: params.c, kind: params.kind, q: params.q, hideStubs })

  // Old links named a part with ?part=…; the part is now a selection like any other topic.
  const legacyPart = search.get('part')
  if (legacyPart) {
    const part = findPart(legacyPart)
    const next = new URLSearchParams(search)
    next.delete('part')
    if (part) next.set('c', part.path)
    next.delete('view')
    return <Navigate to={`/browse${next.size ? `?${next}` : ''}`} replace />
  }

  const title = node ? partTitle(node) : 'All parts'
  const total = node ? count(node.path) : shown.length

  return (
    <main className="px-4 py-8 lg:px-8">
      <div className="grid gap-8 lg:grid-cols-[250px_minmax(0,1fr)]">
        <aside className="hidden lg:block">
          <div className="sticky top-20 max-h-[calc(100svh-6rem)] overflow-y-auto overscroll-contain pr-2 pb-6">
            <TopicRail params={params} />
          </div>
        </aside>

        <div className="min-w-0 space-y-6">
          {/* The trail names the page and stays under the site header while the list scrolls; the filter sits beside it,
              so the contents start at the top. */}
          <div className="sticky top-14 z-10 -mx-2 flex flex-wrap items-center gap-x-4 gap-y-2 border-b bg-background/95 px-2 py-2 backdrop-blur">
            <h1 className="sr-only">{title}</h1>
            <CategoryTrail path={node?.path ?? ''} className="-ml-1 min-w-0 flex-1" />
            <div className="flex w-full items-center gap-3 sm:w-auto">
              <FilterInput
                value={params.q}
                onChange={(q) => set({ q })}
                placeholder={node ? `Filter ${node.title}` : 'Filter all notes'}
              />
              <StubToggle />
            </div>
          </div>
          <div className="lg:hidden">
            <TopicPicker params={params} title={title} total={total} />
          </div>
          {filtering && (
            <p className="text-sm text-muted-foreground">
              {matches.length} of {total} notes match
            </p>
          )}

          {filtering ? (
            <Matches matches={matches} selected={params.c} />
          ) : node ? (
            <TopicContents node={node} shown={shown} />
          ) : (
            <PartsIndex parts={taxonomy} shown={shown} />
          )}
        </div>
      </div>
    </main>
  )
}

/** Filter results: every matching note under the selection, grouped under its topic in taxonomy order. */
function Matches({ matches, selected }: { matches: NoteMeta[]; selected?: string }) {
  const groups = useMemo(() => {
    // An index note leads the group of the category it introduces, which may be an ancestor of its own folder.
    const inView = (path: string) => !selected || path === selected || path.startsWith(`${selected}/`)
    const byCategory = new Map<string, NoteMeta[]>()
    for (const n of matches) {
      const introduces = indexOfCategory(n.slug)
      const home = introduces && inView(introduces) ? introduces : n.category
      byCategory.set(home, [...(byCategory.get(home) ?? []), n])
    }
    return [...byCategory.entries()]
      .sort(([a], [b]) => categoryOrder(a) - categoryOrder(b))
      .map(([path, list]) => [path, indexFirst(list, path)] as const)
  }, [matches, selected])

  if (!matches.length)
    return (
      <p className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
        No notes match. Clear a filter or try another word.
      </p>
    )
  return (
    <div className="space-y-8">
      {groups.map(([path, list]) => (
        <section key={path} className="space-y-3">
          <SectionHeading>{groupTitle(path, selected)}</SectionHeading>
          <ol className="divide-y border-y">
            {list.map((n) => (
              <NoteRow key={n.slug} note={n} />
            ))}
          </ol>
        </section>
      ))}
    </div>
  )
}

function SectionHeading({ children }: { children: ReactNode }) {
  return <h2 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">{children}</h2>
}

/** The text filter: title, aliases, tags and summary. */
function FilterInput({
  value,
  onChange,
  placeholder,
}: {
  value?: string
  onChange: (q: string) => void
  placeholder: string
}) {
  return (
    <div className="relative w-full sm:w-64">
      <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="h-8 pl-8"
      />
    </div>
  )
}

/** Below `lg`: the selected topic as one button, opening the topic rail in a sheet that closes on a choice. */
function TopicPicker({ params, title, total }: { params: BrowseParams; title: string; total: number }) {
  const [open, setOpen] = useState(false)
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger render={<Button variant="outline" className="w-full justify-between" />}>
        <span className="flex min-w-0 items-center gap-2">
          <FolderTree className="size-4 text-muted-foreground" aria-hidden />
          <span className="truncate">{title}</span>
          <span className="text-xs text-muted-foreground tabular-nums">{total}</span>
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

/** A part by its path, number, `part-N` or title, as old `?part=` links named it. */
function findPart(part: string): CategoryNode | undefined {
  return taxonomy.find(
    (p) =>
      p.path === part ||
      String(p.num) === part ||
      `part-${p.num}` === part ||
      p.title.toLowerCase() === part.toLowerCase(),
  )
}

/** A topic's title, with its numeral when it is a part. */
const partTitle = (node: CategoryNode) => `${node.roman ? `${node.roman}: ` : ''}${node.title}`

/**
 * A group's heading: its category trail below the selected topic, which the rail already shows. A group that is the
 * selected category itself keeps its own title.
 */
function groupTitle(path: string, selected?: string): string {
  const trail = categoryTrail(path)
  const below = selected ? trail.filter((c) => !`${selected}/`.startsWith(`${c.path}/`)) : trail
  return (below.length ? below : trail.slice(-1)).map((c) => c.title).join(' › ')
}

/** Old category URLs (/c/…) open the explorer at that category. */
export function CategoryRedirect() {
  const path = useParams()['*'] ?? ''
  return <Navigate to={browseUrl({ c: path })} replace />
}
