import { ChevronDown, ChevronRight, FolderTree, Search } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { Link, Navigate, useParams, useSearchParams } from 'react-router'
import { CategoryTrail } from '@/components/note/TaxonomyTrail'
import { filterNotes, inCategory } from '@/components/browse/filters'
import { NoteCard } from '@/components/browse/NoteCard'
import { StubToggle } from '@/components/browse/StubToggle'
import { TopicRail } from '@/components/browse/TopicRail'
import { TopicIcon } from '@/components/layout/category-icon'
import { kindIcons } from '@/components/layout/kind-icon'
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
  kindLabels,
  notes,
  taxonomy,
  type BrowseParams,
  type CategoryNode,
  type NoteKind,
  type NoteMeta,
} from '@/lib/content'
import { noteKinds } from '@/lib/content-schema'
import { useHideStubs, withoutStubs } from '@/lib/stubs'
import { cn } from '@/lib/utils'

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
  const scoped = filterNotes({ c: params.c, q: params.q, hideStubs })
  const kindCounts = Object.fromEntries(noteKinds.map((k) => [k, scoped.filter((n) => n.kind === k).length]))

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
          {/* The trail stays under the site header while the list scrolls, so the reader always sees where they are. */}
          {params.c && (
            <div className="sticky top-14 z-10 -mx-2 border-b bg-background/95 px-2 py-2 backdrop-blur">
              <CategoryTrail path={params.c} className="-ml-1" />
            </div>
          )}
          <div>
            <h1 className="font-prose text-3xl font-bold">{title}</h1>
            {node?.description && <p className="mt-2 max-w-3xl text-muted-foreground">{node.description}</p>}
            <p className="mt-1 text-sm text-muted-foreground">
              {filtering ? `${matches.length} of ${total} notes match` : `${total} notes`}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="w-full lg:hidden">
              <TopicPicker params={params} title={title} total={total} />
            </div>
            <div className="relative w-full sm:w-72">
              <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={params.q ?? ''}
                onChange={(e) => set({ q: e.target.value })}
                placeholder={node ? `Filter notes in ${node.title}` : 'Filter by title, alias or tag'}
                className="pl-8"
              />
            </div>
            <StubToggle />
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

          {filtering ? (
            <Matches matches={matches} selected={params.c} />
          ) : node ? (
            <Contents node={node} shown={shown} count={count} />
          ) : (
            <TopicGrid nodes={taxonomy} count={count} />
          )}
        </div>
      </div>
    </main>
  )
}

/**
 * What sits directly under a part or topic. The introducing note leads; subtopics follow as cards, since they are the
 * main structure; then the notes filed in the topic itself, in reading order. A heading names each kind only when
 * both are present.
 */
function Contents({ node, shown, count }: { node: CategoryNode; shown: NoteMeta[]; count: (path: string) => number }) {
  const lead = node.index ? shown.find((n) => n.slug === node.index) : undefined
  const own = shown.filter((n) => n.category === node.path && n.slug !== lead?.slug)
  const both = node.children.length > 0 && own.length > 0
  if (!lead && !node.children.length && !own.length)
    return (
      <p className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
        No notes here yet.
      </p>
    )
  return (
    <div className="space-y-8">
      {lead && (
        <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
          <NoteCard note={lead} lead />
        </div>
      )}
      {node.children.length > 0 && (
        <section className="space-y-3">
          {both && <SectionHeading>Topics</SectionHeading>}
          <TopicGrid nodes={node.children} count={count} />
        </section>
      )}
      {own.length > 0 && (
        <section className="space-y-3">
          {both && <SectionHeading>Notes</SectionHeading>}
          <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
            {own.map((n) => (
              <NoteCard key={n.slug} note={n} />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

/** Parts or subtopics as cards that open them: title, description and the notes inside. */
function TopicGrid({ nodes, count }: { nodes: CategoryNode[]; count: (path: string) => number }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
      {nodes.map((c) => {
        const n = count(c.path)
        return (
          <Link
            key={c.path}
            to={browseUrl({ c: c.path })}
            className={cn(
              'group flex flex-col gap-1.5 rounded-lg border p-4 transition-colors hover:border-foreground/30 hover:bg-muted/40',
              n === 0 && 'opacity-60',
            )}
          >
            {c.roman && (
              <span className="pl-6 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
                {c.roman}
              </span>
            )}
            <div className="flex items-start gap-2">
              <TopicIcon icon={c.icon} className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
              <span className="leading-snug font-medium group-hover:underline">{c.title}</span>
              <ChevronRight className="mt-0.5 ml-auto size-4 shrink-0 text-muted-foreground" aria-hidden />
            </div>
            <span className="pl-6 text-xs text-muted-foreground tabular-nums">
              {n} {n === 1 ? 'note' : 'notes'}
              {c.children.length > 0 && ` · ${c.children.length} ${c.children.length === 1 ? 'topic' : 'topics'}`}
            </span>
            {c.description && <p className="line-clamp-2 pl-6 text-sm text-muted-foreground">{c.description}</p>}
          </Link>
        )
      })}
    </div>
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
          <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
            {list.map((n) => (
              <NoteCard key={n.slug} note={n} lead={category(path)?.index === n.slug} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}

function SectionHeading({ children }: { children: ReactNode }) {
  return <h2 className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">{children}</h2>
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
