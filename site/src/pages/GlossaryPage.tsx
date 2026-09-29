import { ArrowRight, ChevronLeft, ChevronRight, FileText, Search, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router'
import { MathText } from '@/components/content/MathText'
import { useDebouncedCallback } from '@/hooks/use-debounced-callback'
import { PageContainer } from '@/components/layout/AppShell'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  category,
  categoryOrder,
  categoryTrail,
  notesBySlug,
  noteUrl,
  type GlossaryEntry,
  type GlossaryKind,
} from '@/lib/content'
import { glossaryEntries, glossaryKindLabels, glossaryUsage, glossHeadword } from '@/lib/glossary'
import { plainMath } from '@/lib/math-text'
import { cn } from '@/lib/utils'

const PAGE_SIZE = 100

/** Lower-case and strip accents, so "frechet" finds "Fréchet". */
const fold = (s: string) =>
  s
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()

type Row = { e: GlossaryEntry; head: string; haystack: string; topics: string[] }

const ROWS: Row[] = glossaryEntries.map((e) => ({
  e,
  head: fold(plainMath(glossHeadword(e))),
  haystack: fold(
    [e.key, ...e.aliases, e.short ?? '', e.long, e.sense ?? '', plainMath(e.definition)].join(' ').replace(/-/g, ' '),
  ),
  topics: [...new Set(e.category.map((c) => c.split('/')[0]))],
}))

type SortKey = 'az' | 'topic'
const kinds = Object.keys(glossaryKindLabels) as GlossaryKind[]

/** The first category of an entry under `c`, or its first category; used to order and group by topic. */
const primaryCategory = (e: GlossaryEntry, c?: string) =>
  e.category.find((p) => !c || p === c || p.startsWith(`${c}/`)) ?? e.category[0]

/** Every entry of content/glossary.yaml: searchable, filtered by kind and category, sorted A to Z or by topic. */
export function GlossaryPage() {
  const [params, setParams] = useSearchParams()
  const { hash } = useLocation()
  const q = params.get('q') ?? ''
  const kind = kinds.find((k) => k === params.get('kind'))
  const c = params.get('c') && category(params.get('c')!) ? params.get('c')! : undefined
  const by: SortKey = params.get('sort') === 'topic' ? 'topic' : 'az'
  const set = (next: Record<string, string | undefined>) =>
    setParams(
      (prev) => {
        const p = new URLSearchParams(prev)
        for (const [k, v] of Object.entries(next)) {
          if (v === undefined || v === '') p.delete(k)
          else p.set(k, v)
        }
        return p
      },
      { replace: true },
    )

  // The input is immediate; the query (URL, filter, KaTeX re-render of the list) follows once typing pauses.
  const [text, setText] = useState(q)
  const commitQuery = useDebouncedCallback((v: string) => set({ q: v, page: undefined }), 200, Infinity)

  const inCategory = (r: Row) => !c || r.e.category.some((p) => p === c || p.startsWith(`${c}/`))
  const matches = useMemo(() => {
    const terms = fold(q).replace(/-/g, ' ').split(/\s+/).filter(Boolean)
    const hits = ROWS.filter(
      (r) => (!kind || r.e.kind === kind) && inCategory(r) && terms.every((t) => r.haystack.includes(t)),
    )
    return hits.sort((a, b) =>
      by === 'topic'
        ? categoryOrder(primaryCategory(a.e, c)) - categoryOrder(primaryCategory(b.e, c)) ||
          a.head.localeCompare(b.head)
        : a.head.localeCompare(b.head) || a.e.key.localeCompare(b.e.key),
    )
  }, [q, kind, c, by]) // eslint-disable-line react-hooks/exhaustive-deps

  // Counts for the topic rail ignore the category filter itself, so every topic stays reachable.
  const counts = useMemo(() => {
    const terms = fold(q).replace(/-/g, ' ').split(/\s+/).filter(Boolean)
    const out = new Map<string, number>([['', 0]])
    for (const r of ROWS) {
      if ((kind && r.e.kind !== kind) || !terms.every((t) => r.haystack.includes(t))) continue
      out.set('', out.get('')! + 1)
      const paths = new Set(r.e.category.flatMap((p) => p.split('/').map((_, i, a) => a.slice(0, i + 1).join('/'))))
      for (const p of paths) out.set(p, (out.get(p) ?? 0) + 1)
    }
    return out
  }, [q, kind])

  const pages = Math.max(1, Math.ceil(matches.length / PAGE_SIZE))
  const page = Math.min(Math.max(1, Number(params.get('page')) || 1), pages)
  const shown = matches.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  // A link to /glossary#key opens the page holding that entry and scrolls to it.
  useEffect(() => {
    const key = decodeURIComponent(hash.slice(1))
    if (!key) return
    const at = matches.findIndex((r) => r.e.key === key)
    if (at < 0) return
    const target = Math.floor(at / PAGE_SIZE) + 1
    if (target !== page) set({ page: String(target) })
    else document.getElementById(key)?.scrollIntoView({ block: 'center' })
  }, [hash, page]) // eslint-disable-line react-hooks/exhaustive-deps

  const groupOf = (r: Row) =>
    by === 'topic'
      ? (categoryTrail(primaryCategory(r.e, c))[c ? c.split('/').length : 0]?.title ?? category(c!)?.title ?? '')
      : /[a-z]/.test(r.head[0] ?? '')
        ? r.head[0].toUpperCase()
        : '#'

  const topLevel = [...counts.keys()]
    .filter((p) => p && !p.includes('/'))
    .sort((a, b) => categoryOrder(a) - categoryOrder(b))
  const open = c?.split('/')[0]
  const children = open ? (category(open)?.children ?? []).filter((n) => counts.has(n.path)).map((n) => n.path) : []

  const pager = (
    <Pager
      page={page}
      pages={pages}
      total={matches.length}
      onPage={(p) => set({ page: p === 1 ? undefined : String(p) })}
    />
  )

  return (
    <PageContainer wide>
      <h1 className="mb-2 font-prose text-3xl font-bold">Glossary</h1>
      <p className="mb-6 text-sm text-muted-foreground">
        {ROWS.length} acronyms, concepts and names used across the notes. Search by term, expansion or definition.
      </p>

      <div className="grid gap-6 lg:grid-cols-[14rem_1fr]">
        <nav aria-label="Topics" className="hidden lg:block">
          <div className="sticky top-20 max-h-[calc(100vh-6rem)] space-y-0.5 overflow-y-auto pr-2 text-sm">
            <RailItem
              label="All topics"
              count={counts.get('')}
              active={!c}
              onClick={() => set({ c: undefined, page: undefined })}
            />
            {topLevel.map((p) => (
              <div key={p}>
                <RailItem
                  label={category(p)?.title ?? p}
                  count={counts.get(p)}
                  active={c === p}
                  onClick={() => set({ c: p, page: undefined })}
                />
                {open === p &&
                  children.map((s) => (
                    <RailItem
                      key={s}
                      label={category(s)?.title ?? s}
                      count={counts.get(s)}
                      active={c === s || !!c?.startsWith(`${s}/`)}
                      onClick={() => set({ c: s, page: undefined })}
                      nested
                    />
                  ))}
              </div>
            ))}
          </div>
        </nav>

        <div className="min-w-0">
          {/* The search, filters and pager stay in view under the site header while the list scrolls. */}
          <div className="sticky top-14 z-10 -mx-4 space-y-3 border-b bg-background/95 px-4 py-3 backdrop-blur lg:mx-0 lg:px-0">
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative min-w-0 flex-1">
                <Search
                  className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  value={text}
                  onChange={(e) => {
                    setText(e.target.value)
                    commitQuery(e.target.value)
                  }}
                  onKeyDown={(e) => e.key === 'Enter' && commitQuery.flush()}
                  placeholder="Search terms, expansions and definitions"
                  className="pr-8 pl-8"
                  aria-label="Search the glossary"
                />
                {text && (
                  <button
                    type="button"
                    onClick={() => {
                      setText('')
                      commitQuery.cancel()
                      set({ q: undefined, page: undefined })
                    }}
                    className="absolute top-1/2 right-2 -translate-y-1/2 rounded-sm p-0.5 text-muted-foreground hover:text-foreground"
                    aria-label="Clear the search"
                    title="Clear"
                  >
                    <X className="size-4" />
                  </button>
                )}
              </div>
              <ToggleGroup
                value={[kind ?? 'all']}
                onValueChange={(v) => v[0] && set({ kind: v[0] === 'all' ? undefined : v[0], page: undefined })}
                variant="outline"
                size="sm"
                spacing={0}
                aria-label="Kind"
              >
                <ToggleGroupItem value="all" className="px-3">
                  All
                </ToggleGroupItem>
                {kinds.map((k) => (
                  <ToggleGroupItem key={k} value={k} className="px-3">
                    {glossaryKindLabels[k]}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
              <ToggleGroup
                value={[by]}
                onValueChange={(v) => v[0] && set({ sort: v[0] === 'topic' ? 'topic' : undefined, page: undefined })}
                variant="outline"
                size="sm"
                spacing={0}
                aria-label="Sort by"
              >
                <ToggleGroupItem value="az" className="px-3">
                  A–Z
                </ToggleGroupItem>
                <ToggleGroupItem value="topic" className="px-3">
                  Topic
                </ToggleGroupItem>
              </ToggleGroup>
            </div>
            {/* Below lg the rail is hidden; the active category shows as a removable chip. */}
            {c && (
              <div className="flex items-center gap-2 text-sm lg:hidden">
                <span className="text-muted-foreground">In</span>
                <Button variant="secondary" size="sm" onClick={() => set({ c: undefined, page: undefined })}>
                  {categoryTrail(c)
                    .map((n) => n.title)
                    .join(' › ')}{' '}
                  ×
                </Button>
              </div>
            )}
            {pager}
          </div>

          <div className="my-3">
            {shown.map((r, i) => {
              const group = groupOf(r)
              return (
                <div key={r.e.key}>
                  {(i === 0 || groupOf(shown[i - 1]) !== group) && (
                    <h2 className="mt-6 mb-2 border-b pb-1 text-sm font-semibold text-muted-foreground">{group}</h2>
                  )}
                  <EntryRow
                    e={r.e}
                    highlighted={hash === `#${r.e.key}`}
                    onCategory={(p) => set({ c: p, page: undefined })}
                  />
                </div>
              )
            })}
            {shown.length === 0 && (
              <p className="py-8 text-center text-sm text-muted-foreground">No glossary entries match.</p>
            )}
          </div>
          {pager}
        </div>
      </div>
    </PageContainer>
  )
}

function RailItem({
  label,
  count,
  active,
  onClick,
  nested = false,
}: {
  label: string
  count?: number
  active: boolean
  onClick: () => void
  nested?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? 'true' : undefined}
      className={cn(
        'flex w-full items-baseline justify-between gap-2 rounded-md px-2 py-1 text-left transition-colors hover:bg-muted',
        nested && 'pl-5 text-xs',
        active ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground',
      )}
    >
      <span className="min-w-0 truncate">{label}</span>
      {count !== undefined && <span className="shrink-0 text-xs tabular-nums">{count}</span>}
    </button>
  )
}

function EntryRow({
  e,
  highlighted,
  onCategory,
}: {
  e: GlossaryEntry
  highlighted: boolean
  onCategory: (path: string) => void
}) {
  const note = e.note ? notesBySlug.get(e.note) : undefined
  const see = e.see.flatMap((s) => notesBySlug.get(s) ?? [])
  const usedIn = glossaryUsage(e.key)
  return (
    <article
      id={e.key}
      className={cn('scroll-mt-40 rounded-md px-3 py-3', highlighted && 'bg-muted')}
      aria-labelledby={`${e.key}-term`}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <h3 id={`${e.key}-term`} className="font-semibold">
          {/* An entry with its own note opens it; the others link to their own anchor. */}
          {note ? (
            <Link
              to={noteUrl(note.slug)}
              title={`Open the note: ${note.title}`}
              className="group inline-flex items-center gap-1 text-primary hover:underline"
            >
              <MathText text={glossHeadword(e)} />
              <FileText className="size-3.5 shrink-0 opacity-70 group-hover:opacity-100" aria-label="has a note" />
            </Link>
          ) : (
            <a href={`#${e.key}`} className="hover:underline">
              <MathText text={glossHeadword(e)} />
            </a>
          )}
        </h3>
        {e.short && (
          <span className="text-sm text-muted-foreground">
            <MathText text={e.long} />
          </span>
        )}
        <Badge variant="secondary" className="text-[10px]">
          {glossaryKindLabels[e.kind]}
        </Badge>
        {e.sense && (
          <Badge variant="outline" className="text-[10px]">
            {e.sense}
          </Badge>
        )}
      </div>
      <p className="mt-1 font-prose text-[15px] leading-relaxed">
        <MathText text={e.definition} />
      </p>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {note ? (
          <Link
            to={noteUrl(note.slug)}
            className="inline-flex items-center gap-1 font-medium text-primary hover:underline"
          >
            <FileText className="size-3" aria-hidden />
            Read the note: {note.title}
            <ArrowRight className="size-3" aria-hidden />
          </Link>
        ) : (
          see.length === 0 && <span className="italic">No note yet</span>
        )}
        {see.length > 0 && (
          <span>
            {note ? 'Also in ' : 'Discussed in '}
            {see.map((n, i) => (
              <span key={n.slug}>
                {i > 0 && ', '}
                <Link to={noteUrl(n.slug)} className="text-foreground hover:underline">
                  {n.title}
                </Link>
              </span>
            ))}
          </span>
        )}
        {e.category.map((p) => (
          <button key={p} type="button" onClick={() => onCategory(p)} className="hover:text-foreground hover:underline">
            {categoryTrail(p)
              .map((n) => n.title)
              .join(' › ')}
          </button>
        ))}
        {usedIn.length > 0 && (
          <span>
            Used in{' '}
            {usedIn.slice(0, 3).map((n, i) => (
              <span key={n.slug}>
                {i > 0 && ', '}
                <Link to={noteUrl(n.slug)} className="hover:underline">
                  {n.title}
                </Link>
              </span>
            ))}
            {usedIn.length > 3 && ` +${usedIn.length - 3} more`}
          </span>
        )}
        <span className="font-mono text-[11px]" title="Names that work in <Gloss name>">
          {[e.key, ...e.aliases].join(' · ')}
        </span>
      </div>
    </article>
  )
}

/** Page numbers with the current page, its neighbours, the ends and ellipses between. */
function pageList(page: number, pages: number): (number | '…')[] {
  const keep = new Set([1, pages, page - 1, page, page + 1].filter((p) => p >= 1 && p <= pages))
  const out: (number | '…')[] = []
  ;[...keep]
    .sort((a, b) => a - b)
    .forEach((p, i, all) => {
      if (i > 0 && p - all[i - 1] > 1) out.push('…')
      out.push(p)
    })
  return out
}

function Pager({
  page,
  pages,
  total,
  onPage,
}: {
  page: number
  pages: number
  total: number
  onPage: (p: number) => void
}) {
  const first = total ? (page - 1) * PAGE_SIZE + 1 : 0
  const last = Math.min(page * PAGE_SIZE, total)
  return (
    <nav aria-label="Pages" className="flex flex-wrap items-center justify-between gap-2 text-sm">
      <span className="text-muted-foreground tabular-nums">
        {first}–{last} of {total}
      </span>
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon-sm"
          disabled={page <= 1}
          onClick={() => onPage(page - 1)}
          aria-label="Previous page"
        >
          <ChevronLeft />
        </Button>
        {pageList(page, pages).map((p, i) =>
          p === '…' ? (
            <span key={`gap-${i}`} className="px-1 text-muted-foreground">
              …
            </span>
          ) : (
            <Button
              key={p}
              variant={p === page ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => onPage(p)}
              aria-current={p === page ? 'page' : undefined}
              className="min-w-8 tabular-nums"
            >
              {p}
            </Button>
          ),
        )}
        <Button
          variant="ghost"
          size="icon-sm"
          disabled={page >= pages}
          onClick={() => onPage(page + 1)}
          aria-label="Next page"
        >
          <ChevronRight />
        </Button>
      </div>
    </nav>
  )
}
