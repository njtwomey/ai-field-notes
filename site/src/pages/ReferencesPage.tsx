import { ArrowDownWideNarrow, ArrowUpNarrowWide, ChevronLeft, ChevronRight, ExternalLink, Search } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router'
import { PageContainer } from '@/components/layout/AppShell'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { formatAuthors, notes, noteUrl, references, type NoteMeta, type Reference } from '@/lib/content'
import { cn } from '@/lib/utils'

const PAGE_SIZE = 100

type Entry = { key: string; ref: Reference; citing: NoteMeta[]; haystack: string }

/** Lower-case and strip accents, so "krahenbuhl" finds "Krähenbühl". */
const fold = (s: string) =>
  s
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()

/** Every reference with the notes that cite it, built once. */
const ENTRIES: Entry[] = (() => {
  const citing = new Map<string, NoteMeta[]>()
  for (const n of notes)
    for (const key of new Set([...n.cited, ...n.references])) citing.set(key, [...(citing.get(key) ?? []), n])
  return Object.entries(references).map(([key, ref]) => ({
    key,
    ref,
    citing: citing.get(key) ?? [],
    haystack: fold(`${ref.title} ${ref.authors?.join(' ') ?? ''}`),
  }))
})()

type SortKey = 'year' | 'title'

function sorted(entries: Entry[], by: SortKey, descending: boolean): Entry[] {
  const out = [...entries]
  const title = (e: Entry) => fold(e.ref.title)
  out.sort((a, b) =>
    by === 'year'
      ? (a.ref.year ?? 0) - (b.ref.year ?? 0) || title(a).localeCompare(title(b))
      : title(a).localeCompare(title(b)) || (a.ref.year ?? 0) - (b.ref.year ?? 0),
  )
  return descending ? out.reverse() : out
}

/** Every source in content/references.yaml, searchable by title and author, sortable, 100 to a page. */
export function ReferencesPage() {
  const [params, setParams] = useSearchParams()
  const { hash } = useLocation()
  const q = params.get('q') ?? ''
  const by: SortKey = params.get('sort') === 'title' ? 'title' : 'year'
  // Years read newest first by default, titles A to Z.
  const descending = params.has('dir') ? params.get('dir') === 'desc' : by === 'year'
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

  const matches = useMemo(() => {
    const terms = fold(q).split(/\s+/).filter(Boolean)
    const hits = terms.length ? ENTRIES.filter((e) => terms.every((t) => e.haystack.includes(t))) : ENTRIES
    return sorted(hits, by, descending)
  }, [q, by, descending])

  const pages = Math.max(1, Math.ceil(matches.length / PAGE_SIZE))
  const page = Math.min(Math.max(1, Number(params.get('page')) || 1), pages)
  const shown = matches.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  // A link to /references#key opens the page holding that reference and scrolls to it.
  useEffect(() => {
    const key = decodeURIComponent(hash.slice(1))
    if (!key) return
    const at = matches.findIndex((e) => e.key === key)
    if (at < 0) return
    const target = Math.floor(at / PAGE_SIZE) + 1
    if (target !== page) set({ page: String(target) })
    else document.getElementById(key)?.scrollIntoView({ block: 'center' })
  }, [hash, page]) // eslint-disable-line react-hooks/exhaustive-deps

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
      <h1 className="mb-2 font-prose text-3xl font-bold">References</h1>
      <p className="mb-6 text-sm text-muted-foreground">
        {ENTRIES.length} sources cited across the notes. Search by title or author.
      </p>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative min-w-0 flex-1">
          <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input
            value={q}
            onChange={(e) => set({ q: e.target.value, page: undefined })}
            placeholder="Search titles and authors"
            className="pl-8"
            aria-label="Search references"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <ToggleGroup
            value={[by]}
            onValueChange={(v) =>
              v[0] && set({ sort: v[0] === 'title' ? 'title' : undefined, dir: undefined, page: undefined })
            }
            variant="outline"
            size="sm"
            spacing={0}
            aria-label="Sort by"
          >
            <ToggleGroupItem value="year" className="px-3">
              Year
            </ToggleGroupItem>
            <ToggleGroupItem value="title" className="px-3">
              Title
            </ToggleGroupItem>
          </ToggleGroup>
          <Button
            variant="outline"
            size="icon-sm"
            onClick={() => set({ dir: descending ? 'asc' : 'desc', page: undefined })}
            aria-label={descending ? 'Descending; switch to ascending' : 'Ascending; switch to descending'}
            title={descending ? 'Descending' : 'Ascending'}
          >
            {descending ? <ArrowDownWideNarrow /> : <ArrowUpNarrowWide />}
          </Button>
        </div>
      </div>

      {pager}
      <div className="my-3 overflow-x-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">Year</TableHead>
              <TableHead>Title</TableHead>
              <TableHead className="hidden md:table-cell">Authors</TableHead>
              <TableHead className="hidden w-20 sm:table-cell">Type</TableHead>
              <TableHead className="hidden lg:table-cell">Cited in</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((e) => (
              <TableRow key={e.key} id={e.key} className={cn('align-top', hash === `#${e.key}` && 'bg-muted')}>
                <TableCell className="text-muted-foreground tabular-nums">{e.ref.year ?? '—'}</TableCell>
                <TableCell className="max-w-xl whitespace-normal">
                  {e.ref.url ? (
                    <a href={e.ref.url} target="_blank" rel="noreferrer" className="font-medium hover:underline">
                      {e.ref.title}
                      <ExternalLink className="ml-1 inline size-3 text-muted-foreground" aria-hidden />
                    </a>
                  ) : (
                    <span className="font-medium">{e.ref.title}</span>
                  )}
                  {e.ref.venue && <div className="text-xs text-muted-foreground">{e.ref.venue}</div>}
                  <div className="text-xs text-muted-foreground md:hidden">{formatAuthors(e.ref.authors ?? [])}</div>
                </TableCell>
                <TableCell className="hidden max-w-xs text-sm whitespace-normal text-muted-foreground md:table-cell">
                  {formatAuthors(e.ref.authors ?? [])}
                </TableCell>
                <TableCell className="hidden text-xs text-muted-foreground capitalize sm:table-cell">
                  {e.ref.type}
                </TableCell>
                <TableCell className="hidden max-w-xs text-xs whitespace-normal lg:table-cell">
                  {e.citing.slice(0, 3).map((n, i) => (
                    <span key={n.slug}>
                      {i > 0 && ', '}
                      <Link to={noteUrl(n.slug)} className="hover:underline">
                        {n.title}
                      </Link>
                    </span>
                  ))}
                  {e.citing.length > 3 && <span className="text-muted-foreground"> +{e.citing.length - 3} more</span>}
                </TableCell>
              </TableRow>
            ))}
            {shown.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                  No references match.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      {pager}
    </PageContainer>
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
