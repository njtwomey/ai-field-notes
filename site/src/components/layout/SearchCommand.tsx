import { BookA, FileText, FolderTree, Hash, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { useNavigate } from 'react-router'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { Kbd } from '@/components/ui/kbd'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { category, kindLabels, noteUrl, prefetchNote, stats, type GlossaryEntry, type NoteMeta } from '@/lib/content'
import { glossaryUrl, searchGlossary } from '@/lib/glossary'
import { loadBodyIndex, parseQuery, search, suggestPaths, suggestTags } from '@/lib/search'
import { kindIcons } from './kind-icon'
import { MathText } from '@/components/content/MathText'

const OPEN_EVENT = 'mlc:open-search'

/** Open the search palette from anywhere, e.g. a search box on the landing page. */
// eslint-disable-next-line react/only-export-components
export function openSearch() {
  window.dispatchEvent(new Event(OPEN_EVENT))
}

/** Results shown per tab; the tab's pill counts every match. */
const SHOWN = 50

type Tab = 'notes' | 'glossary'

/**
 * ⌘K / Ctrl+K palette with two tabs, notes and glossary, each headed by its number of matches; Tab switches between
 * them. Words search titles, aliases, summaries, tags and headings; `#tag` filters by a tag and `/path` by a topic,
 * with completion while they are typed. With no query the palette shows the site's totals.
 */
export function SearchCommand() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [picked, setPicked] = useState<Tab>()
  const navigate = useNavigate()
  // The palette opens empty every time: a query left from an earlier page is cleared on each open and close.
  const reset = () => {
    setQuery('')
    setPicked(undefined)
  }
  const show = (o: boolean) => {
    reset()
    setOpen(o)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        reset()
        setOpen((o) => !o)
      } else if (e.key === '/' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault()
        show(true)
      }
    }
    const onOpen = () => show(true)
    document.addEventListener('keydown', onKey)
    window.addEventListener(OPEN_EVENT, onOpen)
    return () => {
      document.removeEventListener('keydown', onKey)
      window.removeEventListener(OPEN_EVENT, onOpen)
    }
  }, [])

  // Search runs on a debounced copy of the query; the input itself stays immediate.
  const [debounced, setDebounced] = useState('')
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query), 150)
    return () => clearTimeout(t)
  }, [query])
  // Full-text search becomes available in the background after the palette first opens.
  const [bodyReady, setBodyReady] = useState(false)
  useEffect(() => {
    if (open && !bodyReady) loadBodyIndex().then(() => setBodyReady(true))
  }, [open, bodyReady])
  const parsed = parseQuery(query)
  const results = useMemo(() => search(debounced, { fullText: bodyReady, limit: Infinity }), [debounced, bodyReady])
  // Tags and topic paths filter notes only, so they leave the glossary empty.
  const glossHits = useMemo(
    () => (/(^|\s)[#/]/.test(debounced) ? [] : searchGlossary(debounced, Infinity)),
    [debounced],
  )
  // Until the reader picks a tab, a query lands on the tab that has matches, notes first. A pick holds until closing.
  const tab: Tab = picked ?? (results.length > 0 || glossHits.length === 0 ? 'notes' : 'glossary')
  const setTab = setPicked
  // A `#…` being typed at the end of the query switches the list to tag completion.
  const tagging = parsed.partialTag !== undefined
  const suggestions = tagging ? suggestTags(parsed.partialTag!) : []
  // Likewise a trailing `/…` switches it to taxonomy-path completion.
  const pathing = parsed.partialPath !== undefined
  const paths = pathing ? suggestPaths(parsed.partialPath!) : []
  // The highlighted item is controlled: after a tab switch or a new query it falls back to the tab's first item,
  // so Enter always opens something visible.
  const [selected, setSelected] = useState('')
  const shownValues =
    tab === 'notes'
      ? results.slice(0, SHOWN).map((n) => n.slug)
      : glossHits.slice(0, SHOWN).map((e) => `gloss-${e.key}`)
  const inTabs = !tagging && !pathing
  const active = inTabs && !shownValues.includes(selected) ? (shownValues[0] ?? '') : selected
  // New results scroll the list back to the top, so continuing to type after scrolling shows the best matches.
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    listRef.current?.scrollTo({ top: 0 })
  }, [debounced, tab])
  // The top result is the likeliest choice; start loading it while the reader decides.
  useEffect(() => {
    if (results[0]) prefetchNote(results[0].slug)
  }, [results])

  // cmdk's onSelect carries no event, so the modifier is recorded as the click or Enter passes the root (capture phase).
  const newTab = useRef(false)
  const recordModifier = (e: { metaKey: boolean; ctrlKey: boolean }) => {
    newTab.current = e.metaKey || e.ctrlKey
  }

  const go = (to: string) => {
    if (newTab.current) {
      // ⌘/Ctrl-click or ⌘/Ctrl-Enter: open in a new tab, as a link would, and leave the palette open.
      newTab.current = false
      window.open(import.meta.env.BASE_URL.replace(/\/$/, '') + to, '_blank', 'noopener')
      return
    }
    setOpen(false)
    setQuery('')
    setPicked(undefined)
    navigate(to)
  }
  const switchTab = () => setTab(tab === 'notes' ? 'glossary' : 'notes')

  /** Complete the `/…` being typed: a branch with subtopics keeps completing into them; a leaf ends the token. */
  const addPath = (p: string, hasChildren: boolean) =>
    setQuery(`${query.replace(/\/\S*$/, '')}/${p}${hasChildren ? '/' : ' '}`)

  /** Put a tag into the query: replace the `#…` being typed, or append. */
  const addTag = (tag: string) => {
    const base =
      parsed.partialTag !== undefined ? query.replace(/#\S*$/, '') : query.replace(/\s*$/, query.trim() ? ' ' : '')
    setQuery(`${base}#${tag} `)
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => show(true)}
        className="w-full justify-start gap-2 text-muted-foreground sm:w-64"
      >
        <Search className="size-4" aria-hidden />
        <span className="flex-1 text-left">Search concepts…</span>
        <Kbd>⌘K</Kbd>
      </Button>
      <CommandDialog
        open={open}
        onOpenChange={show}
        title="Search"
        description="Search all notes"
        className="sm:max-w-3xl"
      >
        <Command
          shouldFilter={false}
          value={active}
          onValueChange={setSelected}
          onClickCapture={recordModifier}
          onKeyDownCapture={(e) => e.key === 'Enter' && recordModifier(e)}
        >
          <CommandInput
            autoFocus
            placeholder="Search notes · # for tags · / for topics"
            value={query}
            onValueChange={(v) => {
              setQuery(v)
              // A changed query starts again from the top result.
              setSelected('')
            }}
            onKeyDown={(e) => {
              // Tab completes the tag being typed with the top suggestion.
              if (e.key === 'Tab' && tagging && suggestions[0]) {
                e.preventDefault()
                addTag(suggestions[0].tag)
              } else if (e.key === 'Tab' && pathing && paths[0]) {
                e.preventDefault()
                addPath(paths[0].path, paths[0].hasChildren)
              } else if (e.key === 'Tab' && !tagging && !pathing && query.trim()) {
                e.preventDefault()
                switchTab()
              }
            }}
          />
          {query.trim() === '' ? (
            <SiteStats />
          ) : pathing || tagging ? (
            <CommandList ref={listRef} className="max-h-[min(60vh,32rem)]">
              {pathing ? (
                paths.length > 0 ? (
                  <CommandGroup heading="Topics · Enter or Tab to complete, space to finish">
                    {paths.map((p) => (
                      <CommandItem
                        key={p.path}
                        value={`path-${p.path}`}
                        onSelect={() => addPath(p.path, p.hasChildren)}
                      >
                        <FolderTree className="size-4 text-muted-foreground" aria-hidden />
                        <span className="min-w-0 flex-1 truncate">{p.trail}</span>
                        <span className="font-mono text-[11px] text-muted-foreground">/{p.path}</span>
                        <span className="ml-2 text-xs text-muted-foreground tabular-nums">{p.count}</span>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ) : (
                  <div className="py-6 text-center text-sm text-muted-foreground">No topic matches that path.</div>
                )
              ) : suggestions.length > 0 ? (
                <CommandGroup heading="Tags · Enter or Tab to complete">
                  {suggestions.map(({ tag, count }) => (
                    <TagItem key={tag} tag={tag} count={count} onSelect={addTag} />
                  ))}
                </CommandGroup>
              ) : (
                <div className="py-6 text-center text-sm text-muted-foreground">No tag starts with that.</div>
              )}
            </CommandList>
          ) : (
            <>
              <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="border-b px-3 py-2">
                <div className="flex items-center gap-3">
                  <TabsList>
                    <TabsTrigger value="notes" className="gap-2 px-3" onMouseDown={keepFocus}>
                      Notes
                      <CountPill n={results.length} />
                    </TabsTrigger>
                    <TabsTrigger value="glossary" className="gap-2 px-3" onMouseDown={keepFocus}>
                      Glossary
                      <CountPill n={glossHits.length} />
                    </TabsTrigger>
                  </TabsList>
                  <span className="ml-auto hidden text-[11px] text-muted-foreground sm:inline">
                    <Kbd>Tab</Kbd> switches
                  </span>
                </div>
              </Tabs>
              <CommandList ref={listRef} key={tab} className="max-h-[min(60vh,32rem)]">
                {tab === 'notes' ? (
                  results.length > 0 ? (
                    <CommandGroup
                      heading={[
                        parsed.path && `in /${parsed.path}`,
                        parsed.tags.length && `tagged ${parsed.tags.map((t) => `#${t}`).join(' ')}`,
                      ]
                        .filter(Boolean)
                        .join(', ')}
                    >
                      {results.slice(0, SHOWN).map((n) => (
                        <ResultItem key={n.slug} note={n} onSelect={(slug) => go(noteUrl(slug))} />
                      ))}
                      {results.length > SHOWN && <MoreHint n={results.length - SHOWN} />}
                    </CommandGroup>
                  ) : (
                    <CommandEmpty>No matching notes.</CommandEmpty>
                  )
                ) : glossHits.length > 0 ? (
                  <CommandGroup>
                    {glossHits.slice(0, SHOWN).map((e) => (
                      <GlossItem key={e.key} entry={e} onSelect={go} />
                    ))}
                    {glossHits.length > SHOWN && <MoreHint n={glossHits.length - SHOWN} />}
                  </CommandGroup>
                ) : (
                  <CommandEmpty>No matching glossary entries.</CommandEmpty>
                )}
              </CommandList>
            </>
          )}
        </Command>
      </CommandDialog>
    </>
  )
}

function TagItem({ tag, count, onSelect }: { tag: string; count: number; onSelect: (tag: string) => void }) {
  return (
    <CommandItem value={`tag-${tag}`} onSelect={() => onSelect(tag)}>
      <Hash className="size-4 text-muted-foreground" aria-hidden />
      {tag}
      <span className="ml-auto text-xs text-muted-foreground tabular-nums">{count}</span>
    </CommandItem>
  )
}

function ResultItem({ note, onSelect }: { note: NoteMeta; onSelect: (slug: string) => void }) {
  const Icon = kindIcons[note.kind]
  return (
    <CommandItem
      value={note.slug}
      onSelect={() => onSelect(note.slug)}
      onMouseEnter={() => prefetchNote(note.slug)}
      className="items-start gap-3 py-2"
    >
      <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate font-medium">{note.title}</span>
          <span className="shrink-0 text-[11px] text-muted-foreground">
            {kindLabels[note.kind]} · {category(note.category)?.title}
          </span>
        </div>
        <p className="line-clamp-1 max-h-[1.25rem] overflow-hidden text-xs text-muted-foreground">
          <MathText text={note.summary} />
        </p>
      </div>
    </CommandItem>
  )
}

/** Clicking a tab leaves the focus in the search input, so typing carries on. */
const keepFocus = (e: MouseEvent) => e.preventDefault()

function CountPill({ n }: { n: number }) {
  return (
    <span className="rounded-full bg-muted-foreground/15 px-1.5 py-px text-[10px] leading-4 font-medium tabular-nums">
      {n}
    </span>
  )
}

function MoreHint({ n }: { n: number }) {
  return <div className="px-2 py-2 text-xs text-muted-foreground">{n} more; refine the search to narrow them.</div>
}

/** A glossary match. An entry with a note of its own opens the note; the others open their glossary entry. */
function GlossItem({ entry: e, onSelect }: { entry: GlossaryEntry; onSelect: (to: string) => void }) {
  return (
    <CommandItem
      value={`gloss-${e.key}`}
      onSelect={() => onSelect(e.note ? noteUrl(e.note) : glossaryUrl(e.key))}
      onMouseEnter={() => e.note && prefetchNote(e.note)}
      className="items-start gap-3 py-2"
    >
      <BookA className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="flex min-w-0 items-baseline gap-2">
            <span className="font-medium">
              <MathText text={e.short ?? e.long} />
            </span>
            {e.short && (
              <span className="truncate text-xs text-muted-foreground">
                <MathText text={e.long} />
              </span>
            )}
          </span>
          <span className="inline-flex shrink-0 items-center gap-1 text-[11px] text-muted-foreground">
            {e.note ? (
              <>
                <FileText className="size-3" aria-hidden />
                Opens the note
              </>
            ) : (
              'Glossary entry'
            )}
          </span>
        </div>
        <p className="line-clamp-1 max-h-[1.25rem] overflow-hidden text-xs text-muted-foreground">
          <MathText text={e.definition} />
        </p>
      </div>
    </CommandItem>
  )
}

/** The palette before a query: the site's totals, from the build, and how to search. */
function SiteStats() {
  const items: [number, string][] = [
    [stats.notes, 'notes'],
    [stats.topics, 'topics'],
    [stats.categories, 'categories'],
    [stats.glossary, 'glossary entries'],
    [stats.references, 'references'],
    [stats.tags, 'tags'],
    [stats.workedExamples, 'with worked examples'],
    [stats.runnable, 'with runnable code'],
  ]
  return (
    <div className="space-y-4 px-4 py-5">
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {items.map(([n, label]) => (
          <div key={label} className="rounded-md border px-3 py-2">
            <dt className="text-[11px] text-muted-foreground">{label}</dt>
            <dd className="text-lg font-semibold tabular-nums">{n.toLocaleString()}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-muted-foreground">
        Type to search notes and the glossary. <Kbd>#</Kbd> filters by tag, <Kbd>/</Kbd> by topic, <Kbd>Tab</Kbd>{' '}
        switches between the two tabs.
      </p>
    </div>
  )
}
