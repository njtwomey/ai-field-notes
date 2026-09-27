import { FolderTree, Hash, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
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
import { category, kindLabels, noteUrl, prefetchNote, type NoteMeta } from '@/lib/content'
import { loadBodyIndex, parseQuery, search, suggestPaths, suggestTags } from '@/lib/search'
import { kindIcons } from './kind-icon'
import { MathText } from '@/components/content/MathText'

const OPEN_EVENT = 'mlc:open-search'

/** Open the search palette from anywhere, e.g. a search box on the landing page. */
// eslint-disable-next-line react/only-export-components
export function openSearch() {
  window.dispatchEvent(new Event(OPEN_EVENT))
}

/**
 * ⌘K / Ctrl+K palette. Words search titles, aliases, summaries, tags and headings; `#tag` filters by a tag, with
 * autocomplete while it is typed. The palette lists nothing until the reader types.
 */
export function SearchCommand() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((o) => !o)
      } else if (e.key === '/' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault()
        setOpen(true)
      }
    }
    const onOpen = () => setOpen(true)
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
  const results = useMemo(() => search(debounced, { fullText: bodyReady }), [debounced, bodyReady])
  // A `#…` being typed at the end of the query switches the list to tag completion.
  const tagging = parsed.partialTag !== undefined
  const suggestions = tagging ? suggestTags(parsed.partialTag!) : []
  // Likewise a trailing `/…` switches it to taxonomy-path completion.
  const pathing = parsed.partialPath !== undefined
  const paths = pathing ? suggestPaths(parsed.partialPath!) : []
  // The top result is the likeliest choice; start loading it while the reader decides.
  useEffect(() => {
    if (results[0]) prefetchNote(results[0].slug)
  }, [results])

  const go = (to: string) => {
    setOpen(false)
    setQuery('')
    navigate(to)
  }

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
        onClick={() => setOpen(true)}
        className="w-full justify-start gap-2 text-muted-foreground sm:w-64"
      >
        <Search className="size-4" aria-hidden />
        <span className="flex-1 text-left">Search concepts…</span>
        <Kbd>⌘K</Kbd>
      </Button>
      <CommandDialog
        open={open}
        onOpenChange={setOpen}
        title="Search"
        description="Search all notes"
        className="sm:max-w-2xl"
      >
        <Command shouldFilter={false}>
          <CommandInput
            autoFocus
            placeholder="Search notes · # for tags · / for topics"
            value={query}
            onValueChange={setQuery}
            onKeyDown={(e) => {
              // Tab completes the tag being typed with the top suggestion.
              if (e.key === 'Tab' && tagging && suggestions[0]) {
                e.preventDefault()
                addTag(suggestions[0].tag)
              } else if (e.key === 'Tab' && pathing && paths[0]) {
                e.preventDefault()
                addPath(paths[0].path, paths[0].hasChildren)
              }
            }}
          />
          {/* Nothing is listed until the reader types; a trailing #… shows only tag completions. */}
          {query.trim() !== '' && (
            <CommandList>
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
              ) : tagging ? (
                suggestions.length > 0 ? (
                  <CommandGroup heading="Tags · Enter or Tab to complete">
                    {suggestions.map(({ tag, count }) => (
                      <TagItem key={tag} tag={tag} count={count} onSelect={addTag} />
                    ))}
                  </CommandGroup>
                ) : (
                  <div className="py-6 text-center text-sm text-muted-foreground">No tag starts with that.</div>
                )
              ) : results.length > 0 ? (
                <CommandGroup
                  heading={[
                    parsed.path && `in /${parsed.path}`,
                    parsed.tags.length && `tagged ${parsed.tags.map((t) => `#${t}`).join(' ')}`,
                  ]
                    .filter(Boolean)
                    .join(', ')
                    .replace(/^/, 'Notes ')
                    .trim()}
                >
                  {results.map((n) => (
                    <ResultItem key={n.slug} note={n} onSelect={(slug) => go(noteUrl(slug))} />
                  ))}
                </CommandGroup>
              ) : (
                <CommandEmpty>No matching notes.</CommandEmpty>
              )}
            </CommandList>
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
        <p className="line-clamp-1 text-xs text-muted-foreground">
          <MathText text={note.summary} />
        </p>
      </div>
    </CommandItem>
  )
}
