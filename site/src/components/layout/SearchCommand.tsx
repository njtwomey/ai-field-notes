import { FolderOpen, Hash, Search } from 'lucide-react'
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
import { browseUrl, category, kindLabels, noteUrl, taxonomy, type NoteMeta } from '@/lib/content'
import { parseQuery, search, suggestTags, tagCounts } from '@/lib/search'
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
 * autocomplete while it is typed. The empty palette offers popular tags and the topics rather than every note.
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

  const parsed = parseQuery(query)
  const results = useMemo(() => search(query), [query])
  const suggestions = parsed.partialTag !== undefined ? suggestTags(parsed.partialTag) : []

  const go = (to: string) => {
    setOpen(false)
    setQuery('')
    navigate(to)
  }

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
            placeholder="Search titles, summaries and tags · #tag to filter by tag"
            value={query}
            onValueChange={setQuery}
          />
          <CommandList>
            <CommandEmpty>No matching notes.</CommandEmpty>
            {suggestions.length > 0 && (
              <CommandGroup heading="Tags">
                {suggestions.map(({ tag, count }) => (
                  <TagItem key={tag} tag={tag} count={count} onSelect={addTag} />
                ))}
              </CommandGroup>
            )}
            {query.trim() ? (
              results.length > 0 && (
                <CommandGroup
                  heading={parsed.tags.length ? `Notes tagged ${parsed.tags.map((t) => `#${t}`).join(' ')}` : 'Notes'}
                >
                  {results.map((n) => (
                    <ResultItem key={n.slug} note={n} onSelect={(slug) => go(noteUrl(slug))} />
                  ))}
                </CommandGroup>
              )
            ) : (
              <>
                <CommandGroup heading="Popular tags">
                  {tagCounts.slice(0, 12).map(({ tag, count }) => (
                    <TagItem key={tag} tag={tag} count={count} onSelect={addTag} />
                  ))}
                </CommandGroup>
                <CommandGroup heading="Topics">
                  {taxonomy.map((t) => (
                    <CommandItem key={t.path} value={`topic-${t.path}`} onSelect={() => go(browseUrl({ c: t.path }))}>
                      <FolderOpen className="size-4 text-muted-foreground" aria-hidden />
                      {t.title}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}
          </CommandList>
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
    <CommandItem value={note.slug} onSelect={() => onSelect(note.slug)} className="items-start gap-3 py-2">
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
