import { Search } from 'lucide-react'
import { useEffect, useState } from 'react'
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
import { category, kindLabels, notes, noteUrl, taxonomy, type NoteMeta } from '@/lib/content'
import { search } from '@/lib/search'
import { kindIcons } from './kind-icon'

const OPEN_EVENT = 'mlc:open-search'

/** Open the search palette from anywhere, e.g. a search box on the landing page. */
// eslint-disable-next-line react/only-export-components
export function openSearch() {
  window.dispatchEvent(new Event(OPEN_EVENT))
}

/** ⌘K / Ctrl+K palette. Empty query lists notes by top-level category; typing runs a full-text search. */
export function SearchCommand() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<NoteMeta[]>([])
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

  useEffect(() => {
    let live = true
    if (query.trim()) search(query).then((r) => live && setResults(r))
    else setResults([])
    return () => {
      live = false
    }
  }, [query])

  const go = (slug: string) => {
    setOpen(false)
    setQuery('')
    navigate(noteUrl(slug))
  }

  const grouped = taxonomy.map((root) => ({
    title: root.title,
    notes: notes.filter((n) => n.category === root.path || n.category.startsWith(`${root.path}/`)),
  }))

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
            placeholder="Search titles, tags, headings and text…"
            value={query}
            onValueChange={setQuery}
          />
          <CommandList>
            <CommandEmpty>No matching notes.</CommandEmpty>
            {query.trim() ? (
              <CommandGroup heading="Results">
                {results.map((n) => (
                  <ResultItem key={n.slug} note={n} onSelect={go} />
                ))}
              </CommandGroup>
            ) : (
              grouped
                .filter((g) => g.notes.length)
                .map((g) => (
                  <CommandGroup key={g.title} heading={g.title}>
                    {g.notes.map((n) => (
                      <ResultItem key={n.slug} note={n} onSelect={go} />
                    ))}
                  </CommandGroup>
                ))
            )}
          </CommandList>
        </Command>
      </CommandDialog>
    </>
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
        <p className="line-clamp-1 text-xs text-muted-foreground">{note.summary}</p>
      </div>
    </CommandItem>
  )
}
