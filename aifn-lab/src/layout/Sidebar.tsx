import { ChevronRight, FlaskConical, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { Button } from '@lab/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@lab/ui/collapsible'
import { Input } from '@lab/ui/input'
import { Kbd } from '@lab/ui/kbd'
import { cn } from '@lab/lib/utils'
import type { Specimen } from '../specimen'
import { FAMILIES, moduleRank, unmappedModules, type Family } from './families'
import { hrefOf } from './paths'
import { ThemeToggle } from './ThemeToggle'

/** A page of the lab itself (not a specimen), listed under the Lab family, e.g. the UI kit. */
export type LabPage = { key: string; title: string; description: string; render: () => ReactNode }

/** One page in the index: a lab page (`module` is empty) or a specimen. */
export type Entry =
  | { kind: 'page'; key: string; module: ''; family: string; title: string; search: string; page: LabPage }
  | { kind: 'specimen'; key: string; module: string; family: string; title: string; search: string; specimen: Specimen }

type ModuleNode = { module: string; entries: Entry[] }
type FamilyNode = { family: Family; modules: ModuleNode[]; pages: Entry[]; count: number }

/** Groups entries into family → module → page, families in map order, modules in family order. */
function buildTree(entries: readonly Entry[]): FamilyNode[] {
  return FAMILIES.map((family) => {
    const own = entries.filter((e) => e.family === family.id)
    const byModule = new Map<string, Entry[]>()
    for (const e of own) if (e.module) byModule.set(e.module, [...(byModule.get(e.module) ?? []), e])
    const modules = [...byModule]
      .map(([module, list]) => ({ module, entries: list }))
      .sort((a, b) => moduleRank(a.module) - moduleRank(b.module) || a.module.localeCompare(b.module))
    return { family, modules, pages: own.filter((e) => !e.module), count: own.length }
  }).filter((n) => n.count > 0)
}

let warned = false
/** In dev, names modules missing from the family map once (they are listed under "Other"). */
function warnUnmapped(entries: readonly Entry[]) {
  if (warned || !import.meta.env.DEV) return
  warned = true
  const missing = unmappedModules(entries.filter((e) => e.module).map((e) => e.module))
  if (missing.length)
    console.warn(`aifn lab: modules missing from layout/families.ts, listed under Other: ${missing.join(', ')}`)
}

/** Wraps the parts of `text` that match any search word in a highlight. */
function highlight(text: string, words: readonly string[]): ReactNode {
  if (words.length === 0) return text
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const parts = text.split(new RegExp(`(${escaped.join('|')})`, 'gi'))
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="rounded-[2px] bg-primary/15 text-inherit dark:bg-primary/25">
        {part}
      </mark>
    ) : (
      part
    ),
  )
}

type Open = { family?: string; module?: string }

/**
 * The lab's index: a tree of family → module → page with search. Collapsed by default, one family and one module open
 * at a time (the current page's, on load and on navigation). A query filters the tree and expands every match; clearing
 * it restores the accordion. Keys: ↑/↓ move, → or Enter opens or follows, ← collapses or goes to the parent; `/` or
 * ⌘K focuses search.
 */
export function Sidebar({
  entries,
  current,
  onSelect,
  onClose,
}: {
  entries: readonly Entry[]
  current?: Entry
  onSelect: (key: string) => void
  onClose: () => void
}) {
  useEffect(() => warnUnmapped(entries), [entries])
  const [query, setQuery] = useState('')
  const search = useRef<HTMLInputElement>(null)
  const nav = useRef<HTMLElement>(null)

  // "/" (outside a text field) or ⌘K / Ctrl-K focuses the search box.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const slash = e.key === '/' && !target.closest('input, textarea, [contenteditable="true"]')
      const commandK = e.key.toLowerCase() === 'k' && (e.metaKey || e.ctrlKey)
      if (!slash && !commandK) return
      e.preventDefault()
      search.current?.focus()
      search.current?.select()
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [])

  // The accordion follows the current page: reset during render when it changes (no effect, no extra pass).
  const [open, setOpen] = useState<Open>({})
  const [shownKey, setShownKey] = useState<string>()
  if (current && current.key !== shownKey) {
    setShownKey(current.key)
    setOpen({ family: current.family, module: current.module || undefined })
  }

  const words = useMemo(() => query.trim().toLowerCase().split(/\s+/).filter(Boolean), [query])
  const searching = words.length > 0
  const tree = useMemo(
    () => buildTree(searching ? entries.filter((e) => words.every((w) => e.search.includes(w))) : entries),
    [entries, words, searching],
  )
  const count = tree.reduce((n, f) => n + f.count, 0)
  const firstMatch = tree[0] && (tree[0].pages[0] ?? tree[0].modules[0]?.entries[0])

  const familyOpen = (id: string) => searching || open.family === id
  const moduleOpen = (family: string, module: string) => searching || (open.family === family && open.module === module)
  const toggleFamily = (id: string) => {
    if (searching) return
    // Reopening the current page's family reopens its module too.
    const module = current?.family === id ? current.module || undefined : undefined
    setOpen((o) => (o.family === id ? {} : { family: id, module }))
  }
  const toggleModule = (family: string, module: string) => {
    if (searching) return
    setOpen((o) => (o.module === module ? { family } : { family, module }))
  }

  // Keep the current page in view within the sidebar, once its branch has opened.
  const currentKey = current?.key
  useEffect(() => {
    if (!currentKey) return
    const timer = setTimeout(() => {
      const el = nav.current?.querySelector(`[data-entry="${CSS.escape(currentKey)}"]`)
      el?.scrollIntoView({ block: 'nearest' })
    }, 200)
    return () => clearTimeout(timer)
  }, [currentKey, searching])

  const rows = () =>
    [...(nav.current?.querySelectorAll<HTMLElement>('[data-row]') ?? [])].filter(
      (el) => el.offsetParent !== null && !el.closest('[data-ending-style]'),
    )
  const onTreeKey = (e: ReactKeyboardEvent<HTMLElement>) => {
    const list = rows()
    const el = document.activeElement as HTMLElement | null
    const i = el ? list.indexOf(el) : -1
    if (i < 0) return
    const branch = el!.dataset.branch !== undefined
    const expanded = el!.getAttribute('aria-expanded') === 'true'
    const parent = () => list.find((r) => r.dataset.row === el!.dataset.parent)
    let next: HTMLElement | undefined
    if (e.key === 'ArrowDown') next = list[i + 1]
    else if (e.key === 'ArrowUp') next = list[i - 1] ?? search.current ?? undefined
    else if (e.key === 'Home') next = list[0]
    else if (e.key === 'End') next = list.at(-1)
    else if (e.key === 'ArrowRight' && branch) {
      if (expanded) next = list[i + 1]
      else el!.click()
    } else if (e.key === 'ArrowLeft') {
      if (branch && expanded && !searching) el!.click()
      else next = parent()
    } else if (e.key === 'ArrowRight' && !branch) el!.click()
    else return
    e.preventDefault()
    next?.focus()
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <FlaskConical className="size-4 text-muted-foreground" />
        <span className="text-sm font-semibold">aifn lab</span>
        <div className="ml-auto flex items-center gap-1">
          <ThemeToggle />
          <Button variant="ghost" size="icon-sm" aria-label="Close index" className="md:hidden" onClick={onClose}>
            <X />
          </Button>
        </div>
      </div>
      <div className="relative px-4 pb-3">
        <Search className="pointer-events-none absolute top-2 left-6 size-4 text-muted-foreground" />
        <Input
          ref={search}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setQuery('')
            if (e.key === 'Enter' && firstMatch) onSelect(firstMatch.key)
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              rows()[0]?.focus()
            }
          }}
          placeholder="Search modules and pages"
          className="h-8 pr-8 pl-8 text-sm"
          aria-label="Search"
        />
        <Kbd className="pointer-events-none absolute top-1.5 right-6">/</Kbd>
      </div>
      <nav
        ref={nav}
        onKeyDown={onTreeKey}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 pb-4 text-sm"
        aria-label="Specimens"
      >
        {count === 0 && <p className="px-2 text-xs text-muted-foreground">Nothing matches “{query}”.</p>}
        <ul className="flex flex-col gap-0.5">
          {tree.map(({ family, modules, pages, count: n }) => {
            const Icon = family.icon
            const holdsCurrent = current?.family === family.id
            return (
              <li key={family.id}>
                <Collapsible open={familyOpen(family.id)} onOpenChange={() => toggleFamily(family.id)}>
                  <CollapsibleTrigger
                    data-row={`f:${family.id}`}
                    data-branch=""
                    className={cn(
                      'group/row flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[13px] font-medium outline-none hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring',
                      holdsCurrent ? 'text-sidebar-foreground' : 'text-sidebar-foreground/80',
                    )}
                  >
                    <ChevronRight className="size-3.5 shrink-0 text-muted-foreground transition-transform group-aria-expanded/row:rotate-90" />
                    <Icon className="size-4 shrink-0 text-muted-foreground" />
                    <span className="truncate">{highlight(family.title, words)}</span>
                    <Count n={n} />
                  </CollapsibleTrigger>
                  <Panel>
                    <ul className="ml-[15px] flex flex-col gap-0.5 border-l border-sidebar-border py-0.5 pl-1.5">
                      {pages.map((e) => (
                        <li key={e.key}>
                          <Leaf
                            entry={e}
                            parent={`f:${family.id}`}
                            current={current}
                            words={words}
                            onSelect={onSelect}
                          />
                        </li>
                      ))}
                      {modules.map(({ module, entries: list }) => (
                        <li key={module}>
                          <Collapsible
                            open={moduleOpen(family.id, module)}
                            onOpenChange={() => toggleModule(family.id, module)}
                          >
                            <CollapsibleTrigger
                              data-row={`m:${module}`}
                              data-parent={`f:${family.id}`}
                              data-branch=""
                              className={cn(
                                'group/row flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-left font-mono text-[12.5px] outline-none hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring',
                                current?.module === module ? 'text-sidebar-foreground' : 'text-sidebar-foreground/75',
                              )}
                            >
                              <ChevronRight className="size-3 shrink-0 text-muted-foreground transition-transform group-aria-expanded/row:rotate-90" />
                              <span className="truncate">{highlight(module, words)}</span>
                              <Count n={list.length} />
                            </CollapsibleTrigger>
                            <Panel>
                              <ul className="ml-[13px] flex flex-col gap-0.5 border-l border-sidebar-border py-0.5 pl-1.5">
                                {list.map((e) => (
                                  <li key={e.key}>
                                    <Leaf
                                      entry={e}
                                      parent={`m:${module}`}
                                      current={current}
                                      words={words}
                                      onSelect={onSelect}
                                    />
                                  </li>
                                ))}
                              </ul>
                            </Panel>
                          </Collapsible>
                        </li>
                      ))}
                    </ul>
                  </Panel>
                </Collapsible>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

function Count({ n }: { n: number }) {
  return <span className="ml-auto shrink-0 pl-2 font-mono text-[11px] text-muted-foreground tabular-nums">{n}</span>
}

/** A collapsible branch's children, animating their height open and closed. */
function Panel({ children }: { children: ReactNode }) {
  return (
    <CollapsibleContent
      keepMounted
      className="h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-150 ease-out data-ending-style:h-0 data-starting-style:h-0"
    >
      {children}
    </CollapsibleContent>
  )
}

function Leaf({
  entry,
  parent,
  current,
  words,
  onSelect,
}: {
  entry: Entry
  parent: string
  current?: Entry
  words: readonly string[]
  onSelect: (key: string) => void
}) {
  const selected = entry.key === current?.key
  return (
    <a
      href={hrefOf(entry.key)}
      data-row={`p:${entry.key}`}
      data-parent={parent}
      data-entry={entry.key}
      title={entry.title}
      // Client-side navigation for a plain click; modified clicks (new tab, new window) go to the browser.
      onClick={(event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
        event.preventDefault()
        onSelect(entry.key)
      }}
      aria-current={selected ? 'page' : undefined}
      className={cn(
        'block w-full truncate rounded-md px-2 py-1 text-left text-[13px] text-sidebar-foreground/75 outline-none hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 focus-visible:ring-sidebar-ring',
        selected && 'bg-sidebar-accent font-medium text-sidebar-accent-foreground',
      )}
    >
      {highlight(entry.title, words)}
    </a>
  )
}
