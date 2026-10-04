import { ChevronRight, FlaskConical, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { Button } from '@lab/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@lab/ui/collapsible'
import { Input } from '@lab/ui/input'
import { Kbd } from '@lab/ui/kbd'
import { cn } from '@lab/lib/utils'
import type { Specimen } from '../specimen'
import { FAMILIES, LAB_FAMILY, OTHER_FAMILY, TAXONOMY, unmappedModules } from './families'
import { hrefOf } from './paths'
import { ThemeToggle } from './ThemeToggle'

export type PlaceholderPage = {
  key: string
  title: string
  description: string
  part: { id: string; roman: string; title: string }
  subject: { id: string; title: string }
  topic: { id: string; title: string; items: readonly string[] }
}

/** A page of the lab itself (not a specimen), listed under the Lab family, e.g. the UI kit. */
export type LabPage = { key: string; title: string; description: string; render: () => ReactNode }

/** One page in the index: a lab page, a specimen, or a placeholder. */
export type Entry =
  | { kind: 'page'; key: string; module: ''; family: string; title: string; search: string; page: LabPage }
  | { kind: 'specimen'; key: string; module: string; family: string; title: string; search: string; specimen: Specimen }
  | {
      kind: 'placeholder'
      key: string
      module: string
      family: string
      title: string
      search: string
      placeholder: PlaceholderPage
    }

export type TreeNode = {
  id: string
  label: string
  icon?: typeof FlaskConical
  count: number
  isBranch: boolean
  isPlaceholder?: boolean
  children?: TreeNode[]
  entry?: Entry
}

/** Builds the recursive tree of Parts → Subjects → Topics → Specimens. */
function buildTree(entries: readonly Entry[]): TreeNode[] {
  const byFamily = new Map<string, Entry[]>()
  for (const e of entries) {
    byFamily.set(e.family, [...(byFamily.get(e.family) ?? []), e])
  }

  const nodes: TreeNode[] = []

  // 1. Master Taxonomy Parts
  for (const part of TAXONOMY) {
    const partEntries = byFamily.get(part.id) ?? []
    const fam = FAMILIES.find((f) => f.id === part.id)!
    const subjectNodes: TreeNode[] = []

    for (const subj of part.subjects) {
      const subjPrefix = `${part.id}/${subj.id}`
      const subjEntries = partEntries.filter((e) => e.module === subjPrefix || e.module.startsWith(`${subjPrefix}/`))

      if (subj.topics.length === 1 && subj.topics[0].id === 'overview') {
        // Direct topic subject
        const leaves: TreeNode[] = subjEntries
          .map((e) => ({
            id: e.key,
            label: e.title,
            count: e.kind === 'specimen' ? 1 : 0,
            isBranch: false,
            isPlaceholder: e.kind === 'placeholder',
            entry: e,
          }))
          .sort((a, b) => {
            if (a.isPlaceholder && !b.isPlaceholder) return 1
            if (!a.isPlaceholder && b.isPlaceholder) return -1
            return a.label.localeCompare(b.label)
          })

        if (leaves.length > 0) {
          const specCount = leaves.filter((l) => l.entry?.kind === 'specimen').length
          subjectNodes.push({
            id: subjPrefix,
            label: subj.title,
            count: specCount,
            isBranch: true,
            isPlaceholder: specCount === 0,
            children: leaves,
          })
        }
      } else {
        // Multi-topic subject
        const topicNodes: TreeNode[] = []

        for (const topic of subj.topics) {
          const topicPrefix = `${subjPrefix}/${topic.id}`
          const topicEntries = subjEntries.filter(
            (e) => e.module === topicPrefix || e.module.startsWith(`${topicPrefix}/`),
          )

          const leaves: TreeNode[] = topicEntries
            .map((e) => ({
              id: e.key,
              label: e.title,
              count: e.kind === 'specimen' ? 1 : 0,
              isBranch: false,
              isPlaceholder: e.kind === 'placeholder',
              entry: e,
            }))
            .sort((a, b) => {
              if (a.isPlaceholder && !b.isPlaceholder) return 1
              if (!a.isPlaceholder && b.isPlaceholder) return -1
              return a.label.localeCompare(b.label)
            })

          if (leaves.length > 0) {
            const specCount = leaves.filter((l) => l.entry?.kind === 'specimen').length
            topicNodes.push({
              id: topicPrefix,
              label: topic.title,
              count: specCount,
              isBranch: true,
              isPlaceholder: specCount === 0,
              children: leaves,
            })
          }
        }

        if (topicNodes.length > 0) {
          const specCount = topicNodes.reduce((sum, t) => sum + t.count, 0)
          subjectNodes.push({
            id: subjPrefix,
            label: subj.title,
            count: specCount,
            isBranch: true,
            isPlaceholder: specCount === 0,
            children: topicNodes,
          })
        }
      }
    }

    const partSpecCount = subjectNodes.reduce((sum, s) => sum + s.count, 0)
    if (subjectNodes.length > 0) {
      nodes.push({
        id: part.id,
        label: `${part.roman}: ${part.title}`,
        icon: fam.icon,
        count: partSpecCount,
        isBranch: true,
        children: subjectNodes,
      })
    }
  }

  // 2. Lab Family
  const labEntries = byFamily.get(LAB_FAMILY) ?? []
  if (labEntries.length > 0) {
    const fam = FAMILIES.find((f) => f.id === LAB_FAMILY)!

    const GROUPS: { id: string; label: string; keys: string[] }[] = [
      {
        id: 'lab/overview',
        label: 'Galleries & Showcase',
        keys: ['ui-kit', 'diagrams'],
      },
      {
        id: 'lab/mark-layers',
        label: 'Plot: Mark Layers',
        keys: [
          'ui-kit/curve',
          'ui-kit/points',
          'ui-kit/bars',
          'ui-kit/area',
          'ui-kit/signed-area',
          'ui-kit/segments',
          'ui-kit/vectors',
          'ui-kit/rug',
          'ui-kit/annotation',
          'ui-kit/handle',
        ],
      },
      {
        id: 'lab/field-layers',
        label: 'Plot: Fields & Density',
        keys: [
          'ui-kit/histogram',
          'ui-kit/density',
          'ui-kit/mass',
          'ui-kit/support-band',
          'ui-kit/raster',
          'ui-kit/contours',
        ],
      },
      {
        id: 'lab/compositions',
        label: 'Plot: Grids & Compositions',
        keys: ['ui-kit/plots'],
      },
      {
        id: 'lab/architecture',
        label: 'State & Architecture',
        keys: ['ui-kit/state', 'ui-kit/probes', 'ui-kit/scheduler', 'ui-kit/equations'],
      },
    ]

    const entryByKey = new Map(labEntries.map((e) => [e.key, e]))
    const placed = new Set<string>()

    const groupNodes: TreeNode[] = []
    for (const g of GROUPS) {
      const children: TreeNode[] = []
      for (const k of g.keys) {
        const e = entryByKey.get(k)
        if (e) {
          placed.add(k)
          children.push({
            id: e.key,
            label: e.title,
            count: 1,
            isBranch: false,
            entry: e,
          })
        }
      }
      if (children.length > 0) {
        groupNodes.push({
          id: g.id,
          label: g.label,
          count: children.length,
          isBranch: true,
          children,
        })
      }
    }

    const remaining = labEntries.filter((e) => !placed.has(e.key))
    if (remaining.length > 0) {
      groupNodes.push({
        id: 'lab/other',
        label: 'Other Components',
        count: remaining.length,
        isBranch: true,
        children: remaining.map((e) => ({
          id: e.key,
          label: e.title,
          count: 1,
          isBranch: false,
          entry: e,
        })),
      })
    }

    nodes.push({
      id: LAB_FAMILY,
      label: fam.title,
      icon: fam.icon,
      count: labEntries.length,
      isBranch: true,
      children: groupNodes,
    })
  }

  // 3. Other Family (for unmapped nodes if any)
  const otherEntries = byFamily.get(OTHER_FAMILY) ?? []
  if (otherEntries.length > 0) {
    const fam = FAMILIES.find((f) => f.id === OTHER_FAMILY)!
    nodes.push({
      id: OTHER_FAMILY,
      label: fam.title,
      icon: fam.icon,
      count: otherEntries.length,
      isBranch: true,
      children: otherEntries.map((e) => ({
        id: e.key,
        label: e.title,
        count: 1,
        isBranch: false,
        entry: e,
      })),
    })
  }

  return nodes
}

let warned = false
function warnUnmapped(entries: readonly Entry[]) {
  if (warned || !import.meta.env.DEV) return
  warned = true
  const missing = unmappedModules(entries.filter((e) => e.module).map((e) => e.module))
  if (missing.length)
    console.warn(`aifn lab: modules missing from layout/families.ts, listed under Other: ${missing.join(', ')}`)
}

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

function findAncestors(tree: readonly TreeNode[], targetKey: string): string[] {
  const result: string[] = []
  function dfs(nodes: readonly TreeNode[], path: string[]): boolean {
    for (const node of nodes) {
      if (!node.isBranch && node.entry?.key === targetKey) {
        result.push(...path)
        return true
      }
      if (node.isBranch && node.children) {
        if (dfs(node.children, [...path, node.id])) {
          return true
        }
      }
    }
    return false
  }
  dfs(tree, [])
  return result
}

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

  const words = useMemo(() => query.trim().toLowerCase().split(/\s+/).filter(Boolean), [query])
  const searching = words.length > 0

  const filteredEntries = useMemo(() => {
    if (!searching) return entries
    return entries.filter((e) => words.every((w) => e.search.includes(w)))
  }, [entries, words, searching])

  const tree = useMemo(() => buildTree(filteredEntries), [filteredEntries])

  // Track open branch IDs
  const [openBranches, setOpenBranches] = useState<Set<string>>(() => new Set())
  const [lastSelectedKey, setLastSelectedKey] = useState<string>()

  // Auto-expand all ancestors when selection changes
  if (current && current.key !== lastSelectedKey) {
    setLastSelectedKey(current.key)
    const ancestors = findAncestors(tree, current.key)
    if (ancestors.length > 0) {
      setOpenBranches((prev) => new Set([...prev, ...ancestors]))
    }
  }

  const toggleBranch = (id: string) => {
    if (searching) return
    setOpenBranches((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const isBranchOpen = (id: string) => searching || openBranches.has(id)

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
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              rows()[0]?.focus()
            }
          }}
          placeholder="Search taxonomy and specimens"
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
        {tree.length === 0 && <p className="px-2 text-xs text-muted-foreground">Nothing matches “{query}”.</p>}
        <ul className="flex flex-col gap-0.5">
          {tree.map((node) => (
            <TreeNodeRow
              key={node.id}
              node={node}
              parentId=""
              depth={0}
              current={current}
              words={words}
              isOpen={isBranchOpen}
              onToggle={toggleBranch}
              onSelect={onSelect}
            />
          ))}
        </ul>
      </nav>
    </div>
  )
}

function TreeNodeRow({
  node,
  parentId,
  depth,
  current,
  words,
  isOpen,
  onToggle,
  onSelect,
}: {
  node: TreeNode
  parentId: string
  depth: number
  current?: Entry
  words: readonly string[]
  isOpen: (id: string) => boolean
  onToggle: (id: string) => void
  onSelect: (key: string) => void
}) {
  if (!node.isBranch && node.entry) {
    return <Leaf entry={node.entry} parent={parentId} current={current} words={words} onSelect={onSelect} />
  }

  const open = isOpen(node.id)
  const Icon = node.icon

  return (
    <li>
      <Collapsible open={open} onOpenChange={() => onToggle(node.id)}>
        <CollapsibleTrigger
          data-row={`b:${node.id}`}
          data-parent={parentId}
          data-branch=""
          className={cn(
            'group/row flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left outline-none hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring',
            depth === 0
              ? 'text-[13px] font-medium text-sidebar-foreground'
              : depth === 1
                ? 'text-[12.5px] font-medium text-sidebar-foreground/90'
                : 'text-[12px] font-normal text-sidebar-foreground/75',
          )}
        >
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground transition-transform group-aria-expanded/row:rotate-90" />
          {Icon && <Icon className="size-4 shrink-0 text-muted-foreground" />}
          <span className="truncate">{highlight(node.label, words)}</span>
          <Count n={node.count} isPlaceholder={node.isPlaceholder} />
        </CollapsibleTrigger>
        <Panel>
          <ul className="ml-[13px] flex flex-col gap-0.5 border-l border-sidebar-border py-0.5 pl-1.5">
            {node.children?.map((child) => (
              <TreeNodeRow
                key={child.id}
                node={child}
                parentId={`b:${node.id}`}
                depth={depth + 1}
                current={current}
                words={words}
                isOpen={isOpen}
                onToggle={onToggle}
                onSelect={onSelect}
              />
            ))}
          </ul>
        </Panel>
      </Collapsible>
    </li>
  )
}

function Count({ n, isPlaceholder }: { n: number; isPlaceholder?: boolean }) {
  if (isPlaceholder || n === 0) {
    return <span className="ml-auto shrink-0 pl-2 font-mono text-[10px] text-muted-foreground/40 tabular-nums">—</span>
  }
  return <span className="ml-auto shrink-0 pl-2 font-mono text-[11px] text-muted-foreground tabular-nums">{n}</span>
}

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
  const isPlaceholder = entry.kind === 'placeholder'

  return (
    <a
      href={hrefOf(entry.key)}
      data-row={`p:${entry.key}`}
      data-parent={parent}
      data-entry={entry.key}
      title={entry.title}
      onClick={(event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
        event.preventDefault()
        onSelect(entry.key)
      }}
      aria-current={selected ? 'page' : undefined}
      className={cn(
        'flex w-full items-center justify-between truncate rounded-md px-2 py-1 text-left text-[12.5px] outline-none hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 focus-visible:ring-sidebar-ring',
        selected ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground' : 'text-sidebar-foreground/75',
        isPlaceholder && 'text-muted-foreground/75 italic',
      )}
    >
      <span className="truncate">{highlight(entry.title, words)}</span>
      {isPlaceholder && (
        <span className="ml-auto rounded border border-border/40 px-1 text-[9px] tracking-wider text-muted-foreground/50 uppercase">
          plan
        </span>
      )}
    </a>
  )
}
