import { BookOpen, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import {
  categoryTrail,
  indexFirst,
  notesInCategory,
  noteUrl,
  taxonomy,
  type CategoryNode,
  type NoteMeta,
} from '@/lib/content'
import { cn } from '@/lib/utils'

/**
 * The taxonomy around the current note, for the left column: at each level the siblings of the note's path, with
 * only the path expanded, down to the notes of the note's own category. Opening another branch closes its siblings
 * (one open per level), so the panel stays short. The panel is collapsed on every note, including after navigating
 * from it: it is a page selector, opened when wanted. When it opens, only the current path is expanded.
 */
export function NoteNav({ note }: { note: NoteMeta }) {
  // Open for one note only: arriving at another note (from this panel or anywhere) finds it collapsed.
  const [openFor, setOpenFor] = useState<string | null>(null)
  const open = openFor === note.slug
  const path = categoryTrail(note.category).map((c) => c.path)
  return (
    <Collapsible open={open} onOpenChange={(o) => setOpenFor(o ? note.slug : null)} className="mb-6 text-sm">
      <CollapsibleTrigger className="group mb-1 flex w-full items-center gap-1 rounded-md text-left text-xs font-medium tracking-wide text-muted-foreground uppercase outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50">
        <ChevronRight
          className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90"
          aria-hidden
        />
        Browse
      </CollapsibleTrigger>
      <CollapsibleContent>
        {/* Keyed by note so that navigating resets every level to the new path. */}
        <nav aria-label="Browse topics" key={note.slug}>
          <Level categories={taxonomy} notes={[]} path={path} depth={0} current={note.slug} />
        </nav>
      </CollapsibleContent>
    </Collapsible>
  )
}

/** One level of the tree: sibling categories (one open at a time) followed by the parent category's own notes. */
function Level({
  categories,
  notes,
  path,
  depth,
  current,
}: {
  categories: CategoryNode[]
  notes: NoteMeta[]
  /** Category paths from the part down to the current note's category. */
  path: string[]
  depth: number
  current: string
}) {
  const [open, setOpen] = useState<string | null>(path[depth] ?? null)
  return (
    <ul className={cn(depth > 0 && 'ml-[7px] border-l border-border pl-1.5')}>
      {categories.map((c) => (
        <Branch
          key={c.path}
          node={c}
          open={open === c.path}
          onOpenChange={(o) => setOpen(o ? c.path : null)}
          onPath={path[depth] === c.path}
          path={path}
          depth={depth}
          current={current}
        />
      ))}
      {notes.map((n) => (
        <NoteItem key={n.slug} note={n} current={n.slug === current} />
      ))}
    </ul>
  )
}

function Branch({
  node,
  open,
  onOpenChange,
  onPath,
  path,
  depth,
  current,
}: {
  node: CategoryNode
  open: boolean
  onOpenChange: (open: boolean) => void
  onPath: boolean
  path: string[]
  depth: number
  current: string
}) {
  const count = notesInCategory(node.path).length
  const empty = count === 0 && node.children.length === 0
  return (
    <li>
      <Collapsible open={open} onOpenChange={onOpenChange}>
        {/* A page selector: the whole row expands the category in place (no navigation); only notes are links. */}
        <CollapsibleTrigger
          disabled={empty}
          aria-label={`${open ? 'Collapse' : 'Expand'} ${node.title}`}
          className={cn(
            'group flex w-full items-start gap-0.5 rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
            !empty && 'hover:bg-muted',
          )}
        >
          <span className="mt-[3px] grid size-4 shrink-0 place-items-center text-muted-foreground" aria-hidden>
            {!empty && <ChevronRight className="size-3.5 transition-transform group-data-[panel-open]:rotate-90" />}
          </span>
          <span
            className={cn(
              'flex min-w-0 flex-1 items-baseline gap-2 px-1 py-0.5 leading-snug',
              onPath ? 'font-medium text-foreground' : 'text-muted-foreground group-hover:text-foreground',
              count === 0 && 'text-muted-foreground/60',
            )}
          >
            <span className="min-w-0 flex-1">
              {depth === 0 && node.roman && (
                <span className="mr-1 text-xs font-normal text-muted-foreground tabular-nums">{node.roman}</span>
              )}
              {node.title}
            </span>
            <span className="text-xs font-normal text-muted-foreground tabular-nums">{count}</span>
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          {open && (
            <Level
              categories={node.children}
              notes={categoryNotes(node.path)}
              path={onPath ? path : []}
              depth={depth + 1}
              current={current}
            />
          )}
        </CollapsibleContent>
      </Collapsible>
    </li>
  )
}

/** A category's own notes, its index note first and the rest alphabetically, as in the breadcrumb's note menu. */
function categoryNotes(path: string): NoteMeta[] {
  return indexFirst(
    notesInCategory(path, false).sort((a, b) => a.title.localeCompare(b.title)),
    path,
  )
}

function NoteItem({ note, current }: { note: NoteMeta; current: boolean }) {
  const lead = categoryTrail(note.category).at(-1)?.index === note.slug
  return (
    <li>
      <Link
        to={noteUrl(note.slug)}
        aria-current={current ? 'page' : undefined}
        title={lead ? 'Start here' : undefined}
        className={cn(
          'flex items-start gap-1 rounded-sm py-0.5 pr-1 pl-[22px] leading-snug outline-none hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50',
          current ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground',
          lead && 'pl-1',
        )}
      >
        {lead && <BookOpen className="mt-[3px] size-3.5 shrink-0" aria-label="Start here" />}
        <span className="min-w-0">{note.title}</span>
      </Link>
    </li>
  )
}
