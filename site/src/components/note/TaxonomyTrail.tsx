import { BookOpen, Check, CornerDownRight, LayoutGrid } from 'lucide-react'
import { Fragment, useLayoutEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  browseUrl,
  categoryIndexNote,
  indexFirst,
  categoryTrail,
  notesInCategory,
  noteUrl,
  taxonomy,
  type CategoryNode,
  type NoteMeta,
} from '@/lib/content'
import { cn } from '@/lib/utils'

/**
 * How many leading crumbs to fold into the ellipsis so the trail fits on one line. The trail renders in full, then
 * folds one more crumb per layout pass while it overflows, all before paint; a change in the width available to it (its
 * parent's width) unfolds it and measures again. `max` is the most crumbs that may fold.
 */
function useFold(key: string, max: number) {
  const ref = useRef<HTMLElement>(null)
  const [state, setState] = useState({ key, fold: 0 })
  const fold = state.key === key ? state.fold : 0
  if (state.key !== key) setState({ key, fold: 0 })

  useLayoutEffect(() => {
    const el = ref.current
    if (el && fold < max && el.scrollWidth > el.clientWidth + 1) setState({ key, fold: fold + 1 })
  })

  useLayoutEffect(() => {
    const parent = ref.current?.parentElement
    if (!parent) return
    let width = parent.clientWidth
    const observer = new ResizeObserver(() => {
      if (parent.clientWidth === width) return
      width = parent.clientWidth
      setState((s) => ({ ...s, fold: 0 }))
    })
    observer.observe(parent)
    return () => observer.disconnect()
  }, [])

  return { ref, fold }
}

/**
 * A note's place in the taxonomy as a /-joined trail ending in the note itself. Each crumb opens a menu on click: a
 * category lists its siblings, so a reader can move sideways at any level; the note lists the other notes in its
 * category. The whole path shows when it fits; otherwise the outermost categories fold into an ellipsis that opens them
 * as a menu, and on the narrowest screens only the note shows.
 */
export function TaxonomyTrail({ note, className }: { note: NoteMeta; className?: string }) {
  const trail = categoryTrail(note.category)
  const { ref, fold } = useFold(note.slug, trail.length)
  return (
    <nav
      ref={ref}
      aria-label="Breadcrumb"
      className={cn('flex min-w-0 items-center overflow-hidden text-[13px] whitespace-nowrap', className)}
    >
      {fold > 0 && (
        <>
          <FoldedCrumbs nodes={trail.slice(0, fold)} />
          <Slash />
        </>
      )}
      {trail.slice(fold).map((c) => {
        const i = trail.indexOf(c)
        return (
          <Fragment key={c.path}>
            <CategoryCrumb node={c} siblings={i === 0 ? taxonomy : trail[i - 1].children} className="shrink-0" />
            <Slash />
          </Fragment>
        )
      })}
      <NoteCrumb note={note} className={cn(fold < trail.length && 'shrink-0')} />
    </nav>
  )
}

/**
 * A category's place in the taxonomy as the same trail, for the browse page: "All topics", each ancestor, then the
 * category itself in bold. The last crumb also opens the category's subtopics, so the tree can be walked both up and
 * down. Ancestors fold into an ellipsis when the trail does not fit.
 */
export function CategoryTrail({ path, className }: { path: string; className?: string }) {
  const trail = categoryTrail(path)
  const { ref, fold } = useFold(path, Math.max(0, trail.length - 1))
  return (
    <nav
      ref={ref}
      aria-label="Breadcrumb"
      className={cn('flex min-w-0 items-center overflow-hidden text-[13px] whitespace-nowrap', className)}
    >
      <Link to={browseUrl()} className={cn(triggerClass, 'shrink-0 text-muted-foreground')}>
        All topics
      </Link>
      {fold > 0 && (
        <>
          <Slash />
          <FoldedCrumbs nodes={trail.slice(0, fold)} />
        </>
      )}
      {trail.slice(fold).map((c) => {
        const i = trail.indexOf(c)
        const current = i === trail.length - 1
        return (
          <Fragment key={c.path}>
            <Slash />
            <CategoryCrumb
              node={c}
              siblings={i === 0 ? taxonomy : trail[i - 1].children}
              current={current}
              className={cn(current && fold === trail.length - 1 ? 'min-w-0' : 'shrink-0')}
            />
          </Fragment>
        )
      })}
    </nav>
  )
}

function Slash() {
  return (
    <span className="shrink-0 px-0.5 text-muted-foreground/50 select-none" aria-hidden>
      /
    </span>
  )
}

const triggerClass =
  'inline-flex min-w-0 items-center rounded px-1 py-0.5 hover:bg-muted hover:text-foreground cursor-pointer ' +
  'data-popup-open:bg-muted data-popup-open:text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring'

/** The categories folded out of the trail, outermost first, as one ellipsis that opens them as a menu. */
function FoldedCrumbs({ nodes }: { nodes: CategoryNode[] }) {
  const navigate = useNavigate()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={`Show the full path: ${nodes.map((c) => c.title).join(' / ')}`}
        className={cn(triggerClass, 'shrink-0 text-muted-foreground')}
      >
        …
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-auto min-w-56">
        {nodes.map((c, i) => (
          <DropdownMenuItem key={c.path} onClick={() => navigate(browseUrl({ c: c.path }))}>
            <span style={{ paddingLeft: `${i * 0.75}rem` }}>{c.title}</span>
            <span className="ml-auto pl-4 text-xs text-muted-foreground tabular-nums">
              {notesInCategory(c.path).length}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function CategoryCrumb({
  node,
  siblings,
  current = false,
  className,
}: {
  node: CategoryNode
  siblings: CategoryNode[]
  /** The category being viewed: set in the text colour, and its menu also lists its subtopics. */
  current?: boolean
  className?: string
}) {
  const navigate = useNavigate()
  const lead = categoryIndexNote(node.path)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-current={current ? 'page' : undefined}
        className={cn(triggerClass, current ? 'font-medium text-foreground' : 'text-muted-foreground', className)}
      >
        <span className="truncate">{node.title}</span>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-auto min-w-56">
        {lead && (
          <DropdownMenuItem onClick={() => navigate(noteUrl(lead.slug))} className="font-medium">
            <BookOpen /> {lead.title}
            <span className="ml-auto pl-4 text-xs text-muted-foreground">start here</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem onClick={() => navigate(browseUrl({ c: node.path }))}>
          <LayoutGrid /> Browse {node.title}
          <span className="ml-auto pl-4 text-xs text-muted-foreground tabular-nums">
            {notesInCategory(node.path).length}
          </span>
        </DropdownMenuItem>
        {current && node.children.length > 0 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuLabel>Inside</DropdownMenuLabel>
              {node.children.map((s) => {
                const count = notesInCategory(s.path).length
                return (
                  <DropdownMenuItem
                    key={s.path}
                    onClick={() => navigate(browseUrl({ c: s.path }))}
                    className={cn(count === 0 && 'text-muted-foreground')}
                  >
                    <CornerDownRight />
                    {s.title}
                    <span className="ml-auto pl-4 text-xs text-muted-foreground tabular-nums">{count}</span>
                  </DropdownMenuItem>
                )
              })}
            </DropdownMenuGroup>
          </>
        )}
        {siblings.length > 1 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuLabel>Alongside</DropdownMenuLabel>
              {siblings.map((s) => {
                const count = notesInCategory(s.path).length
                return (
                  <DropdownMenuItem
                    key={s.path}
                    onClick={() => navigate(browseUrl({ c: s.path }))}
                    className={cn(count === 0 && 'text-muted-foreground')}
                  >
                    <Check className={cn(s.path !== node.path && 'invisible')} />
                    {s.title}
                    <span className="ml-auto pl-4 text-xs text-muted-foreground tabular-nums">{count}</span>
                  </DropdownMenuItem>
                )
              })}
            </DropdownMenuGroup>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** The current note, opening the other notes filed in the same category. */
export function NoteCrumb({ note, className }: { note: NoteMeta; className?: string }) {
  const navigate = useNavigate()
  // The category's index note leads, set apart; the rest follow in reading order.
  const siblings = indexFirst(notesInCategory(note.category, false), note.category)
  const lead = categoryIndexNote(note.category)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger aria-current="page" className={cn(triggerClass, 'font-medium text-foreground', className)}>
        <span className="truncate">{note.title}</span>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-auto max-w-sm min-w-56">
        <DropdownMenuGroup>
          <DropdownMenuLabel>{categoryTrail(note.category).at(-1)?.title}</DropdownMenuLabel>
          {siblings.map((n) => (
            <Fragment key={n.slug}>
              <DropdownMenuItem
                onClick={() => navigate(noteUrl(n.slug))}
                className={cn(n.slug === lead?.slug && 'font-medium')}
              >
                <Check className={cn(n.slug !== note.slug && 'invisible')} />
                <span className="truncate">{n.title}</span>
                {n.slug === lead?.slug && <span className="ml-auto pl-4 text-xs text-muted-foreground">overview</span>}
              </DropdownMenuItem>
              {n.slug === lead?.slug && siblings.length > 1 && <DropdownMenuSeparator />}
            </Fragment>
          ))}
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
