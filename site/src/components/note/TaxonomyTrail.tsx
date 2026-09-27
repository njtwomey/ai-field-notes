import { BookOpen, Check, ChevronDown, ChevronRight, CornerDownRight, LayoutGrid } from 'lucide-react'
import { Fragment } from 'react'
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
import { kindIcons } from '@/components/layout/kind-icon'
import {
  browseUrl,
  categoryIndexNote,
  indexFirst,
  kindLabels,
  categoryTrail,
  notesInCategory,
  noteUrl,
  taxonomy,
  type CategoryNode,
  type NoteMeta,
} from '@/lib/content'
import { cn } from '@/lib/utils'

/**
 * A note's place in the taxonomy as a trail of dropdowns, ending in the note itself. Each category opens its siblings,
 * so a reader can move sideways at any level without going back; the note opens the other notes in its category.
 * On narrow screens only the note crumb shows.
 */
export function TaxonomyTrail({ note, className }: { note: NoteMeta; className?: string }) {
  const trail = categoryTrail(note.category)
  const Icon = kindIcons[note.kind]
  return (
    <nav aria-label="Breadcrumb" className={cn('flex min-w-0 items-center gap-x-1 text-sm', className)}>
      {trail.map((c, i) => (
        <Fragment key={c.path}>
          <CategoryCrumb node={c} siblings={i === 0 ? taxonomy : trail[i - 1].children} />
          <ChevronRight className="hidden size-3.5 shrink-0 text-muted-foreground md:block" aria-hidden />
        </Fragment>
      ))}
      <Icon className="ml-1 size-3.5 shrink-0 text-muted-foreground" aria-label={kindLabels[note.kind]} />
      <NoteCrumb note={note} />
    </nav>
  )
}

/**
 * A category's place in the taxonomy as the same trail of dropdowns, for the browse page: "All topics", each ancestor,
 * then the category itself in bold. The last crumb also opens the category's subtopics, so the tree can be walked both
 * up and down.
 */
export function CategoryTrail({ path, className }: { path: string; className?: string }) {
  const trail = categoryTrail(path)
  return (
    <nav aria-label="Breadcrumb" className={cn('flex min-w-0 flex-wrap items-center gap-x-1 text-sm', className)}>
      <Link to={browseUrl()} className={cn(triggerClass, 'text-muted-foreground')}>
        All topics
      </Link>
      {trail.map((c, i) => (
        <Fragment key={c.path}>
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
          <CategoryCrumb
            node={c}
            siblings={i === 0 ? taxonomy : trail[i - 1].children}
            current={i === trail.length - 1}
            className="inline-flex"
          />
        </Fragment>
      ))}
    </nav>
  )
}

const triggerClass =
  'inline-flex min-w-0 items-center gap-0.5 rounded-md px-1 py-0.5 hover:bg-muted hover:text-foreground ' +
  'data-popup-open:bg-muted data-popup-open:text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring'

function CategoryCrumb({
  node,
  siblings,
  current = false,
  className,
}: {
  node: CategoryNode
  siblings: CategoryNode[]
  /** The category being viewed: set in bold, and its menu also lists its subtopics. */
  current?: boolean
  className?: string
}) {
  const navigate = useNavigate()
  const lead = categoryIndexNote(node.path)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-current={current ? 'page' : undefined}
        className={cn(
          triggerClass,
          'hidden md:inline-flex',
          current ? 'font-semibold text-foreground' : 'text-muted-foreground',
          className,
        )}
      >
        <span className="truncate">{node.title}</span>
        <ChevronDown className="size-3 shrink-0 opacity-60" aria-hidden />
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
  // The category's index note leads, set apart; the rest follow alphabetically.
  const siblings = indexFirst(
    notesInCategory(note.category, false).sort((a, b) => a.title.localeCompare(b.title)),
    note.category,
  )
  const lead = categoryIndexNote(note.category)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger aria-current="page" className={cn(triggerClass, 'font-semibold text-foreground', className)}>
        <span className="truncate">{note.title}</span>
        <ChevronDown className="size-3 shrink-0 opacity-60" aria-hidden />
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
