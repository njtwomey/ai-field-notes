import { Check, ChevronDown, ChevronRight, LayoutGrid } from 'lucide-react'
import { Fragment } from 'react'
import { useNavigate } from 'react-router'
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

const triggerClass =
  'inline-flex min-w-0 items-center gap-0.5 rounded-md px-1 py-0.5 hover:bg-muted hover:text-foreground ' +
  'data-popup-open:bg-muted data-popup-open:text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring'

function CategoryCrumb({ node, siblings }: { node: CategoryNode; siblings: CategoryNode[] }) {
  const navigate = useNavigate()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger className={cn(triggerClass, 'hidden text-muted-foreground md:inline-flex')}>
        <span className="truncate">{node.title}</span>
        <ChevronDown className="size-3 shrink-0 opacity-60" aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-auto min-w-56">
        <DropdownMenuItem onClick={() => navigate(browseUrl({ c: node.path }))}>
          <LayoutGrid /> Browse {node.title}
          <span className="ml-auto pl-4 text-xs text-muted-foreground tabular-nums">
            {notesInCategory(node.path).length}
          </span>
        </DropdownMenuItem>
        {siblings.length > 1 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuLabel>Alongside</DropdownMenuLabel>
              {siblings.map((s) => (
                <DropdownMenuItem key={s.path} onClick={() => navigate(browseUrl({ c: s.path }))}>
                  <Check className={cn(s.path !== node.path && 'invisible')} />
                  {s.title}
                  <span className="ml-auto pl-4 text-xs text-muted-foreground tabular-nums">
                    {notesInCategory(s.path).length}
                  </span>
                </DropdownMenuItem>
              ))}
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
  const siblings = notesInCategory(note.category, false).sort((a, b) => a.title.localeCompare(b.title))
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
            <DropdownMenuItem key={n.slug} onClick={() => navigate(noteUrl(n.slug))}>
              <Check className={cn(n.slug !== note.slug && 'invisible')} />
              <span className="truncate">{n.title}</span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
