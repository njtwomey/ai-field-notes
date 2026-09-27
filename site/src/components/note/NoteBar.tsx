import { BookText, ChevronRight, Code2, SquareTerminal } from 'lucide-react'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { Heading } from '@/hooks/use-reading-position'
import type { NoteMeta } from '@/lib/content'
import { TaxonomyTrail } from './TaxonomyTrail'

export type NoteTab = 'concept' | 'code' | 'outputs'

/**
 * Sticky bar under the site header and the page's only breadcrumb: the taxonomy trail ending in the note's title,
 * then the section being read, a reading-progress line, and (when
 * the note has code) its tabs. The page's h1 is in NoteHeader.
 */
export function NoteBar({
  note,
  tab,
  onTab,
  section = [],
  progress,
}: {
  note: NoteMeta
  tab: NoteTab
  onTab: (tab: NoteTab) => void
  /** The section being read, outermost first, e.g. [Choosing k, AIC and BIC]. */
  section?: Heading[]
  /** Fraction of the page read, 0 to 1. Omit to hide the progress line. */
  progress?: number
}) {
  return (
    <div className="sticky top-14 z-20 -mx-4 flex h-14 items-center gap-4 border-b bg-background/90 px-4 backdrop-blur">
      {progress !== undefined && (
        <div
          aria-hidden
          className="absolute bottom-[-1px] left-0 h-0.5 bg-primary transition-[width] duration-100"
          style={{ width: `${progress * 100}%` }}
        />
      )}
      <div className="flex min-w-0 items-center gap-1 text-sm">
        <TaxonomyTrail note={note} className="shrink-0" />
        {section.length > 0 && (
          <a
            href={`#${section[section.length - 1].id}`}
            className="flex min-w-0 items-center gap-1 text-muted-foreground hover:text-foreground"
          >
            <ChevronRight className="size-3.5 shrink-0" aria-hidden />
            <span className="truncate">{section.map((h) => h.text).join(' › ')}</span>
          </a>
        )}
      </div>
      {note.code && (
        <Tabs value={tab} onValueChange={(v) => onTab(v as NoteTab)} className="ml-auto shrink-0">
          <TabsList>
            <TabsTrigger value="concept">
              <BookText /> Concept
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code2 /> Code
            </TabsTrigger>
            <TabsTrigger value="outputs">
              <SquareTerminal /> Outputs
            </TabsTrigger>
          </TabsList>
        </Tabs>
      )}
    </div>
  )
}
