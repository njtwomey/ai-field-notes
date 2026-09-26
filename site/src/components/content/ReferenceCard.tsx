import { BookOpen, Code2, ExternalLink, FileText, GraduationCap, Newspaper, Video } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { formatAuthors, type Reference } from '@/lib/content'

export const referenceIcons: Record<Reference['type'], LucideIcon> = {
  paper: FileText,
  blog: Newspaper,
  book: BookOpen,
  video: Video,
  docs: FileText,
  code: Code2,
  course: GraduationCap,
}

/**
 * A reference: title link, authors, venue and year. The full form (hover cards, reference lists) adds a type icon and
 * the note; `compact` (margin notes) shows only the essentials.
 */
export function ReferenceCard({ reference, compact }: { reference: Reference; compact?: boolean }) {
  const Icon = referenceIcons[reference.type]
  const meta = [formatAuthors(reference.authors), reference.venue, reference.year].filter(Boolean).join(' · ')
  return (
    <div className="flex gap-2">
      {!compact && <Icon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" aria-hidden />}
      <div className="min-w-0 space-y-0.5">
        <a
          href={reference.url}
          target="_blank"
          rel="noreferrer"
          className="group inline-flex items-start gap-1 text-sm leading-snug font-medium hover:underline"
        >
          {reference.title}
          <ExternalLink className="mt-1 size-3 shrink-0 opacity-50 group-hover:opacity-100" aria-hidden />
        </a>
        {meta && <div className="text-xs text-muted-foreground">{meta}</div>}
        {!compact && reference.note && <p className="text-xs text-muted-foreground">{reference.note}</p>}
      </div>
    </div>
  )
}
