import { Link } from 'react-router'
import { kindIcons } from '@/components/layout/kind-icon'
import { category, kindLabels, noteUrl, type NoteMeta } from '@/lib/content'
import { cn } from '@/lib/utils'
import { MathText } from '@/components/content/MathText'

/** A note at a glance: kind and title first, then where it sits, then one or two lines of summary. */
export function NoteCard({ note, className }: { note: NoteMeta; className?: string }) {
  const Icon = kindIcons[note.kind]
  return (
    <Link
      to={noteUrl(note.slug)}
      className={cn(
        'group flex flex-col gap-1.5 rounded-lg border p-4 transition-colors hover:border-foreground/30 hover:bg-muted/40',
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-label={kindLabels[note.kind]} />
        <span className="leading-snug font-medium group-hover:underline">{note.title}</span>
        {note.status === 'stub' && <span className="ml-auto text-[10px] text-muted-foreground italic">stub</span>}
      </div>
      <span className="pl-6 text-xs text-muted-foreground">
        {kindLabels[note.kind]} · {category(note.category)?.title}
      </span>
      <p className="line-clamp-2 pl-6 text-sm text-muted-foreground">
        <MathText text={note.summary} />
      </p>
    </Link>
  )
}
