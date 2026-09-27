import type { ReactNode } from 'react'
import { Link } from 'react-router'
import { Badge } from '@/components/ui/badge'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { kindLabels, notesBySlug, noteUrl, prefetchNote } from '@/lib/content'
import { MathText } from './MathText'

/** Link to another note, with its summary on hover: `<NoteLink to="softmax">softmax</NoteLink>`. Checked at build. */
export function NoteLink({ to, children }: { to: string; children?: ReactNode }) {
  const target = notesBySlug.get(to)
  if (!target) return <span className="text-destructive">{children ?? to}</span>
  return (
    <HoverCard>
      <HoverCardTrigger
        render={
          <Link
            to={noteUrl(to)}
            onMouseEnter={() => prefetchNote(to)}
            onFocus={() => prefetchNote(to)}
            className="underline decoration-muted-foreground/50 underline-offset-2"
          />
        }
      >
        {children ?? target.title}
      </HoverCardTrigger>
      <HoverCardContent className="w-80" side="top">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">{target.title}</span>
            <Badge variant="secondary" className="text-[10px]">
              {kindLabels[target.kind]}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            <MathText text={target.summary} />
          </p>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
