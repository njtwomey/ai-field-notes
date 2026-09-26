import { AlertTriangle, BookMarked, Info, Lightbulb, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

const variants = {
  note: { icon: Info, label: 'Note' },
  tip: { icon: Lightbulb, label: 'Tip' },
  warning: { icon: AlertTriangle, label: 'Pitfall' },
  definition: { icon: BookMarked, label: 'Definition' },
} satisfies Record<string, { icon: LucideIcon; label: string }>

export type CalloutVariant = keyof typeof variants

export function Callout({
  variant = 'note',
  title,
  children,
}: {
  variant?: CalloutVariant
  title?: string
  children: ReactNode
}) {
  const { icon: Icon, label } = variants[variant]
  return (
    <aside
      className={cn(
        'not-prose my-8 rounded-lg border px-5 py-4',
        variant === 'definition' ? 'border-foreground/20 bg-muted/60' : 'bg-card',
      )}
    >
      <div className="mb-1 flex items-center gap-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
        <Icon className="size-3.5" aria-hidden />
        {title ?? label}
      </div>
      <div className="space-y-2 font-prose text-[1.0625rem] leading-relaxed [&_.katex-display]:my-2">{children}</div>
    </aside>
  )
}

/** The one-paragraph definition that opens a concept note. */
export function Definition({ children }: { children: ReactNode }) {
  return <Callout variant="definition">{children}</Callout>
}
