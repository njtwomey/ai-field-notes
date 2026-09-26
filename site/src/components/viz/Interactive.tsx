import type { ReactNode } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export type InteractiveProps = {
  title: string
  /** One or two sentences: what to change and what to watch. */
  caption?: ReactNode
  /** Parameter controls, laid out in a responsive grid above the figure. */
  controls?: ReactNode
  /** Small readouts (e.g. current loss) shown under the figure. */
  readout?: ReactNode
  children: ReactNode
  className?: string
}

/** The standard frame for every interactive figure: title, controls, figure, readout, caption. */
export function Interactive({ title, caption, controls, readout, children, className }: InteractiveProps) {
  return (
    <Card className={cn('not-prose my-8 gap-4', className)}>
      <CardHeader>
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {caption && <CardDescription className="text-xs">{caption}</CardDescription>}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {controls && <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">{controls}</div>}
        {children}
        {readout && <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">{readout}</div>}
      </CardContent>
    </Card>
  )
}

export function Readout({ label, value }: { label: string; value: ReactNode }) {
  return (
    <span>
      {label} <span className="font-mono text-foreground tabular-nums">{value}</span>
    </span>
  )
}
