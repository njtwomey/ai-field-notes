import { type ReactNode } from 'react'
import { Figure } from '@render/layout/Figure'
import { cn } from '@render/lib/utils'

export type InteractiveProps = {
  title: string
  caption?: ReactNode
  purpose?: ReactNode
  controls?: ReactNode
  readout?: ReactNode
  children: ReactNode
  className?: string
}

/** The standard frame for every interactive figure: backed by Figure from @render/layout. */
export function Interactive({ title, caption, purpose, controls, readout, children, className }: InteractiveProps) {
  return (
    <Figure
      title={title}
      purpose={purpose ?? ''}
      caption={caption}
      controls={controls}
      readouts={readout}
      className={cn('not-prose my-8', className)}
    >
      {children}
    </Figure>
  )
}

export function Readout({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <span>
      {label} <span className="font-mono text-foreground tabular-nums">{value}</span>
    </span>
  )
}
