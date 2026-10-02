import type { ReactNode } from 'react'
import { cn } from '@lab/lib/utils'

/** The small muted label every control sits under. */
export function ControlLabel({
  children,
  htmlFor,
  id,
  className,
}: {
  children: ReactNode
  htmlFor?: string
  id?: string
  className?: string
}) {
  return (
    <label
      id={id}
      htmlFor={htmlFor}
      className={cn('text-xs leading-none text-muted-foreground select-none', className)}
    >
      {children}
    </label>
  )
}
