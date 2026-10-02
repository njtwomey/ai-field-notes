import type { ReactNode } from 'react'
import { cn } from '@render/lib/utils'

/**
 * The layout for a set of controls: a responsive grid whose columns are at least 14rem, so labels and tracks line up
 * and sliders never collapse. Figures use it for their controls slot; use it anywhere else controls sit together,
 * rather than an ad-hoc flex row. Children fill their cell.
 */
export function Controls({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        'grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] items-end gap-x-6 gap-y-4 *:w-full *:max-w-none',
        className,
      )}
    >
      {children}
    </div>
  )
}

/**
 * One full-width row of related controls inside a `Controls` grid (a Figure's controls slot), e.g. a selector followed
 * by its own parameters. Rows stack, so each group reads on its own line instead of wrapping into its neighbours.
 */
export function ControlRow({
  label,
  children,
  className,
}: {
  label?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('col-span-full flex flex-col gap-2', className)}>
      {label && <div className="text-xs font-medium text-muted-foreground">{label}</div>}
      <Controls>{children}</Controls>
    </div>
  )
}
