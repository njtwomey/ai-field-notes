import type { ComponentType, ReactNode } from 'react'
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

/**
 * A structured section or card of related controls (e.g. Visual, Color, Parameters, Playback),
 * with an optional title, subtitle/description, badge, and icon.
 */
export function ControlGroup({
  title,
  description,
  badge,
  icon: Icon,
  children,
  className,
}: {
  title?: ReactNode
  description?: ReactNode
  badge?: ReactNode
  icon?: ComponentType<{ className?: string }>
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'col-span-full flex flex-col gap-3 rounded-lg border bg-card/50 p-3.5 text-card-foreground shadow-xs',
        className,
      )}
    >
      {(title || description || badge || Icon) && (
        <div className="flex items-center justify-between gap-2 border-b border-border/60 pb-2">
          <div className="flex items-center gap-2 min-w-0">
            {Icon && <Icon className="size-4 shrink-0 text-muted-foreground" />}
            <div className="min-w-0">
              {title && <h3 className="text-xs font-semibold tracking-wide uppercase text-foreground/80">{title}</h3>}
              {description && <p className="text-[11px] text-muted-foreground">{description}</p>}
            </div>
          </div>
          {badge && (
            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground shrink-0">
              {badge}
            </span>
          )}
        </div>
      )}
      <Controls>{children}</Controls>
    </div>
  )
}

