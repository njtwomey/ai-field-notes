/** `Show` and `QuickFigure`: an object's registered view as a panel, or in a standard figure. */
import type { ReactNode } from 'react'
import { Figure, type FigureProps } from '@lab/layout'
import { useFigureState } from '@lab/state/useFigureState'
import { defaultOptions, kindOf, viewFor } from './registry'

function Missing({ value }: { value: unknown }) {
  return (
    <div className="text-sm text-muted-foreground">
      No view is registered for this object (kind {String(kindOf(value) ?? 'unknown')}).
    </div>
  )
}

/** The registered view of an object, as a panel. `options` override the view's display defaults. */
export function Show({
  value,
  as,
  options,
  step,
}: {
  value: unknown
  as?: string
  options?: Record<string, unknown>
  step?: number
}) {
  const def = viewFor(value, as)
  if (!def) return <Missing value={value} />
  return <>{def.render(value, { ...defaultOptions(def), ...options } as never, { step })}</>
}

export type QuickFigureProps = Omit<FigureProps, 'children' | 'title' | 'purpose' | 'state'> & {
  value: unknown
  /** A view key, in place of the default view for the object's kind. */
  as?: string
  title?: string
  /** The figure's purpose; until a specimen states one (TODO(5c)), the view's description stands in. */
  purpose?: ReactNode
}

/** A standard figure around the view of an object: its title, its display options as state, the panel. */
export function QuickFigure({ value, as, title, purpose, ...frame }: QuickFigureProps) {
  const def = viewFor(value, as)
  const state = useFigureState(def?.options ?? {})
  if (!def)
    return (
      <Figure title={title ?? 'Object'} purpose={purpose ?? 'No view is registered for this object.'} {...frame}>
        <Missing value={value} />
      </Figure>
    )
  return (
    <Figure
      {...frame}
      title={title ?? def.title?.(value) ?? def.key}
      purpose={purpose ?? frame.description ?? def.description}
      description={purpose ? frame.description : undefined}
      state={def.options && Object.keys(def.options).length ? state : undefined}
    >
      {def.render(value, state.values as never, {})}
    </Figure>
  )
}
