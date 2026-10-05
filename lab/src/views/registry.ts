/**
 * The view registry (design S §4.1–4.2). A **view** draws one aifn object as a panel: charts, a table or a diagram,
 * with no frame and no state of the page's (a view never renders a `Figure`; its own controls and readouts go to the
 * enclosing figure through `PanelSlot`). Views are registered by the object's `kind`; `Show` picks the view for an
 * object, and `QuickFigure` / `quickFigure` wrap one in a standard `Figure` for the lab's "show me this object" pages,
 * with the view's display options as the figure's state.
 *
 *   registerView({ key: 'distribution/density', kind: 'distribution', title: (d) => d.name, render: (d) => <DistributionPanel distribution={d} /> })
 *   <Show value={dist} />                              // the panel alone (a dashboard cell, a page)
 *   <QuickFigure value={dist} purpose="…" />            // the panel in a Figure
 */
import type { ReactNode } from 'react'
import type { ParamDefs, Values } from 'aifn-render/state'

/** What a view may read from its figure beyond the object and its options. */
export type ViewContext = {
  /** The figure's current step, when it has a `Player`. */
  step?: number
}

export type ViewDef<O = unknown, S extends ParamDefs = ParamDefs> = {
  /** Unique, `kind/name`: 'distribution/density', 'trace/series'. */
  key: string
  /** The object kind it draws (see `kindOf`). */
  kind: string
  /** Narrower than kind: univariate only, 2-D inputs only, a model with `decide`. */
  accepts?: (o: O) => boolean
  /** One line: what the view shows. Listed in the catalog and on the UI kit's views page. */
  description: string
  /** The default title of a quick figure. */
  title?: (o: O) => string
  /** Display options (figure-state fields), drawn as the quick figure's control rows. */
  options?: S
  render: (o: O, options: Values<S>, ctx: ViewContext) => ReactNode
}

const views: ViewDef[] = []
const kinds: { kind: string; test: (o: unknown) => boolean }[] = []

/** Register a view. Registering a key twice replaces the earlier view (a hot reload). */
export function registerView<O, S extends ParamDefs = Record<never, never>>(def: ViewDef<O, S>): ViewDef<O, S> {
  const at = views.findIndex((v) => v.key === def.key)
  if (at >= 0) views[at] = def as unknown as ViewDef
  else views.push(def as unknown as ViewDef)
  return def
}

/**
 * Teach `kindOf` an object kind that its objects do not name themselves (a distribution, a tensor): the first test
 * that passes wins, after the object's own `kind` field.
 */
export function registerKind(kind: string, test: (o: unknown) => boolean): void {
  if (!kinds.some((k) => k.kind === kind)) kinds.push({ kind, test })
}

/** The kind of an aifn object: its own `kind` field, else the first registered test that matches. */
export function kindOf(o: unknown): string | undefined {
  for (const k of kinds) if (k.test(o)) return k.kind
  if (typeof o === 'object' && o !== null && typeof (o as { kind?: unknown }).kind === 'string')
    return (o as { kind: string }).kind
  return undefined
}

/** Every registered view, in registration order. */
export const registeredViews = (): readonly ViewDef[] => views

/** The view for an object: the named one (`as`), else the first registered whose kind and `accepts` match. */
export function viewFor(o: unknown, as?: string): ViewDef | undefined {
  if (as) return views.find((v) => v.key === as)
  const kind = kindOf(o)
  return views.find((v) => v.kind === kind && (!v.accepts || v.accepts(o)))
}

/** A view's display options at their initial values. */
export const defaultOptions = (def: ViewDef) =>
  Object.fromEntries(Object.entries(def.options ?? {}).map(([k, d]) => [k, 'initial' in d ? d.initial : undefined]))
