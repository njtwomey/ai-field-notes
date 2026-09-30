import { useContext, type ReactNode } from 'react'
import { DEFAULT_HEIGHT, FrameContext } from '@lab/viz'

/** The panels of a figure side by side, each with a title, a body filling the frame and an optional footer. */
export function Columns({ panels }: { panels: { title: string; body: ReactNode; footer?: ReactNode }[] }) {
  const { height } = useContext(FrameContext)
  return (
    <div
      className="grid gap-4"
      style={{ height: height ?? DEFAULT_HEIGHT, gridTemplateColumns: `repeat(${panels.length}, minmax(0, 1fr))` }}
    >
      {panels.map((p) => (
        <div key={p.title} className="flex min-h-0 min-w-0 flex-col gap-1">
          <div className="shrink-0 text-xs font-medium text-muted-foreground">{p.title}</div>
          <div className="min-h-0 flex-1">{p.body}</div>
          {p.footer && <div className="shrink-0 text-xs">{p.footer}</div>}
        </div>
      ))}
    </div>
  )
}

/** A labelled row of chips, e.g. a queue or a stack, with a note on which end is which. */
export function Sequence({ label, items, ends }: { label: string; items: readonly string[]; ends?: string }) {
  return (
    <div className="flex min-h-7 flex-wrap items-center gap-1">
      <span className="text-muted-foreground">
        {label}
        {ends && <span className="ml-1 opacity-70">({ends})</span>}
      </span>
      {items.length === 0 && <span className="text-muted-foreground italic">empty</span>}
      {items.map((x, i) => (
        <span key={i} className="rounded border bg-muted px-1.5 py-0.5 font-mono tabular-nums">
          {x}
        </span>
      ))}
    </div>
  )
}

/** One body filling the frame's height, with a footer (e.g. the frontier) below it. */
export function Framed({ children, footer }: { children: ReactNode; footer?: ReactNode }) {
  const { height } = useContext(FrameContext)
  return (
    <div className="flex flex-col gap-2" style={{ height: height ?? DEFAULT_HEIGHT }}>
      <div className="min-h-0 flex-1">{children}</div>
      {footer && <div className="shrink-0 text-xs">{footer}</div>}
    </div>
  )
}
