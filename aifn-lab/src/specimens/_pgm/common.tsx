import { useContext, type ReactNode } from 'react'
import { DEFAULT_HEIGHT, FrameContext } from '@lab/viz'

/** Panels side by side, each with a small title and a body filling the frame's height. */
export function Columns({
  panels,
  widths,
}: {
  panels: { title: string; body: ReactNode }[]
  widths?: readonly number[]
}) {
  const { height } = useContext(FrameContext)
  return (
    <div
      className="grid gap-4"
      style={{
        height: height ?? DEFAULT_HEIGHT,
        gridTemplateColumns: (widths ?? panels.map(() => 1)).map((w) => `minmax(0, ${w}fr)`).join(' '),
      }}
    >
      {panels.map((p) => (
        <div key={p.title} className="flex min-h-0 min-w-0 flex-col gap-1">
          <div className="shrink-0 text-xs font-medium text-muted-foreground">{p.title}</div>
          <div className="min-h-0 flex-1">{p.body}</div>
        </div>
      ))}
    </div>
  )
}
