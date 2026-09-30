import { useContext, useMemo, type ReactNode } from 'react'
import { DEFAULT_HEIGHT, FrameContext } from './frame'

/**
 * Overrides the height of the charts inside, relative to the frame: `scale={0.5}` for a panel half the frame's height
 * (e.g. a timing strip under a trace), or an absolute `height`. Data and hover registration pass through.
 */
export function ChartSize({ height, scale, children }: { height?: number; scale?: number; children: ReactNode }) {
  const outer = useContext(FrameContext)
  const base = outer.height ?? DEFAULT_HEIGHT
  const next = height ?? Math.round(base * (scale ?? 1))
  const value = useMemo(() => ({ ...outer, height: next }), [outer, next])
  return <FrameContext.Provider value={value}>{children}</FrameContext.Provider>
}
