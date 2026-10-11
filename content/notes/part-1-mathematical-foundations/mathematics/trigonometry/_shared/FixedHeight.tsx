import { useContext, type ReactNode } from 'react'
import { FrameContext } from 'aifn-render'

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}
