import { useContext, useMemo, type ReactNode } from 'react'
import { RenderMathContext } from './math-macros'

/** Provider to supply or override mathematical macros for a subtree. */
export function RenderMathProvider({ macros, children }: { macros?: Record<string, string>; children: ReactNode }) {
  const parentMacros = useContext(RenderMathContext)
  const merged = useMemo(() => ({ ...(parentMacros ?? {}), ...(macros ?? {}) }), [parentMacros, macros])
  return <RenderMathContext.Provider value={merged}>{children}</RenderMathContext.Provider>
}
