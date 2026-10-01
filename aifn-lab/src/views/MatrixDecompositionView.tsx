import { useContext, useMemo, useState, type ReactNode } from 'react'
import type { Tensor } from 'aifn/foundation/tensor'
import { Select } from '@lab/controls'
import { PanelSlot } from '@lab/layout'
import { DEFAULT_HEIGHT, FrameContext } from '@lab/viz'
import { TENSOR_MODES, TensorPanel, type TensorMode } from './TensorView'

export type MatrixDecompositionPanelProps = {
  /** The factors in the order of the product, e.g. [{ name: 'L', tensor: L }, { name: 'Lᵀ', tensor: Lt }]. */
  factors: { name: string; tensor: Tensor }[]
  /** The matrix being decomposed, shown first. */
  input?: { name: string; tensor: Tensor }
  /** Flags and diagnostics (`Readout`s), e.g. jitter or `singular`, shown with the figure's readouts. */
  notes?: ReactNode
}

/**
 * A matrix and its factors side by side as one panel, each a `TensorPanel` heatmap (or table), with the view choice
 * and any diagnostics the decomposition reported in the enclosing figure. The panels share the frame's height.
 */
export function MatrixDecompositionPanel({ factors, input, notes }: MatrixDecompositionPanelProps) {
  const [mode, setMode] = useState<TensorMode>('chart')
  const panels = input ? [input, ...factors] : factors
  // Each heatmap takes three quarters of the frame's height, leaving room for its name and shape line.
  const outer = useContext(FrameContext)
  const frame = useMemo(() => ({ ...outer, height: Math.round((outer.height ?? DEFAULT_HEIGHT) * 0.75) }), [outer])
  return (
    <>
      <PanelSlot slot="controls">
        <Select label="view" value={mode} onChange={setMode} options={TENSOR_MODES(2)} />
      </PanelSlot>
      {notes && <PanelSlot slot="readouts">{notes}</PanelSlot>}
      <FrameContext.Provider value={frame}>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-4">
          {panels.map((p) => (
            <TensorPanel key={p.name} name={p.name} tensor={p.tensor} mode={mode} />
          ))}
        </div>
      </FrameContext.Provider>
    </>
  )
}
