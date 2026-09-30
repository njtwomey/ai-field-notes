import { useState, type ReactNode } from 'react'
import type { Tensor } from 'aifn/foundation/tensor'
import { Select } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize } from '@lab/viz'
import type { FrameProps } from './frame'
import { TENSOR_MODES, TensorPanel, type TensorMode } from './TensorView'

export type MatrixDecompositionViewProps = FrameProps & {
  /** The factors in the order of the product, e.g. [{ name: 'L', tensor: L }, { name: 'Lᵀ', tensor: Lt }]. */
  factors: { name: string; tensor: Tensor }[]
  /** The matrix being decomposed, shown first. */
  input?: { name: string; tensor: Tensor }
  /** How the factors combine, e.g. "A = L Lᵀ"; the figure's title unless one is given. */
  equation: string
  /** Flags and diagnostics (`Readout`s), e.g. jitter or `singular`, shown with the figure's readouts. */
  notes?: ReactNode
}

/**
 * A matrix and its factors side by side as one figure, each a `TensorPanel` heatmap (or table), with the equation that
 * relates them and any diagnostics the decomposition reported. The panels share the frame's height.
 */
export function MatrixDecompositionView({
  factors,
  input,
  equation,
  notes,
  title,
  controls,
  readouts,
  ...frame
}: MatrixDecompositionViewProps) {
  const [mode, setMode] = useState<TensorMode>('chart')
  const panels = input ? [input, ...factors] : factors
  return (
    <Figure
      title={title ?? equation}
      defaultSize="L"
      {...frame}
      controls={
        <>
          {controls}
          <Select label="view" value={mode} onChange={setMode} options={TENSOR_MODES(2)} />
        </>
      }
      readouts={
        (readouts || notes) && (
          <>
            {readouts}
            {notes}
          </>
        )
      }
    >
      <ChartSize scale={0.75}>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-4">
          {panels.map((p) => (
            <TensorPanel key={p.name} name={p.name} tensor={p.tensor} mode={mode} />
          ))}
        </div>
      </ChartSize>
    </Figure>
  )
}
