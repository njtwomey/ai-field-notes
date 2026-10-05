import { useMemo } from 'react'
import { Annotation, Curve, Figure, Handle, Plot, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'

const X = toFlat(linspace(-6, 6, 121))

export function SigmoidExplorer() {
  const state = useFigureState({
    w: slider(-4, 4, 1, { step: 0.1, label: 'weight w' }),
    b: slider(-4, 4, 0, { step: 0.1, label: 'bias b' }),
  })
  const { w, b } = state
  const y = useMemo(() => X.map((xi) => sigmoid(w * xi + b)), [w, b])
  // The crossing point x = −b/w as a handle: dragging it sets b = −w·x. Undefined when w = 0 (no crossing). Shown
  // at the edge of the plot when the crossing lies beyond it.
  const crossing = Math.abs(w) > 1e-9 ? -b / w : undefined
  const xAxis = useAxis({ label: 'x', range: [-6, 6] })
  const yAxis = useAxis({ label: 'P(y = 1 | x)', range: [0, 1] })
  return (
    <Figure
      title="The logistic function"
      state={state}
      caption="The weight sets the steepness. The bias shifts the point where the probability crosses 0.5, at x = −b/w. Drag the dashed line to move that point; the bias follows."
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Curve name="σ(wx + b)" x={X} y={y} slot={0} />
        <Annotation y={0.5} text="0.5" dashed muted />
        {crossing !== undefined && (
          <Handle
            kind="x"
            at={Math.min(Math.max(crossing, -6), 6)}
            label="P = 0.5"
            onDrag={(at) => state.set('b', Math.min(Math.max(-w * at, -4), 4))}
          />
        )}
      </Plot>
    </Figure>
  )
}
