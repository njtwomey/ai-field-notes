import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, tField, SensitivityPanel, type ConstrainedProblem } from './ConstrainedExplorer'

const SQRT5 = Math.sqrt(5)

/** Minimise x² + y² on the line x + 2y = c. The line is parametrised by signed distance t from its closest point. */
function problem(c: number): ConstrainedProblem {
  return {
    f: (x, y) => x * x + y * y,
    gradF: (x, y) => [2 * x, 2 * y],
    gradG: () => [1, 2],
    curve: (t) => [c / 5 + (2 * t) / SQRT5, (2 * c) / 5 - t / SQRT5],
    tRange: [-4, 4],
    xRange: [-2, 5],
    yRange: [-2, 4],
    levels: [0.5, 2, 4.5, 8, 12.5, 18],
    goal: 'min',
  }
}

const optimum = (c: number) => (c * c) / 5
const multiplier = (c: number) => (2 * c) / 5

export function ClosestPointOnLine() {
  const state = useFigureState({
    t: tField([-4, 4], -1.5, 0.01, 'position t along the line'),
    c: slider(0.5, 8, 5, { step: 0.1, label: 'constraint level c' }),
  })
  const p = useMemo(() => problem(state.c), [state.c])
  return (
    <ConstrainedExplorer
      state={state}
      problem={p}
      title="The closest point on a line, and the multiplier as a slope"
      caption="Left: circles x² + y² = const (grey), the line x + 2y = c, and the arrow of ∇f against the line's normal (1, 2). Drag the point along the line, or set its position t. Right, top: f along the line. Right, bottom: the optimal value f*(c) = c²/5 against the level c; drag the vertical line or use the slider to move the constraint. The dashed tangent has slope λ = 2c/5, the multiplier at the optimum."
      xLabel="x"
      yLabel="y"
      tLabel="position t along the line"
      tSymbol="t"
      fLabel="f"
      readout={() => (
        <>
          <Readout
            label="optimum (c/5, 2c/5)"
            value={`(${formatNumber(state.c / 5)}, ${formatNumber((2 * state.c) / 5)})`}
          />
          <Readout label="f*(c) = c²/5" value={formatNumber(optimum(state.c))} />
          <Readout label="λ* = 2c/5" value={formatNumber(multiplier(state.c))} />
        </>
      )}
      profileHeight={220}
      extra={
        <SensitivityPanel
          param={state.bind('c')}
          optimum={optimum}
          multiplier={multiplier}
          xLabel="constraint level c"
          symbol="c"
          yLabel="f*(c)"
        />
      }
    />
  )
}
