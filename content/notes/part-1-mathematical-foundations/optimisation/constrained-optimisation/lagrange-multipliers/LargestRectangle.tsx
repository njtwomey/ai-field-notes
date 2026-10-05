import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, SensitivityPanel, type ConstrainedProblem } from './ConstrainedExplorer'
import { tField } from './t-field'

/** Maximise xy on the line x + y = c, parametrised by signed distance t from the midpoint (c/2, c/2). */
function problem(c: number): ConstrainedProblem {
  return {
    f: (x, y) => x * y,
    gradF: (x, y) => [y, x],
    gradG: () => [1, 1],
    curve: (t) => [c / 2 + t / Math.SQRT2, c / 2 - t / Math.SQRT2],
    tRange: [-4.5, 4.5],
    xRange: [-1.5, 6.5],
    yRange: [-1.5, 6.5],
    levels: [-4, -1, 1, 2.25, 4, 6.25, 9, 12.25],
    goal: 'max',
  }
}

const optimum = (c: number) => (c * c) / 4
const multiplier = (c: number) => c / 2

export function LargestRectangle() {
  const state = useFigureState({
    t: tField([-4.5, 4.5], 1.5, 0.01, 'position t along the line'),
    c: slider(1, 6, 4, { step: 0.1, label: 'half-perimeter c' }),
  })
  const p = useMemo(() => problem(state.c), [state.c])
  return (
    <ConstrainedExplorer
      state={state}
      problem={p}
      title="Maximise xy on x + y = c"
      caption="Left: hyperbolas xy = const (grey), the line x + y = c, and the arrow of ∇f = (y, x) against the line's normal (1, 1). Drag the point along the line. Right, top: the area xy along the line. Right, bottom: the largest area f*(c) = c²/4 against c; drag the vertical line or use the slider. The dashed tangent has slope λ = c/2, the extra area per unit of extra half-perimeter."
      xLabel="x"
      yLabel="y"
      tLabel="position t along the line"
      tSymbol="t"
      fLabel="xy"
      readout={() => (
        <>
          <Readout label="optimum (c/2, c/2)" value={`(${formatNumber(state.c / 2)}, ${formatNumber(state.c / 2)})`} />
          <Readout label="f*(c) = c²/4" value={formatNumber(optimum(state.c))} />
          <Readout label="λ* = c/2" value={formatNumber(multiplier(state.c))} />
        </>
      )}
      profileHeight={220}
      extra={
        <SensitivityPanel
          param={state.bind('c')}
          optimum={optimum}
          multiplier={multiplier}
          xLabel="half-perimeter c"
          symbol="c"
          yLabel="f*(c)"
        />
      }
    />
  )
}
