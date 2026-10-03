import { useMemo } from 'react'
import { ParamSlider, Readout, formatNumber, useParam } from 'aifn-render'
import { ConstrainedExplorer, SensitivityPanel, type ConstrainedProblem } from './ConstrainedExplorer'

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
  const c = useParam(4, { min: 1, max: 6, step: 0.1 })
  const p = useMemo(() => problem(c.value), [c.value])
  return (
    <ConstrainedExplorer
      problem={p}
      title="Maximise xy on x + y = c"
      caption="Left: hyperbolas xy = const (grey), the line x + y = c, and the arrow of ∇f = (y, x) against the line's normal (1, 1). Drag the point along the line. Right, top: the area xy along the line. Right, bottom: the largest area f*(c) = c²/4 against c; drag the vertical line or use the slider. The dashed tangent has slope λ = c/2, the extra area per unit of extra half-perimeter."
      initialT={1.5}
      tStep={0.01}
      xLabel="x"
      yLabel="y"
      tLabel="position t along the line"
      tSymbol="t"
      fLabel="xy"
      controls={<ParamSlider label="half-perimeter c" param={c} />}
      readout={() => (
        <>
          <Readout label="optimum (c/2, c/2)" value={`(${formatNumber(c.value / 2)}, ${formatNumber(c.value / 2)})`} />
          <Readout label="f*(c) = c²/4" value={formatNumber(optimum(c.value))} />
          <Readout label="λ* = c/2" value={formatNumber(multiplier(c.value))} />
        </>
      )}
      profileHeight={220}
      extra={
        <SensitivityPanel
          param={c}
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
