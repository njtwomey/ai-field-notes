import { useMemo } from 'react'
import { ParamSlider, Readout, formatNumber, useParam } from '@/components/viz'
import { ConstrainedExplorer, SensitivityPanel, type ConstrainedProblem } from './ConstrainedExplorer'

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
  const c = useParam(5, { min: 0.5, max: 8, step: 0.1 })
  const p = useMemo(() => problem(c.value), [c.value])
  return (
    <ConstrainedExplorer
      problem={p}
      title="The closest point on a line, and the multiplier as a slope"
      caption="Left: circles x² + y² = const (grey), the line x + 2y = c, and the arrow of ∇f against the line's normal (1, 2). Drag the point along the line, or set its position t. Right, top: f along the line. Right, bottom: the optimal value f*(c) = c²/5 against the level c; drag the vertical line or use the slider to move the constraint. The dashed tangent has slope λ = 2c/5, the multiplier at the optimum."
      initialT={-1.5}
      tStep={0.01}
      xLabel="x"
      yLabel="y"
      tLabel="position t along the line"
      tSymbol="t"
      fLabel="f"
      controls={<ParamSlider label="constraint level c" param={c} />}
      readout={() => (
        <>
          <Readout
            label="optimum (c/5, 2c/5)"
            value={`(${formatNumber(c.value / 5)}, ${formatNumber((2 * c.value) / 5)})`}
          />
          <Readout label="f*(c) = c²/5" value={formatNumber(optimum(c.value))} />
          <Readout label="λ* = 2c/5" value={formatNumber(multiplier(c.value))} />
        </>
      )}
      profileHeight={220}
      extra={
        <SensitivityPanel
          param={c}
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
