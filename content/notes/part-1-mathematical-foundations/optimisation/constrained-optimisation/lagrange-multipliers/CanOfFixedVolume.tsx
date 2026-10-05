import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, SensitivityPanel, type ConstrainedProblem } from './ConstrainedExplorer'
import { tField } from './t-field'

const PI = Math.PI

/** Minimise the surface area 2πr² + 2πrh of a closed cylinder subject to the volume πr²h = V. */
function problem(volume: number): ConstrainedProblem {
  return {
    f: (r, h) => (r < 0 || h < 0 ? NaN : 2 * PI * r * r + 2 * PI * r * h),
    gradF: (r, h) => [4 * PI * r + 2 * PI * h, 2 * PI * r],
    gradG: (r, h) => [2 * PI * r * h, PI * r * r],
    curve: (r) => [r, volume / (PI * r * r)],
    tRange: [0.5, 3],
    xRange: [0, 3],
    yRange: [0, 4],
    levels: [4, 8, 13, 19, 26, 34, 43, 53],
    goal: 'min',
  }
}

const radius = (v: number) => Math.cbrt(v / (2 * PI))
const optimum = (v: number) => 6 * PI * radius(v) ** 2
const multiplier = (v: number) => 2 / radius(v)

export function CanOfFixedVolume() {
  const state = useFigureState({
    t: tField([0.5, 3], 1.6, 0.005, 'radius r'),
    volume: slider(2, 12, 2 * PI, { step: 0.01, label: 'volume V' }),
  })
  const p = useMemo(() => problem(state.volume), [state.volume])
  const r = radius(state.volume)
  return (
    <ConstrainedExplorer
      state={state}
      problem={p}
      title="The can with least surface area for its volume"
      caption="Left: contours of the surface area A(r, h) = 2πr² + 2πrh (grey) and the curve of cans with volume πr²h = V. Drag the point along the curve, or set the radius r. Right, top: the area along the curve against r. Right, bottom: the least area A*(V) against V; drag the vertical line or use the slider. The dashed tangent has slope λ = 2/r*, the extra area per unit of extra volume."
      xLabel="radius r"
      yLabel="height h"
      tLabel="radius r"
      tSymbol="r"
      fLabel="A"
      readout={({ point }) => (
        <>
          <Readout label="h / r" value={formatNumber(point[1] / point[0])} />
          <Readout label="optimum (r*, 2r*)" value={`(${formatNumber(r)}, ${formatNumber(2 * r)})`} />
          <Readout label="λ* = 2/r*" value={formatNumber(multiplier(state.volume))} />
        </>
      )}
      profileHeight={220}
      extra={
        <SensitivityPanel
          param={state.bind('volume')}
          optimum={optimum}
          multiplier={multiplier}
          xLabel="volume V"
          symbol="V"
          yLabel="A*(V)"
        />
      }
    />
  )
}
