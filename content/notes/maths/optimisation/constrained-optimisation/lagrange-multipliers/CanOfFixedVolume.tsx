import { useMemo } from 'react'
import { ParamSlider, Readout, formatNumber, useParam } from '@/components/viz'
import { ConstrainedExplorer, SensitivityPanel, type ConstrainedProblem } from './ConstrainedExplorer'

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
  const volume = useParam(2 * PI, { min: 2, max: 12, step: 0.01 })
  const p = useMemo(() => problem(volume.value), [volume.value])
  const r = radius(volume.value)
  return (
    <ConstrainedExplorer
      problem={p}
      title="The can with least surface area for its volume"
      caption="Left: contours of the surface area A(r, h) = 2πr² + 2πrh (grey) and the curve of cans with volume πr²h = V. Drag the point along the curve, or set the radius r. Right, top: the area along the curve against r. Right, bottom: the least area A*(V) against V; drag the vertical line or use the slider. The dashed tangent has slope λ = 2/r*, the extra area per unit of extra volume."
      initialT={1.6}
      tStep={0.005}
      xLabel="radius r"
      yLabel="height h"
      tLabel="radius r"
      tSymbol="r"
      fLabel="A"
      controls={<ParamSlider label="volume V" param={volume} />}
      readout={({ point }) => (
        <>
          <Readout label="h / r" value={formatNumber(point[1] / point[0])} />
          <Readout label="optimum (r*, 2r*)" value={`(${formatNumber(r)}, ${formatNumber(2 * r)})`} />
          <Readout label="λ* = 2/r*" value={formatNumber(multiplier(volume.value))} />
        </>
      )}
      profileHeight={220}
      extra={
        <SensitivityPanel
          param={volume}
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
