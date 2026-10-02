import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { erf } from '@/lib/math/special'

const DX = 0.05
const X = linspace(-3, 3, Math.round(6 / DX) + 1)
const HALF = 0.25
const U0 = X.map((x): number => (Math.abs(x) < HALF ? 1 : 0))
const FINE = linspace(-3, 3, 601)

/** Exact solution for a box of half-width a on the whole line (D = 1): the box convolved with a Gaussian. */
const exact = (x: number, t: number) =>
  t <= 0
    ? Math.abs(x) < HALF
      ? 1
      : 0
    : 0.5 * (erf((x + HALF) / Math.sqrt(4 * t)) - erf((x - HALF) / Math.sqrt(4 * t)))

/** Explicit finite differences for u_t = u_xx: uⱼ ← uⱼ + r(uⱼ₊₁ − 2uⱼ + uⱼ₋₁), with the ends held at 0. */
export function FiniteDifference() {
  const r = useParam(0.4, { min: 0.1, max: 0.7, step: 0.02 })
  const steps = useParam(40, { min: 0, max: 200, step: 1 })

  const { u, t } = useMemo(() => {
    let cur = U0.slice()
    for (let n = 0; n < steps.value; n++) {
      const next = cur.slice()
      for (let j = 1; j < cur.length - 1; j++) next[j] = cur[j] + r.value * (cur[j + 1] - 2 * cur[j] + cur[j - 1])
      next[0] = 0
      next[cur.length - 1] = 0
      cur = next
    }
    return { u: cur, t: steps.value * r.value * DX * DX }
  }, [r.value, steps.value])

  const series = useMemo<XYSeries[]>(
    () => [
      { name: 'exact', type: 'line', x: FINE, y: FINE.map((x) => exact(x, t)), emphasis: true, dashed: true },
      { name: 'finite differences', type: 'line', x: X, y: u, slot: 0 },
      { name: 'grid values', type: 'scatter', x: X, y: u, slot: 0 },
    ],
    [u, t],
  )
  const maxAbs = Math.max(...u.map(Math.abs))

  return (
    <Interactive
      title="The explicit scheme and its stability limit"
      caption="A box of heat evolved by the explicit finite-difference scheme on a grid with Δx = 0.05, against the exact solution (dashed). The ratio r = D Δt / Δx² sets the time step. For r ≤ ½ each new value is an average of three old values with non-negative weights, and the scheme is stable. Push r above ½ and step forward: a saw-tooth pattern on the grid scale grows until it swamps the solution."
      controls={
        <>
          <ParamSlider label="r = D Δt / Δx²" param={r} />
          <ParamSlider label="time steps n" param={steps} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="time t = n r Δx²" value={formatNumber(t)} />
          <Readout label="max |u|" value={maxAbs > 1e4 ? maxAbs.toExponential(1) : formatNumber(maxAbs)} />
          <Readout label="saw-tooth multiplier 1 − 4r" value={formatNumber(1 - 4 * r.value)} />
        </>
      }
    >
      <XYChart height={300} xLabel="x" yLabel="u" series={series} xRange={[-1.5, 1.5]} yRange={[-0.6, 1.4]} />
    </Interactive>
  )
}
