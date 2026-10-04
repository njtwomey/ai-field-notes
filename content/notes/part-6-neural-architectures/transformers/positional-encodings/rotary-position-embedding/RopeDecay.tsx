import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

const MAX_DISTANCE = 2000
const DISTANCES = Array.from({ length: 401 }, (_, i) => (i * MAX_DISTANCE) / 400)

/**
 * For each distance r: the mean of |S_j| over j = 1..d/2, with S_j = Σ_{i<j} exp(i r θ_i) (Su et al.'s decay bound),
 * normalised to 1 at r = 0, and the score of a query with itself, Σ_i cos(r θ_i), divided by d/2.
 */
function curves(d: number, base: number) {
  const half = d / 2
  const theta = Array.from({ length: half }, (_, i) => base ** ((-2 * i) / d))
  const bound: number[] = []
  const self: number[] = []
  for (const r of DISTANCES) {
    let re = 0
    let im = 0
    let sumAbs = 0
    for (const t of theta) {
      re += Math.cos(r * t)
      im += Math.sin(r * t)
      sumAbs += Math.hypot(re, im)
    }
    // Normalised so the bound is 1 at distance 0, where |S_j| = j and the mean is (d/2 + 1) / 2.
    bound.push(sumAbs / half / ((half + 1) / 2))
    self.push(re / half)
  }
  return { bound, self, wavelength: 2 * Math.PI * base ** ((d - 2) / d) }
}

/** Long-term decay of RoPE: how the bound and the self-score fall with distance, for a chosen base and head width. */
export function RopeDecay() {
  const state = useFigureState({
    logBase: float(4, {
      min: 2,
      max: 6.5,
      step: 0.1,
      label: 'base',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => Math.round(10 ** v).toLocaleString(),
    }),
    d: int(128, { min: 16, max: 256, step: 16, label: 'head width d' }),
  })
  const base = 10 ** state.logBase
  const r = useMemo(() => curves(state.d, base), [state.d, base])
  const series = [
    { name: 'decay bound (mean |S_j|, normalised)', x: DISTANCES, y: r.bound, slot: 0 },
    { name: 'score of q with itself, divided by d/2', x: DISTANCES, y: r.self, slot: 1 },
  ] as const
  const xAxis = useAxis({ label: 'distance m − n', hold: 'union' })
  const yAxis = useAxis({ label: 'relative size', range: [-0.2, 1] })
  return (
    <Figure
      title="Long-term decay of rotary scores"
      state={state}
      caption="The decay bound is the mean of |S_j| from Su et al.'s argument, normalised to 1 at distance 0; it bounds how large a rotary score can be at each distance. The second curve is the rotated score of a query with an identical key at distance r, when every pair contributes equally. Both fall with distance, with ripples. A larger base slows every rotation, so the fall is slower and reaches further."

      readouts={<Readout label="longest wavelength (tokens)" value={formatNumber(Math.round(r.wavelength))} />}
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
