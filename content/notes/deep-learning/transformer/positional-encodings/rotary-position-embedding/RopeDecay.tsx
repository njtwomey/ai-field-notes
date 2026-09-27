import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

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
  const logBase = useParam(4, { min: 2, max: 6.5, step: 0.1 })
  const d = useParam(128, { min: 16, max: 256, step: 16 })
  const base = 10 ** logBase.value
  const r = useMemo(() => curves(d.value, base), [d.value, base])
  const series: XYSeries[] = [
    { name: 'decay bound (mean |S_j|, normalised)', type: 'line', x: DISTANCES, y: r.bound, slot: 0 },
    { name: 'score of q with itself, divided by d/2', type: 'line', x: DISTANCES, y: r.self, slot: 1 },
  ]
  return (
    <Interactive
      title="Long-term decay of rotary scores"
      caption="The decay bound is the mean of |S_j| from Su et al.'s argument, normalised to 1 at distance 0; it bounds how large a rotary score can be at each distance. The second curve is the rotated score of a query with an identical key at distance r, when every pair contributes equally. Both fall with distance, with ripples. A larger base slows every rotation, so the fall is slower and reaches further."
      controls={
        <>
          <ParamSlider label="base" param={logBase} format={(v) => Math.round(10 ** v).toLocaleString()} />
          <ParamSlider label="head width d" param={d} />
        </>
      }
      readout={<Readout label="longest wavelength (tokens)" value={formatNumber(Math.round(r.wavelength))} />}
    >
      <XYChart series={series} xLabel="distance m − n" yLabel="relative size" yRange={[-0.2, 1]} height={300} />
    </Interactive>
  )
}
