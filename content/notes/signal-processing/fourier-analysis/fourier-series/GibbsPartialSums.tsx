import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const T = linspace(-0.5, 1.5, 2001)
/** Square wave of period 1: +1 on (0, ½), −1 on (½, 1). */
const square = (t: number) => {
  const u = t - Math.floor(t)
  return u < 0.5 ? 1 : -1
}
/** Partial Fourier sum with odd harmonics up to K: (4/π) Σ sin(2π k t)/k. */
const partial = (t: number, K: number) => {
  let s = 0
  for (let k = 1; k <= K; k += 2) s += Math.sin(2 * Math.PI * k * t) / k
  return (4 / Math.PI) * s
}

/**
 * Partial Fourier sums of a square wave. The error narrows toward the jumps as harmonics are added, but the overshoot
 * next to each jump stays near 9% of the jump: the Gibbs phenomenon.
 */
export function GibbsPartialSums() {
  const harmonics = useParam(7, { min: 1, max: 101, step: 2 })

  const r = useMemo(() => {
    const y = T.map((t) => partial(t, harmonics.value))
    // The peak just after the jump at t = 0 sits near t = 1/(2K); search the first quarter period.
    const peak = Math.max(...T.filter((t) => t > 0 && t < 0.25).map((t) => partial(t, harmonics.value)))
    return { y, peak }
  }, [harmonics.value])

  const series: XYSeries[] = [
    { name: 'square wave', type: 'line', x: T, y: T.map(square), slot: 0 },
    { name: `sum to harmonic ${harmonics.value}`, type: 'line', x: T, y: r.y, slot: 1 },
  ]

  return (
    <Interactive
      title="The Gibbs phenomenon"
      caption="Partial Fourier sums of a square wave with odd harmonics up to K. Adding harmonics squeezes the ripple toward the jumps, but the first peak beside each jump never shrinks: it tends to 1.179, an overshoot of about 9% of the jump from −1 to +1. The sums converge at every point away from the jumps, but not uniformly."
      controls={<ParamSlider label="highest harmonic K" param={harmonics} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="terms" value={(harmonics.value + 1) / 2} />
          <Readout label="peak value" value={formatNumber(r.peak)} />
          <Readout label="overshoot, % of jump" value={`${formatNumber((100 * (r.peak - 1)) / 2)}%`} />
          <Readout label="limit (2/π) Si(π)" value="1.1790" />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="t (periods)"
        yLabel="x(t)"
        xRange={[-0.5, 1.5]}
        yRange={[-1.4, 1.4]}
        height={300}
      />
    </Interactive>
  )
}
