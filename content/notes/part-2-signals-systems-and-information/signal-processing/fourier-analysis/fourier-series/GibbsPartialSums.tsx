import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const T = toFlat(linspace(-0.5, 1.5, 2001))
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
  const state = useFigureState({
    harmonics: int(7, { min: 1, max: 101, step: 2, label: 'highest harmonic K', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const y = T.map((t) => partial(t, state.harmonics))
    // The peak just after the jump at t = 0 sits near t = 1/(2K); search the first quarter period.
    const peak = Math.max(...T.filter((t) => t > 0 && t < 0.25).map((t) => partial(t, state.harmonics)))
    return { y, peak }
  }, [state.harmonics])

  const series = [
    { name: 'square wave', x: T, y: T.map(square), slot: 0 },
    { name: `sum to harmonic ${state.harmonics}`, x: T, y: r.y, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 't (periods)', range: [-0.5, 1.5] })
  const yAxis = useAxis({ label: 'x(t)', range: [-1.4, 1.4] })
  return (
    <Figure
      title="The Gibbs phenomenon"
      state={state}
      caption="Partial Fourier sums of a square wave with odd harmonics up to K. Adding harmonics squeezes the ripple toward the jumps, but the first peak beside each jump never shrinks: it tends to 1.179, an overshoot of about 9% of the jump from −1 to +1. The sums converge at every point away from the jumps, but not uniformly."

      readouts={
        <>
          <Readout label="terms" value={(state.harmonics + 1) / 2} />
          <Readout label="peak value" value={formatNumber(r.peak)} />
          <Readout label="overshoot, % of jump" value={`${formatNumber((100 * (r.peak - 1)) / 2)}%`} />
          <Readout label="limit (2/π) Si(π)" value="1.1790" />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
