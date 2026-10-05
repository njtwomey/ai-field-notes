import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { applyFilter, iirLowpass } from '../_shared/design'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 300

/**
 * Forward–backward filtering: filter, reverse, filter again, reverse. Odd reflection padding at both ends, as in
 * scipy.signal.filtfilt, reduces the start-up transients.
 */
function filtfilt(b: number[], a: number[], x: number[]): number[] {
  const pad = Math.min(3 * Math.max(a.length, b.length), x.length - 1)
  const head = Array.from({ length: pad }, (_, i) => 2 * x[0] - x[pad - i])
  const tail = Array.from({ length: pad }, (_, i) => 2 * x[x.length - 1] - x[x.length - 2 - i])
  const ext = [...head, ...x, ...tail]
  const forward = applyFilter(b, a, ext).reverse()
  const backward = applyFilter(b, a, forward).reverse()
  return backward.slice(pad, pad + x.length)
}

/** A step and a pulse in noise, filtered causally and forward–backward with the same Butterworth filter. */
export function ZeroPhaseDemo() {
  const state = useFigureState({
    order: int(4, { min: 1, max: 8, step: 1, label: 'Butterworth order', format: (v) => String(v) }),
    cutoff: slider(0.02, 0.4, 0.08, { step: 0.01, label: 'cutoff (×π)' }),
  })

  const r = useMemo(() => {
    const g = stream(5)
    const clean = Array.from({ length: N }, (_, n) => (n >= 80 ? 1 : 0) + Math.exp(-0.5 * ((n - 200) / 6) ** 2))
    const x = clean.map((v) => v + 0.15 * normal(g))
    const f = iirLowpass('butter', state.order, state.cutoff * Math.PI)
    const causal = applyFilter(f.b, f.a, x)
    const zero = filtfilt(f.b, f.a, x)
    // Where each output crosses half the step height: the causal filter's crossing is delayed.
    const cross = (y: number[]) => y.findIndex((v, n) => n > 60 && v > 0.5)
    return { x, clean, causal, zero, delay: cross(causal) - cross(zero) }
  }, [state.order, state.cutoff])

  const n = Array.from({ length: N }, (_, i) => i)
  const series = [
    { name: 'noisy input', x: n, y: r.x, muted: true },
    { name: 'causal (filter once)', x: n, y: r.causal, slot: 2 },
    { name: 'zero-phase (forward–backward)', x: n, y: r.zero, slot: 0 },
    { name: 'clean signal', x: n, y: r.clean, slot: 1, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'n', range: [0, N - 1] })
  const yAxis = useAxis({ label: 'amplitude', range: [-0.5, 2] })
  return (
    <Figure
      title="Forward–backward filtering removes the delay"
      state={state}
      caption="A step at n = 80 and a pulse at n = 200 in noise. Filtering once with a Butterworth low-pass filter smooths the noise but delays and distorts the edges, because the filter's phase is not linear. Filtering forward, then backward in time, cancels the phase: the edges stay where they were and the pulse stays symmetric. The price is a squared magnitude response and a non-causal filter, usable only on recorded data."

      readouts={
        <>
          <Readout label="step delay, causal vs zero-phase" value={`${r.delay} samples`} />
          <Readout label="effective order" value={`${2 * state.order} (magnitude squared)`} />
          <Readout label="gain at cutoff, zero-phase" value={`${formatNumber(-6.02)} dB`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
