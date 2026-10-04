import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { imagPart, realPart, toFlat } from 'aifn/foundation/tensor'
import { hilbert } from 'aifn/signal'

const FS = 1000
const N = 1024
const T = Array.from({ length: N }, (_, n) => n / FS)

/**
 * The analytic signal via the FFT (aifn's `hilbert`): keep DC and Nyquist, double the positive frequencies, zero the
 * negative ones, and invert (Marple 1999). Returns the real and imaginary parts; the imaginary part is the Hilbert
 * transform of x.
 */
function analytic(x: number[]) {
  const z = hilbert(x)
  return { re: toFlat(realPart(z)), im: toFlat(imagPart(z)) }
}

/** An amplitude-modulated tone; its envelope and instantaneous frequency recovered from the analytic signal. */
export function AnalyticEnvelope() {
  const state = useFigureState({
    carrier: int(60, { min: 20, max: 200, step: 5, label: 'carrier (Hz)' }),
    modulation: float(4, { min: 1, max: 80, step: 0.5, label: 'modulation (Hz)' }),
    depth: slider(0, 1, 0.6, { step: 0.05, label: 'modulation depth' }),
  })

  const r = useMemo(() => {
    const envelope = T.map((t) => 1 + state.depth * Math.cos(2 * Math.PI * state.modulation * t))
    const x = T.map((t, n) => envelope[n] * Math.cos(2 * Math.PI * state.carrier * t))
    const z = analytic(x)
    const magnitude = Array.from(z.re, (v, n) => Math.hypot(v, z.im[n]))
    // Instantaneous frequency: the derivative of the unwrapped phase, in Hz.
    const phase = Array.from(z.re, (v, n) => Math.atan2(z.im[n], v))
    const freq: number[] = []
    for (let n = 1; n < N; n++) {
      let d = phase[n] - phase[n - 1]
      if (d > Math.PI) d -= 2 * Math.PI
      if (d < -Math.PI) d += 2 * Math.PI
      freq.push((d * FS) / (2 * Math.PI))
    }
    // Ignore the edges, where the FFT's circularity distorts the estimate.
    const inner = magnitude.slice(100, N - 100).map((m, i) => Math.abs(m - envelope[i + 100]))
    return { x, envelope, magnitude, freq, err: Math.max(...inner) }
  }, [state.carrier, state.modulation, state.depth])

  const signal = [
    { name: 'x(t)', x: T, y: r.x, slot: 0 },
    { name: '|analytic signal|', x: T, y: r.magnitude, slot: 1 },
    { name: 'true envelope', x: T, y: r.envelope, slot: 2, dashed: true },
  ] as const
  const frequency = [{ name: 'instantaneous frequency', x: T.slice(1), y: r.freq, slot: 1 }] as const

  const xAxis = useAxis({ label: 't (s)', range: [0, N / FS] })
  const yAxis = useAxis({ label: 'amplitude', range: [-2.1, 2.1] })
  const xAxis2 = useAxis({ label: 't (s)', range: [0, N / FS] })
  const yAxis2 = useAxis({ label: 'Hz', range: [0, 250] })
  return (
    <Figure
      title="Envelope and instantaneous frequency"
      state={state}
      caption="An amplitude-modulated tone and its analytic signal, computed with the FFT. The magnitude of the analytic signal traces the envelope and the derivative of its phase gives the instantaneous frequency, here the carrier. They match to about 1% away from the ends of the segment, where the FFT's circularity distorts them. Once the modulation frequency exceeds the carrier, the lower sideband crosses zero frequency and the envelope estimate breaks down."

      readouts={
        <>
          <Readout label="max envelope error (interior)" value={formatNumber(r.err)} />
          <Readout label="carrier" value={`${state.carrier} Hz`} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...signal[0]} />
          <Curve {...signal[1]} />
          <Curve {...signal[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={160}>
          <Curve {...frequency[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
