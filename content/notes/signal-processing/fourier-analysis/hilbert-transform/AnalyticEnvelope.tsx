import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { fft, ifft } from '@/lib/dsp'

const FS = 1000
const N = 1024
const T = Array.from({ length: N }, (_, n) => n / FS)

/**
 * The analytic signal via the FFT: keep DC and Nyquist, double the positive frequencies, zero the negative ones, and
 * invert (Marple 1999). Returns the real and imaginary parts; the imaginary part is the Hilbert transform of x.
 */
function analytic(x: number[]) {
  const { re, im } = fft(x, N)
  for (let k = 1; k < N / 2; k++) {
    re[k] *= 2
    im[k] *= 2
  }
  for (let k = N / 2 + 1; k < N; k++) {
    re[k] = 0
    im[k] = 0
  }
  return ifft(re, im)
}

/** An amplitude-modulated tone; its envelope and instantaneous frequency recovered from the analytic signal. */
export function AnalyticEnvelope() {
  const carrier = useParam(60, { min: 20, max: 200, step: 5 })
  const modulation = useParam(4, { min: 1, max: 80, step: 0.5 })
  const depth = useParam(0.6, { min: 0, max: 1, step: 0.05 })

  const r = useMemo(() => {
    const envelope = T.map((t) => 1 + depth.value * Math.cos(2 * Math.PI * modulation.value * t))
    const x = T.map((t, n) => envelope[n] * Math.cos(2 * Math.PI * carrier.value * t))
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
  }, [carrier.value, modulation.value, depth.value])

  const signal: XYSeries[] = [
    { name: 'x(t)', type: 'line', x: T, y: r.x, slot: 0 },
    { name: '|analytic signal|', type: 'line', x: T, y: r.magnitude, slot: 1 },
    { name: 'true envelope', type: 'line', x: T, y: r.envelope, slot: 2, dashed: true },
  ]
  const frequency: XYSeries[] = [{ name: 'instantaneous frequency', type: 'line', x: T.slice(1), y: r.freq, slot: 1 }]

  return (
    <Interactive
      title="Envelope and instantaneous frequency"
      caption="An amplitude-modulated tone and its analytic signal, computed with the FFT. The magnitude of the analytic signal traces the envelope and the derivative of its phase gives the instantaneous frequency, here the carrier. They match to about 1% away from the ends of the segment, where the FFT's circularity distorts them. Once the modulation frequency exceeds the carrier, the lower sideband crosses zero frequency and the envelope estimate breaks down."
      controls={
        <>
          <ParamSlider label="carrier (Hz)" param={carrier} />
          <ParamSlider label="modulation (Hz)" param={modulation} />
          <ParamSlider label="modulation depth" param={depth} />
        </>
      }
      readout={
        <>
          <Readout label="max envelope error (interior)" value={formatNumber(r.err)} />
          <Readout label="carrier" value={`${carrier.value} Hz`} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart
          series={signal}
          xLabel="t (s)"
          yLabel="amplitude"
          xRange={[0, N / FS]}
          yRange={[-2.1, 2.1]}
          height={260}
        />
        <XYChart series={frequency} xLabel="t (s)" yLabel="Hz" xRange={[0, N / FS]} yRange={[0, 250]} height={160} />
      </div>
    </Interactive>
  )
}
