import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { db, freqz, lfilter } from '@/lib/dsp'

const FS = 1000
const N = 600
const SHOW = 400

type Mode = 'notch' | 'comb'

/**
 * Removing mains hum sampled at 1 kHz: a second-order notch at one frequency, or a comb that notches every harmonic
 * of 50 Hz. The signal of interest is a 7 Hz sinusoid.
 */
export function NotchDemo() {
  const [mode, setMode] = useState<Mode>('notch')
  const f0 = useParam(50, { min: 10, max: 400, step: 1 })
  const r = useParam(0.98, { min: 0.8, max: 0.999, step: 0.001 })

  const out = useMemo(() => {
    const w0 = (2 * Math.PI * f0.value) / FS
    let b: number[]
    let a: number[]
    if (mode === 'notch') {
      b = [1, -2 * Math.cos(w0), 1]
      a = [1, -2 * r.value * Math.cos(w0), r.value * r.value]
      // Normalise the gain at DC to 1.
      const g = (a[0] + a[1] + a[2]) / (b[0] + b[1] + b[2])
      b = b.map((v) => v * g)
    } else {
      // Comb notch: zeros at every multiple of f0 (including DC), poles just inside at radius r.
      const D = Math.round(FS / f0.value)
      b = new Array<number>(D + 1).fill(0)
      a = new Array<number>(D + 1).fill(0)
      b[0] = 1
      b[D] = -1
      a[0] = 1
      a[D] = -(r.value ** D)
      const g = (1 + r.value ** D) / 2
      b = b.map((v) => v * g)
    }
    const resp = freqz(b, a, 2048)
    const hz = resp.omega.map((w) => (w * FS) / (2 * Math.PI))
    const magDb = resp.magnitude.map((m) => db(m, -80))
    // −3 dB bandwidth of the notch around f0 (single-notch mode).
    let bw = NaN
    if (mode === 'notch') {
      const i0 = hz.findIndex((f) => f >= f0.value)
      let lo = i0
      let hi = i0
      while (lo > 0 && resp.magnitude[lo] < Math.SQRT1_2) lo--
      while (hi < hz.length - 1 && resp.magnitude[hi] < Math.SQRT1_2) hi++
      bw = hz[hi] - hz[lo]
    }
    const t = Array.from({ length: N }, (_, n) => n / FS)
    const signal = t.map((s) => Math.sin(2 * Math.PI * 7 * s))
    const hum = t.map((s) => 0.8 * Math.sin(2 * Math.PI * 50 * s) + 0.4 * Math.sin(2 * Math.PI * 150 * s + 1))
    const x = signal.map((v, i) => v + hum[i])
    const y = lfilter(b, a, x)
    const n = Array.from({ length: SHOW }, (_, i) => N - SHOW + i)
    return { hz, magDb, bw, n, x: n.map((i) => x[i]), y: n.map((i) => y[i]), s: n.map((i) => signal[i]) }
  }, [mode, f0.value, r.value])

  const handles: Handle[] = [{ kind: 'x', at: f0.value, label: 'f₀', onDrag: (v) => f0.set(Math.round(v)) }]
  const freq: XYSeries[] = [{ name: '|H| (dB)', type: 'line', x: out.hz, y: out.magDb, slot: 0 }]
  const time: XYSeries[] = [
    { name: '7 Hz signal + 50 and 150 Hz hum', type: 'line', x: out.n, y: out.x, muted: true },
    { name: 'filtered', type: 'line', x: out.n, y: out.y, slot: 0 },
    { name: '7 Hz signal alone', type: 'line', x: out.n, y: out.s, slot: 1, dashed: true },
  ]

  return (
    <Interactive
      title="Notching out mains hum"
      caption="Sampled at 1 kHz, a 7 Hz signal is contaminated by 50 Hz mains hum and its third harmonic at 150 Hz. A notch filter puts a pair of zeros on the unit circle at f₀ and a pair of poles just inside at radius r; the closer r is to 1, the narrower the notch. Drag f₀ to 50 Hz to remove the fundamental; the 150 Hz harmonic remains. The comb filter notches every multiple of f₀ at once, including 0 Hz, which also removes any constant offset."
      controls={
        <>
          <ParamChoice
            label="filter"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'notch', label: 'single notch' },
              { value: 'comb', label: 'comb (all harmonics)' },
            ]}
          />
          <ParamSlider label="notch frequency f₀ (Hz)" param={f0} format={(v) => `${v} Hz`} withArrows />
          <ParamSlider label="pole radius r" param={r} format={(v) => v.toFixed(3)} />
        </>
      }
      readout={
        <>
          <Readout label="ω₀ (×π)" value={formatNumber((2 * f0.value) / FS)} />
          {mode === 'notch' && <Readout label="−3 dB bandwidth" value={`${formatNumber(out.bw)} Hz`} />}
          {mode === 'notch' && (
            <Readout label="approximation (1 − r)·f_s/π" value={`${formatNumber(((1 - r.value) * FS) / Math.PI)} Hz`} />
          )}
        </>
      }
    >
      <XYChart
        series={freq}
        xLabel="frequency (Hz)"
        yLabel="magnitude (dB)"
        xRange={[0, 500]}
        yRange={[-80, 5]}
        handles={handles}
        height={240}
      />
      <XYChart
        series={time}
        xLabel="sample n"
        yLabel="amplitude"
        xRange={[N - SHOW, N - 1]}
        yRange={[-2.5, 2.5]}
        height={240}
      />
    </Interactive>
  )
}
