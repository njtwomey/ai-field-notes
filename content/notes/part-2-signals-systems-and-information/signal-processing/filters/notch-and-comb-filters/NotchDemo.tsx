import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { applyFilter, db, response as freqz } from '../_shared/design'

const FS = 1000
const N = 600
const SHOW = 400

type Mode = 'notch' | 'comb'

/**
 * Removing mains hum sampled at 1 kHz: a second-order notch at one frequency, or a comb that notches every harmonic
 * of 50 Hz. The signal of interest is a 7 Hz sinusoid.
 */
export function NotchDemo() {
  const state = useFigureState({
    mode: choice<Mode>(
      [
        { value: 'notch', label: 'single notch' },
        { value: 'comb', label: 'comb (all harmonics)' },
      ],
      'notch',
      { label: 'filter' },
    ),
    f0: int(50, { min: 10, max: 400, step: 1, label: 'notch frequency f₀ (Hz)', format: (v) => `${v} Hz` }),
    r: float(0.98, { min: 0.8, max: 0.999, step: 0.001, label: 'pole radius r', format: (v) => v.toFixed(3) }),
  })

  const out = useMemo(() => {
    const w0 = (2 * Math.PI * state.f0) / FS
    let b: number[]
    let a: number[]
    if (state.mode === 'notch') {
      b = [1, -2 * Math.cos(w0), 1]
      a = [1, -2 * state.r * Math.cos(w0), state.r * state.r]
      // Normalise the gain at DC to 1.
      const g = (a[0] + a[1] + a[2]) / (b[0] + b[1] + b[2])
      b = b.map((v) => v * g)
    } else {
      // Comb notch: zeros at every multiple of f0 (including DC), poles just inside at radius r.
      const D = Math.round(FS / state.f0)
      b = new Array<number>(D + 1).fill(0)
      a = new Array<number>(D + 1).fill(0)
      b[0] = 1
      b[D] = -1
      a[0] = 1
      a[D] = -(state.r ** D)
      const g = (1 + state.r ** D) / 2
      b = b.map((v) => v * g)
    }
    const resp = freqz(b, a, 2048)
    const hz = resp.omega.map((w) => (w * FS) / (2 * Math.PI))
    const magDb = resp.magnitude.map((m) => db(m, -80))
    // −3 dB bandwidth of the notch around f0 (single-notch mode).
    let bw = NaN
    if (state.mode === 'notch') {
      const i0 = hz.findIndex((f) => f >= state.f0)
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
    const y = applyFilter(b, a, x)
    const n = Array.from({ length: SHOW }, (_, i) => N - SHOW + i)
    return { hz, magDb, bw, n, x: n.map((i) => x[i]), y: n.map((i) => y[i]), s: n.map((i) => signal[i]) }
  }, [state.mode, state.f0, state.r])

  const freq = [{ name: '|H| (dB)', x: out.hz, y: out.magDb, slot: 0 }] as const
  const time = [
    { name: '7 Hz signal + 50 and 150 Hz hum', x: out.n, y: out.x, muted: true },
    { name: 'filtered', x: out.n, y: out.y, slot: 0 },
    { name: '7 Hz signal alone', x: out.n, y: out.s, slot: 1, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [0, 500] })
  const yAxis = useAxis({ label: 'magnitude (dB)', range: [-80, 5] })
  const xAxis2 = useAxis({ label: 'sample n', range: [N - SHOW, N - 1] })
  const yAxis2 = useAxis({ label: 'amplitude', range: [-2.5, 2.5] })
  return (
    <Figure
      title="Notching out mains hum"
      state={state}
      caption="Sampled at 1 kHz, a 7 Hz signal is contaminated by 50 Hz mains hum and its third harmonic at 150 Hz. A notch filter puts a pair of zeros on the unit circle at f₀ and a pair of poles just inside at radius r; the closer r is to 1, the narrower the notch. Drag f₀ to 50 Hz to remove the fundamental; the 150 Hz harmonic remains. The comb filter notches every multiple of f₀ at once, including 0 Hz, which also removes any constant offset."

      readouts={
        <>
          <Readout label="ω₀ (×π)" value={formatNumber((2 * state.f0) / FS)} />
          {state.mode === 'notch' && <Readout label="−3 dB bandwidth" value={`${formatNumber(out.bw)} Hz`} />}
          {state.mode === 'notch' && (
            <Readout label="approximation (1 − r)·f_s/π" value={`${formatNumber(((1 - state.r) * FS) / Math.PI)} Hz`} />
          )}
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={240}>
        <Curve {...freq[0]} />
        <Handle kind="x" at={state.f0} label="f₀" onDrag={(v) => state.set('f0', Math.round(v))} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={240}>
        <Curve {...time[0]} />
        <Curve {...time[1]} />
        <Curve {...time[2]} />
      </Plot>
    </Figure>
  )
}
