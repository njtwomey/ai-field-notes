import { useMemo } from 'react'
import {
  Curve,
  Figure,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { arProcess, arSpectrum, multitaper, periodogram, powerDb } from '../_shared/spectra'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const N = 128
const OMEGA = toFlat(linspace(0, Math.PI, 257))

/** AR coefficients (x[n] = Σ a_k x[n−k] + e[n]) with pole pairs r e^{±iθ}. */
function arFromPoles(pairs: [number, number][]): number[] {
  let p = [1]
  for (const [r, theta] of pairs) {
    const q = [1, -2 * r * Math.cos(theta), r * r]
    const next = new Array<number>(p.length + 2).fill(0)
    p.forEach((pv, i) => q.forEach((qv, j) => (next[i + j] += pv * qv)))
    p = next
  }
  return p.slice(1).map((v) => -v)
}

// A sharp peak and a weaker, broader one: a spectrum with about 50 dB of dynamic range.
const A = arFromPoles([
  [0.98, 0.25 * Math.PI],
  [0.85, 0.65 * Math.PI],
])
const TRUE = arSpectrum(A, 1, OMEGA)

/** Thomson's multitaper estimate against the raw periodogram, with the Slepian tapers it uses. */
export function MultitaperDemo() {
  const state = useFigureState({
    nw: slider(1, 6, 4, { step: 0.5, label: 'time-bandwidth product NW' }),
    k: int(7, { min: 1, max: 11, step: 1, label: 'tapers K (capped at 2NW − 1)', format: (v) => String(v) }),
    seed: int(2, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const tapers = Math.min(state.k, Math.max(1, Math.floor(2 * state.nw - 1)))

  const r = useMemo(() => {
    const x = arProcess(A, N, state.seed)
    const mt = multitaper(x, state.nw, tapers, 256)
    const raw = periodogram(x, 'rectangular', 256)
    return {
      mt: mt.psd.map((v) => powerDb(v)),
      raw: raw.psd.map((v) => powerDb(v)),
      omega: mt.omega.map((v) => v / Math.PI),
      tapers: mt.tapers,
    }
  }, [state.nw, tapers, state.seed])

  const spectra = [
    { name: 'periodogram (rectangular)', x: r.omega, y: r.raw, muted: true },
    { name: 'multitaper estimate', x: r.omega, y: r.mt, slot: 0 },
    {
      name: 'true spectrum',
      x: OMEGA.map((v) => v / Math.PI),
      y: TRUE.map((v) => powerDb(v)),
      slot: 1,
      dashed: true,
    },
  ] as const
  const taperSeries: SeriesSpec[] = r.tapers.map((v, i) => ({
    name: i < 7 ? `taper ${i}` : 'higher tapers',
    type: 'line',
    x: v.map((_, n) => n),
    y: v,
    ...(i < 7 ? { slot: i } : { muted: true }),
  }))

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'power (dB)', range: [-30, 50] })
  const xAxis2 = useAxis({ label: 'n', range: [0, N - 1] })
  const yAxis2 = useAxis({ label: 'taper v_k[n]', hold: 'union' })
  return (
    <Figure
      title="Averaging over orthogonal tapers"
      state={state}
      caption="An AR(4) process of N = 128 samples with a sharp peak and a weaker broad one. The multitaper estimate averages the eigenspectra of K Slepian tapers with time-bandwidth product NW, whose bandwidth W = NW/N sets the resolution. Up to 2NW − 1 tapers keep their energy concentrated in the band; the slider stops there. More tapers lower the variance, a wider bandwidth smooths the peaks, and the low sidelobes of the tapers keep the weak parts of the spectrum clear of leakage, unlike the raw periodogram (grey)."

      readouts={
        <>
          <Readout label="tapers used" value={tapers} />
          <Readout label="half-bandwidth 2πW (rad/sample, ×π)" value={((2 * state.nw) / N).toFixed(4)} />
          <Readout label="degrees of freedom ≈ 2K" value={2 * tapers} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...spectra[0]} />
          <Curve {...spectra[1]} />
          <Curve {...spectra[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={220}>
          {seriesLayers(taperSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
