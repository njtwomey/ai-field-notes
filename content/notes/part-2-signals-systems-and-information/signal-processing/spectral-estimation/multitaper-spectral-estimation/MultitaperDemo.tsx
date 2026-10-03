import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, useParam, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { arProcess, arSpectrum, multitaper, periodogram, powerDb } from '../_shared/spectra'

const N = 128
const OMEGA = linspace(0, Math.PI, 257)

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
  const nw = useParam(4, { min: 1, max: 6, step: 0.5 })
  const k = useParam(7, { min: 1, max: 11, step: 1 })
  const seed = useParam(2, { min: 1, max: 30, step: 1 })
  const tapers = Math.min(k.value, Math.max(1, Math.floor(2 * nw.value - 1)))

  const r = useMemo(() => {
    const x = arProcess(A, N, seed.value)
    const mt = multitaper(x, nw.value, tapers, 256)
    const raw = periodogram(x, 'rectangular', 256)
    return {
      mt: mt.psd.map((v) => powerDb(v)),
      raw: raw.psd.map((v) => powerDb(v)),
      omega: mt.omega.map((v) => v / Math.PI),
      tapers: mt.tapers,
    }
  }, [nw.value, tapers, seed.value])

  const spectra: XYSeries[] = [
    { name: 'periodogram (rectangular)', type: 'line', x: r.omega, y: r.raw, muted: true },
    { name: 'multitaper estimate', type: 'line', x: r.omega, y: r.mt, slot: 0 },
    {
      name: 'true spectrum',
      type: 'line',
      x: OMEGA.map((v) => v / Math.PI),
      y: TRUE.map((v) => powerDb(v)),
      slot: 1,
      dashed: true,
    },
  ]
  const taperSeries: XYSeries[] = r.tapers.map((v, i) => ({
    name: i < 7 ? `taper ${i}` : 'higher tapers',
    type: 'line',
    x: v.map((_, n) => n),
    y: v,
    ...(i < 7 ? { slot: i } : { muted: true }),
  }))

  return (
    <Interactive
      title="Averaging over orthogonal tapers"
      caption="An AR(4) process of N = 128 samples with a sharp peak and a weaker broad one. The multitaper estimate averages the eigenspectra of K Slepian tapers with time-bandwidth product NW, whose bandwidth W = NW/N sets the resolution. Up to 2NW − 1 tapers keep their energy concentrated in the band; the slider stops there. More tapers lower the variance, a wider bandwidth smooths the peaks, and the low sidelobes of the tapers keep the weak parts of the spectrum clear of leakage, unlike the raw periodogram (grey)."
      controls={
        <>
          <ParamSlider label="time-bandwidth product NW" param={nw} withArrows />
          <ParamSlider label="tapers K (capped at 2NW − 1)" param={k} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="tapers used" value={tapers} />
          <Readout label="half-bandwidth 2πW (rad/sample, ×π)" value={((2 * nw.value) / N).toFixed(4)} />
          <Readout label="degrees of freedom ≈ 2K" value={2 * tapers} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={spectra} xLabel="ω / π" yLabel="power (dB)" xRange={[0, 1]} yRange={[-30, 50]} height={300} />
        <XYChart series={taperSeries} xLabel="n" yLabel="taper v_k[n]" xRange={[0, N - 1]} height={220} />
      </div>
    </Interactive>
  )
}
