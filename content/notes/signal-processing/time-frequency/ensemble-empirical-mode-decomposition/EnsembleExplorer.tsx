import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { ceemdan, eemd, emd, type Decomposition } from '../_shared/emd'

const N = 512
const TIME = Array.from({ length: N }, (_, n) => n)
const SHOWN = 4

// A slow sine with two short bursts of a fast one: the classic cause of mode mixing.
const SLOW = Float64Array.from(TIME, (n) => Math.sin(2 * Math.PI * 0.01 * n))
const BURST = Float64Array.from(TIME, (n) =>
  (n > 150 && n < 250) || (n > 380 && n < 420) ? 0.3 * Math.sin(2 * Math.PI * 0.2 * n) : 0,
)
const X = Float64Array.from(TIME, (n) => SLOW[n] + BURST[n])

type Method = 'emd' | 'eemd' | 'ceemdan'

const relErr = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let num = 0
  let den = 0
  for (let i = 0; i < b.length; i++) {
    num += (a[i] - b[i]) ** 2
    den += b[i] ** 2
  }
  return Math.sqrt(num / den)
}

/** Index of the IMF closest to a target component, and its relative error. */
function bestMatch(imfs: Float64Array[], target: Float64Array) {
  let best = { index: 0, error: Infinity }
  imfs.forEach((c, index) => {
    const error = relErr(c, target)
    if (error < best.error) best = { index, error }
  })
  return best
}

/**
 * EMD, ensemble EMD and CEEMDAN on an intermittent signal. Plain EMD mixes the bursts and the slow sine in its first
 * IMF; adding noise and averaging over an ensemble separates them.
 */
export function EnsembleExplorer() {
  const [method, setMethod] = useState<Method>('eemd')
  const epsilon = useParam(0.2, { min: 0.02, max: 0.6, step: 0.02 })
  const trials = useParam(50, { min: 10, max: 100, step: 10 })

  const r = useMemo(() => {
    let d: Decomposition
    if (method === 'emd') d = emd(X)
    else if (method === 'eemd') d = eemd(X, { trials: trials.value, epsilon: epsilon.value, maxImfs: 7, seed: 7 })
    else d = ceemdan(X, { trials: trials.value, epsilon: epsilon.value, maxImfs: 7, seed: 7 })
    const recon = Float64Array.from(X, (_, i) => d.imfs.reduce((s, c) => s + c[i], d.residue[i]))
    let rms = 0
    for (let i = 0; i < N; i++) rms += (recon[i] - X[i]) ** 2
    return {
      d,
      burst: bestMatch(d.imfs, BURST),
      slow: bestMatch(d.imfs, SLOW),
      rms: Math.sqrt(rms / N),
    }
  }, [method, epsilon.value, trials.value])

  const panels = r.d.imfs.slice(0, SHOWN).map((c, i) => {
    const series: XYSeries[] = []
    if (i === r.burst.index) series.push({ name: 'bursts', type: 'line', x: TIME, y: Array.from(BURST), muted: true })
    if (i === r.slow.index) series.push({ name: 'slow sine', type: 'line', x: TIME, y: Array.from(SLOW), muted: true })
    series.push({ name: `IMF ${i + 1}`, type: 'line', x: TIME, y: Array.from(c), slot: 0 })
    return { name: `IMF ${i + 1}`, series }
  })

  return (
    <Interactive
      title="Noise cures mode mixing"
      caption="The signal is a slow sine plus two short bursts of a fast sine. Grey lines show the true bursts and the true slow sine behind the IMF that matches each best. Plain EMD puts the bursts and pieces of the slow sine into the same first IMF. Ensemble EMD adds white noise of standard deviation ε times the signal's, decomposes each noisy copy with 10 sifts per IMF, and averages; the bursts then stay in one IMF and the slow sine in another. Its reconstruction error is the averaged noise, which falls as one over the square root of the ensemble size. CEEMDAN reconstructs the signal to rounding error."
      controls={
        <>
          <ParamChoice
            label="method"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'emd', label: 'EMD' },
              { value: 'eemd', label: 'EEMD' },
              { value: 'ceemdan', label: 'CEEMDAN' },
            ]}
          />
          <ParamSlider label="noise level ε" param={epsilon} />
          <ParamSlider label="ensemble size" param={trials} format={(v) => `${v}`} />
        </>
      }
      readout={
        <>
          <Readout
            label="bursts: best IMF, relative error"
            value={`${r.burst.index + 1}, ${formatNumber(r.burst.error)}`}
          />
          <Readout
            label="slow sine: best IMF, relative error"
            value={`${r.slow.index + 1}, ${formatNumber(r.slow.error)}`}
          />
          <Readout label="reconstruction RMS error" value={r.rms < 1e-12 ? '< 1e-12' : formatNumber(r.rms)} />
        </>
      }
    >
      <div className="min-w-0 space-y-1">
        <div className="text-center text-xs text-muted-foreground">signal</div>
        <XYChart series={[{ name: 'x', type: 'line', x: TIME, y: Array.from(X) }]} xRange={[0, N - 1]} height={150} />
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {panels.map((p) => (
          <div key={p.name} className="min-w-0">
            <div className="text-center text-xs text-muted-foreground">{p.name}</div>
            <XYChart series={p.series} xRange={[0, N - 1]} height={150} />
          </div>
        ))}
      </div>
    </Interactive>
  )
}
