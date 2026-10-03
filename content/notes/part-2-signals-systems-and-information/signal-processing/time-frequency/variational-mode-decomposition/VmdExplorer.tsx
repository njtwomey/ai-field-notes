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
} from 'aifn-render'
import { rng } from '@/lib/math'
import { vmd } from '../_shared/emd'

const N = 512
const MAX_ITER = 150
const TIME = Array.from({ length: N }, (_, n) => n)

type Choice = 'three' | 'close' | 'noisy'

/** Test signals in cycles per sample. */
function makeSignal(choice: Choice): Float64Array {
  const tone = (f: number, n: number) => Math.cos(2 * Math.PI * f * n)
  if (choice === 'close') return Float64Array.from(TIME, (n) => tone(0.1, n) + tone(0.12, n))
  const r = rng(5)
  const noise = choice === 'noisy' ? 0.4 : 0
  return Float64Array.from(TIME, (n) => tone(0.02, n) + 0.5 * tone(0.1, n) + 0.3 * tone(0.3, n) + noise * r.normal())
}

/**
 * VMD iteration by iteration: each mode's spectrum is the residual spectrum through a Wiener-like filter centred on
 * the mode's frequency, and each centre frequency moves to the centre of gravity of its mode's power.
 */
export function VmdExplorer() {
  const [choice, setChoice] = useState<Choice>('three')
  const K = useParam(3, { min: 1, max: 5, step: 1 })
  const logAlpha = useParam(3.3, { min: 1, max: 4.5, step: 0.1 })
  const iter = useParam(3, { min: 1, max: MAX_ITER, step: 1 })
  const alpha = 10 ** logAlpha.value

  const x = useMemo(() => makeSignal(choice), [choice])

  // Full run, for the centre-frequency trajectories and the convergence point.
  const full = useMemo(
    () => vmd(x, { K: K.value, alpha, tol: 1e-7, maxIter: MAX_ITER + 1, init: 'uniform' }),
    [x, K.value, alpha],
  )
  // State after exactly `iter` iterations (tolerance 0 disables early stopping).
  const now = useMemo(
    () => vmd(x, { K: K.value, alpha, tol: 0, maxIter: iter.value + 1, init: 'uniform' }),
    [x, K.value, alpha, iter.value],
  )

  const omega = now.omega.at(-1) ?? []
  const peak = Math.max(...now.signalSpectrum)
  const freqs = Array.from(now.freqs)
  const spectrum: XYSeries[] = [
    {
      name: '|signal spectrum|',
      type: 'line',
      x: freqs,
      y: Array.from(now.signalSpectrum, (v) => v / peak),
      muted: true,
    },
    ...now.spectra.map((s, k): XYSeries => ({
      name: `mode ${k + 1}`,
      type: 'line',
      x: freqs,
      y: Array.from(s, (v) => v / peak),
      slot: k,
    })),
  ]

  const iterations = Array.from({ length: full.omega.length }, (_, i) => i)
  const trajectories: XYSeries[] = Array.from({ length: K.value }, (_, k) => ({
    name: `ω${k + 1}`,
    type: 'line',
    x: iterations,
    y: full.omega.map((o) => o[k]),
    slot: k,
  }))
  // Fit the iteration axis to the run, so fast convergence is not squashed against the left edge.
  const span = Math.min(MAX_ITER, Math.max(20, full.iterations + 5, iter.value + 2))
  const handles: Handle[] = [{ kind: 'x', at: iter.value, onDrag: iter.set, label: 'iteration' }]

  let err = 0
  let energy = 0
  for (let i = 0; i < N; i++) {
    const sum = now.modes.reduce((s, m) => s + m[i], 0)
    err += (x[i] - sum) ** 2
    energy += x[i] ** 2
  }

  return (
    <Interactive
      title="VMD, one ADMM iteration at a time"
      caption="Top left: the magnitude spectrum of the mirrored signal (grey) and of each mode after the chosen iteration, with a vertical line at each centre frequency. Top right: the centre frequencies over the iterations, starting spread evenly over [0, 0.5); drag the vertical line to step through them. Bottom: the modes. With the right number of modes the centre frequencies lock onto the tones within a few iterations. Too few modes and a tone is dropped (the widget runs without the reconstruction constraint, τ = 0); too many and a tone is split between two modes. VMD separates the two tones at 0.10 and 0.12 cycles per sample, which EMD returns as one IMF. A small α widens the filters and lets the noise into every mode."
      controls={
        <>
          <ParamChoice
            label="signal"
            value={choice}
            onChange={setChoice}
            options={[
              { value: 'three', label: 'three tones' },
              { value: 'close', label: 'close tones' },
              { value: 'noisy', label: 'three tones + noise' },
            ]}
          />
          <ParamSlider label="modes K" param={K} format={(v) => `${v}`} />
          <ParamSlider label="bandwidth penalty α" param={logAlpha} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="iteration" param={iter} withArrows format={(v) => `${v}`} />
        </>
      }
      readout={
        <>
          <Readout label="centre frequencies" value={omega.map((o) => o.toFixed(4)).join(', ')} />
          <Readout
            label="converged after"
            value={full.iterations < MAX_ITER ? `${full.iterations} iterations` : `> ${MAX_ITER} iterations`}
          />
          <Readout label="reconstruction error (relative RMS)" value={formatNumber(Math.sqrt(err / energy))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">spectra after iteration {iter.value}</div>
          <XYChart
            series={spectrum}
            segments={omega.map((o) => ({ from: [o, 0], to: [o, 1] }))}
            xRange={[0, 0.5]}
            yRange={[0, 1.05]}
            xLabel="frequency (cycles/sample)"
            height={260}
          />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">centre frequencies</div>
          <XYChart
            series={trajectories}
            handles={handles}
            xRange={[0, span]}
            yRange={[0, 0.5]}
            xLabel="iteration"
            yLabel="cycles/sample"
            height={260}
          />
        </div>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {now.modes.map((m, k) => (
          <div key={k} className="min-w-0">
            <div className="text-center text-xs text-muted-foreground">mode {k + 1}</div>
            <XYChart
              series={[{ name: `mode ${k + 1}`, type: 'line', x: TIME, y: Array.from(m), slot: k }]}
              xRange={[0, N - 1]}
              height={150}
            />
          </div>
        ))}
      </div>
    </Interactive>
  )
}
