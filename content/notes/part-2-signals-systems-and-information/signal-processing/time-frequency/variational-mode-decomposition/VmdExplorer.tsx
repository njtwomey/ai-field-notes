import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { vmd } from '../_shared/emd'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 512
const MAX_ITER = 150
const TIME = Array.from({ length: N }, (_, n) => n)

type SignalName = 'three' | 'close' | 'noisy'

/** Test signals in cycles per sample. */
function makeSignal(name: SignalName): Float64Array {
  const tone = (f: number, n: number) => Math.cos(2 * Math.PI * f * n)
  if (name === 'close') return Float64Array.from(TIME, (n) => tone(0.1, n) + tone(0.12, n))
  const r = stream(5)
  const noise = name === 'noisy' ? 0.4 : 0
  return Float64Array.from(TIME, (n) => tone(0.02, n) + 0.5 * tone(0.1, n) + 0.3 * tone(0.3, n) + noise * normal(r))
}

/** One mode on its own small chart. */
function ModePlot({ k, y }: { k: number; y: ArrayLike<number> }) {
  const xAxis = useAxis({ range: [0, N - 1] })
  const yAxis = useAxis({ hold: 'union' })
  const yy = useMemo(() => Array.from(y), [y])
  return (
    <Plot x={xAxis} y={yAxis} height={150}>
      <Curve name={`mode ${k + 1}`} x={TIME} y={yy} slot={k} />
    </Plot>
  )
}

/**
 * VMD iteration by iteration: each mode's spectrum is the residual spectrum through a Wiener-like filter centred on
 * the mode's frequency, and each centre frequency moves to the centre of gravity of its mode's power.
 */
export function VmdExplorer() {
  const state = useFigureState({
    signal: choice<SignalName>(
      [
        { value: 'three', label: 'three tones' },
        { value: 'close', label: 'close tones' },
        { value: 'noisy', label: 'three tones + noise' },
      ],
      'three',
      { label: 'signal' },
    ),
    K: int(3, { min: 1, max: 5, suggestions: [1, 2, 3, 4, 5], label: 'modes K' }),
    alpha: float(2000, {
      gt: 0,
      scale: 'log10',
      suggestions: [10, 100, 1000, 2000, 10000],
      label: 'bandwidth penalty α',
    }),
  })
  const alpha = state.alpha
  const K = state.K
  // The walk-through restarts at the first iteration when an input changes: the position remembers its inputs.
  const inputs = `${state.signal}:${K}:${alpha}`
  const [pos, setPos] = useState({ inputs, step: 0 })
  const step = pos.inputs === inputs ? pos.step : 0
  const iteration = step + 1

  const x = useMemo(() => makeSignal(state.signal), [state.signal])

  // Full run, for the centre-frequency trajectories and the convergence point.
  const full = useMemo(() => vmd(x, { K, alpha, tol: 1e-7, maxIter: MAX_ITER + 1, init: 'uniform' }), [x, K, alpha])
  // State after exactly `iteration` iterations (tolerance 0 disables early stopping).
  const now = useMemo(
    () => vmd(x, { K, alpha, tol: 0, maxIter: iteration + 1, init: 'uniform' }),
    [x, K, alpha, iteration],
  )

  const omega = now.omega.at(-1) ?? []
  const peak = Math.max(...now.signalSpectrum)
  const freqs = Array.from(now.freqs)
  const spectrum: SeriesSpec[] = [
    {
      name: '|signal spectrum|',
      type: 'line',
      x: freqs,
      y: Array.from(now.signalSpectrum, (v) => v / peak),
      muted: true,
    },
    ...now.spectra.map((s, k): SeriesSpec => ({
      name: `mode ${k + 1}`,
      type: 'line',
      x: freqs,
      y: Array.from(s, (v) => v / peak),
      slot: k,
    })),
  ]

  const iterations = Array.from({ length: full.omega.length }, (_, i) => i)
  const trajectories: SeriesSpec[] = Array.from({ length: K }, (_, k) => ({
    name: `ω${k + 1}`,
    type: 'line',
    x: iterations,
    y: full.omega.map((o) => o[k]),
    slot: k,
  }))
  // Fit the iteration axis to the run, so fast convergence is not squashed against the left edge.
  const span = Math.min(MAX_ITER, Math.max(20, full.iterations + 5, iteration + 2))

  let err = 0
  let energy = 0
  for (let i = 0; i < N; i++) {
    const sum = now.modes.reduce((s, m) => s + m[i], 0)
    err += (x[i] - sum) ** 2
    energy += x[i] ** 2
  }

  const xAxis = useAxis({ label: 'frequency (cycles/sample)', range: [0, 0.5] })
  const yAxis = useAxis({ range: [0, 1.05] })
  const xAxis2 = useAxis({ label: 'iteration', range: [0, span] })
  const yAxis2 = useAxis({ label: 'cycles/sample', range: [0, 0.5] })
  return (
    <Figure
      title="VMD, one ADMM iteration at a time"
      caption="Top left: the magnitude spectrum of the mirrored signal (grey) and of each mode after the chosen iteration, with a vertical line at each centre frequency. Top right: the centre frequencies over the iterations, starting spread evenly over [0, 0.5); drag the vertical line to step through them. Bottom: the modes. With the right number of modes the centre frequencies lock onto the tones within a few iterations. Too few modes and a tone is dropped (the widget runs without the reconstruction constraint, τ = 0); too many and a tone is split between two modes. VMD separates the two tones at 0.10 and 0.12 cycles per sample, which EMD returns as one IMF. A small α widens the filters and lets the noise into every mode."
      state={state}
      controls={
        <Player
          value={step}
          onChange={(k) => setPos({ inputs, step: k })}
          count={MAX_ITER}
          label="iteration"
          format={(k) => `${k + 1}`}
        />
      }
      readouts={
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
          <div className="text-center text-xs text-muted-foreground">spectra after iteration {iteration}</div>
          <Plot x={xAxis} y={yAxis} height={260}>
            {seriesLayers(spectrum)}
            <Segments segments={omega.map((o) => ({ from: [o, 0], to: [o, 1] }))} />
          </Plot>
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">centre frequencies</div>
          <Plot x={xAxis2} y={yAxis2} height={260}>
            {seriesLayers(trajectories)}
            <Handle
              kind="x"
              at={iteration}
              onDrag={(v) => setPos({ inputs, step: Math.min(MAX_ITER, Math.max(1, Math.round(v))) - 1 })}
              label="iteration"
            />
          </Plot>
        </div>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {now.modes.map((m, k) => (
          <div key={k} className="min-w-0">
            <div className="text-center text-xs text-muted-foreground">mode {k + 1}</div>
            <ModePlot k={k} y={m} />
          </div>
        ))}
      </div>
    </Figure>
  )
}
