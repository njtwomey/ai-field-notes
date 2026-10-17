import { useMemo } from 'react'
import { rungeKutta } from 'aifn-compute/dynamics/ode'
import { trace } from 'aifn-compute/foundation/trace'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { amplitude, blobSheet, linearRate, perturbedSheet } from './sheet'

/** Frames kept for the player. */
const FRAMES = 120

const fmt = (v: number) => formatNumber(v)

/** Krasny's vortex-blob sheet rolling up, with its amplitude against the linear growth law. */
export function SheetRollUp() {
  const state = useFigureState({
    blobs: int(150, { label: 'blobs N', ge: 16, le: 400, suggestions: [64, 100, 150, 200, 300] }),
    delta: float(0.2, { label: 'smoothing δ', ge: 0, le: 1, suggestions: [0, 0.05, 0.1, 0.2, 0.3, 0.5] }),
    eps: float(0.01, { label: 'amplitude ε', gt: 0, le: 0.1, suggestions: [0.001, 0.005, 0.01, 0.03] }),
    duration: float(3, { label: 'duration t', gt: 0, le: 8, suggestions: [1, 2, 3, 4] }),
    dt: float(0.01, { label: 'RK4 step Δt', gt: 0, le: 0.05, suggestions: [0.005, 0.01, 0.02] }),
  })
  const { blobs, delta, eps, duration, dt } = state

  const run = useMemo(() => {
    const steps = Math.max(1, Math.round(duration / dt))
    const every = Math.max(1, Math.round(steps / FRAMES))
    const tr = trace(
      rungeKutta(blobSheet(blobs, delta), 'rk4', { stepSize: dt }),
      { x0: perturbedSheet(blobs, eps) },
      steps,
      {
        every,
        keep: 'all',
        timing: false,
        stopOnNonFinite: false,
      },
    )
    const frames = tr.steps.map((s) => toFlat(s.x))
    return { frames, times: tr.steps.map((s) => s.time), amps: frames.map(amplitude) }
  }, [blobs, delta, eps, duration, dt])

  const count = run.frames.length
  const [frame, setFrame] = usePlayhead(count)
  const at = Math.min(frame, count - 1)
  const z = run.frames[at]

  // The sheet over one period, closed by the first blob shifted one period to the right.
  const sheet = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    for (let j = 0; j <= blobs; j++) {
      const k = j % blobs
      x.push(z[2 * k] + (j === blobs ? 1 : 0))
      y.push(z[2 * k + 1])
    }
    return { x, y }
  }, [z, blobs])

  const sigma = linearRate(1, blobs, delta)
  const shortest = Math.floor(blobs / 2)
  const sigmaShort = linearRate(shortest, blobs, delta)
  const sigmaShortBare = linearRate(shortest, blobs, 0)
  const growth = useMemo(() => {
    const t = run.times
    return {
      blob: t.map((s) => eps * Math.exp(sigma * s)),
      sheet: t.map((s) => eps * Math.exp(Math.PI * s)),
    }
  }, [run, eps, sigma])

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-0.32, 0.32], equal: xAxis })
  const tAxis = useAxis({ label: 't', range: [0, duration] })
  const aAxis = useAxis({ label: 'amplitude', log: true, range: [eps / 2, 1] })
  return (
    <Figure
      title="Vortex-sheet roll-up"
      state={state}
      controls={
        <Player
          value={at}
          onChange={setFrame}
          count={count}
          label="time"
          format={(v) => `t = ${formatNumber(run.times[Math.min(v, count - 1)])}`}
        />
      }
      readouts={{
        now: (
          <>
            <Readout label="t" value={fmt(run.times[at])} />
            <Readout label="amplitude" value={fmt(run.amps[at])} />
          </>
        ),
        'linear growth rates': (
          <>
            <Readout label="σ, wavelength 1 (δ-sheet)" value={fmt(sigma)} />
            <Readout label="σ, wavelength 1 (sharp sheet)" value={fmt(Math.PI)} />
            <Readout label={`σ, wavelength 1/${shortest} (δ-sheet)`} value={fmt(sigmaShort)} />
            <Readout label={`σ, wavelength 1/${shortest} (δ = 0)`} value={fmt(sigmaShortBare)} />
          </>
        ),
      }}
      caption="One period of a vortex sheet carrying unit circulation, discretised by N blobs (dots) and started on the growing mode with amplitude ε. The flow above the sheet moves left at ½ and the flow below moves right at ½. Play the time: the wave grows, steepens and rolls up into a spiral around x = ½. The lower chart compares the measured amplitude (half the height of the sheet) with the linear law ε e^{σt}. Set δ to 0 for Rosenhead's point vortices: the shortest waves then grow fastest, they amplify rounding errors, and the sheet turns ragged by about t = 0.3. Larger δ slows the roll-up and gives fewer, fatter turns."
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve name="sheet" x={sheet.x} y={sheet.y} slot={0} width={1.5} silent />
        <Points name="blobs" x={sheet.x.slice(0, blobs)} y={sheet.y.slice(0, blobs)} slot={0} thin size={3} />
      </Plot>
      <Plot x={tAxis} y={aAxis} height={200}>
        <Curve name="measured amplitude" x={run.times.slice(0, at + 1)} y={run.amps.slice(0, at + 1)} slot={0} />
        <Curve name="ε e^{σt}, δ-sheet" x={run.times} y={growth.blob} slot={1} dashed />
        <Curve name="ε e^{πt}, sharp sheet" x={run.times} y={growth.sheet} muted dashed />
      </Plot>
    </Figure>
  )
}
