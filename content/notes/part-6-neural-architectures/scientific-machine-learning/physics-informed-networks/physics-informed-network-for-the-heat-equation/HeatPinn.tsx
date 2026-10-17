import { useEffect, useMemo, useState } from 'react'
import {
  call,
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
  useStreamed,
} from 'aifn-render'
import type { PinnRun } from 'aifn-methods/neural/pinn'
import { FixedHeight, rowsOf } from '../_shared/useSlicedRuns'
import { useStreamedRun } from '../_shared/useStreamedRun'

const PANEL = 230
/** Collocation counts of the sweep. */
const COUNTS = [4, 8, 16, 32, 64] as const

/**
 * A physics-informed network trained on the heat equation u_t = κu_xx from u(x, 0) = sin πx with zero ends (the
 * engine's `pinnRun` on `heat`, streamed from the worker), against the exact solution e^{−κπ²t} sin πx.
 */
export function HeatPinn() {
  const state = useFigureState({
    kappa: slider(0.01, 0.5, 0.1, { step: 0.01, label: 'diffusivity κ' }),
    collocation: int(64, { ge: 4, le: 4096, suggestions: [16, 64, 256, 1024], label: 'collocation points' }),
    width: int(12, { ge: 4, le: 64, suggestions: [8, 12, 16, 32], label: 'units per hidden layer' }),
    adamSteps: int(300, { ge: 0, le: 10000, suggestions: [300, 500, 2000], label: 'Adam steps' }),
    rate: float(0.005, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.005, 0.02], label: 'Adam learning rate' }),
    lbfgsSteps: int(100, { ge: 0, le: 5000, suggestions: [0, 100, 300, 1000], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed' }),
  })
  const { kappa, collocation, width, adamSteps, rate, lbfgsSteps, seed } = state
  const task = useMemo(
    () =>
      call<PinnRun>('neural/pinn/pinnRun', call('neural/pinn/heat', kappa), {
        hidden: [width, width],
        collocation,
        initialPoints: 50,
        boundaryPoints: 25,
        adamSteps,
        learningRate: rate,
        lbfgsSteps,
        seed,
        grid: [41, 21],
        every: 25,
      }),
    [kappa, collocation, width, adamSteps, rate, lbfgsSteps, seed],
  )
  const { snapshots, running, error } = useStreamedRun(task)
  const count = Math.max(1, snapshots.length)
  const [pos, setPos] = useState({ task, at: 0 })
  const following = pos.task !== task || pos.at >= count - 1
  const at = following ? count - 1 : pos.at
  const s = snapshots[Math.min(at, snapshots.length - 1)]
  const [slice, setSlice] = useState(0.5)

  const view = useMemo(() => {
    if (!s || !s.solution) return null
    const nx = s.xs.length
    const prediction = Array.from(s.prediction)
    const solution = Array.from(s.solution)
    const row = Math.round(slice * (s.ts.length - 1)) * nx
    const points = Array.from(s.collocation)
    return {
      xs: Array.from(s.xs),
      ts: Array.from(s.ts),
      error: rowsOf(
        prediction.map((v, i) => v - solution[i]),
        nx,
      ),
      scale: Math.max(1e-6, ...prediction.map((v, i) => Math.abs(v - solution[i]))),
      exact: solution.slice(row, row + nx),
      network: prediction.slice(row, row + nx),
      px: points.filter((_, i) => i % 2 === 0),
      pt: points.filter((_, i) => i % 2 === 1),
    }
  }, [s, slice])

  const xA = useAxis({ label: 'x', range: [-1, 1] })
  const tA = useAxis({ label: 't', range: [0, 1] })
  const xB = useAxis({ label: 'x', range: [-1, 1] })
  const uB = useAxis({ label: 'u', range: [-1.1, 1.1] })

  return (
    <Figure
      title="The heat equation, learned"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption={`The heat equation u_t = κu_xx on [−1, 1] × [0, 1] from u(x, 0) = sin πx with u(±1, t) = 0, whose solution is e^{−κπ²t} sin πx. A network with two hidden layers of tanh units is trained in the compute worker on the residual at a scrambled Sobol set of collocation points (dots) and the misfits at 50 initial and 2 × 25 boundary points, by Adam, then L-BFGS. The player scrubs the snapshots as they arrive. Left: the error u_θ − u over the domain. Drag the horizontal line to choose the time of the slice on the right, where the network (solid) is drawn over the exact solution (dashed).`}
      controls={
        <Player
          value={at}
          onChange={(a) => setPos({ task, at: a })}
          count={count}
          label="snapshot"
          format={(a) => {
            const c = snapshots[a]
            return c ? `step ${c.done} (${c.phase === 'adam' ? 'Adam' : 'L-BFGS'})` : 'step 0'
          }}
        />
      }
      readouts={
        <>
          <Readout label="step" value={s ? `${s.done}${running ? ', training…' : ''}` : '…'} />
          <Readout
            label="relative error"
            value={s?.error2.length ? `${(100 * (s.error2.at(-1) ?? NaN)).toFixed(3)}%` : '…'}
          />
          <Readout label="largest pointwise error" value={view ? view.scale.toExponential(2) : '…'} />
          {(error || s?.error) && <Readout label="error" value={error ?? s?.error ?? ''} />}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xA} y={tA} title="error u_θ − u">
            {view && (
              <Raster
                x={view.xs}
                y={view.ts}
                z={view.error}
                scale="diverging"
                range={[-view.scale, view.scale]}
                valueLabel="u_θ − u"
              />
            )}
            {view && view.px.length > 0 && <Points name="collocation points" x={view.px} y={view.pt} size={3} muted />}
            <Handle kind="y" at={slice} label="slice" onDrag={(t) => setSlice(Math.min(1, Math.max(0, t)))} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xB} y={uB} title={`u(x, t) at t = ${slice.toFixed(2)}`}>
            {view && <Curve name="exact" x={view.xs} y={view.exact} muted dashed />}
            {view && <Curve name="network" x={view.xs} y={view.network} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}

type SweepPoint = { count: number; error: number; atPoints: number; onGrid: number }

/**
 * The heat equation solved once for each collocation count in a sweep, one worker run after another: the relative
 * error against the exact solution, and the mean squared residual at the collocation points against that on a fine
 * grid.
 */
export function CollocationSweep() {
  const state = useFigureState({
    kappa: slider(0.01, 0.5, 0.1, { step: 0.01, label: 'diffusivity κ' }),
    adamSteps: int(300, { ge: 0, le: 10000, suggestions: [300, 500, 2000], label: 'Adam steps' }),
    lbfgsSteps: int(100, { ge: 0, le: 5000, suggestions: [0, 100, 300, 1000], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed' }),
  })
  const { kappa, adamSteps, lbfgsSteps, seed } = state
  const key = useMemo(() => ({ kappa, adamSteps, lbfgsSteps, seed }), [kappa, adamSteps, lbfgsSteps, seed])
  const [results, setResults] = useState<{ key: typeof key; points: SweepPoint[] }>({ key, points: [] })
  const points = results.key === key ? results.points : []
  const next = points.length < COUNTS.length ? COUNTS[points.length] : null
  const task = useMemo(
    () =>
      next === null
        ? null
        : call<PinnRun>('neural/pinn/pinnRun', call('neural/pinn/heat', key.kappa), {
            hidden: [12, 12],
            collocation: next,
            initialPoints: 50,
            boundaryPoints: 25,
            adamSteps: key.adamSteps,
            lbfgsSteps: key.lbfgsSteps,
            seed: key.seed,
            grid: [41, 21],
            every: 100000,
          }),
    [key, next],
  )
  const run = useStreamed(task)
  const value = run.value
  useEffect(() => {
    if (!value || !value.finished || next === null || value.collocation.length !== 2 * next) return
    const atPoints = value.residualLoss.at(-1) ?? NaN
    const onGrid = Array.from(value.residual).reduce((m, r) => m + r * r, 0) / value.residual.length
    setResults((r) => {
      const list = r.key === key ? r.points : []
      if (list.some((p) => p.count === next)) return r
      return { key, points: [...list, { count: next, error: value.error2.at(-1) ?? NaN, atPoints, onGrid }] }
    })
  }, [value, key, next])

  const nA = useAxis({ label: 'collocation points', log: true, range: [3, 80] })
  const eA = useAxis({ label: 'relative error', log: true })
  const nB = useAxis({ label: 'collocation points', log: true, range: [3, 80] })
  const rB = useAxis({ label: 'mean squared residual', log: true })
  const xs = points.map((p) => p.count)

  return (
    <Figure
      title="How many collocation points"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption={`The heat equation of the figure above, solved by a network with two hidden layers of 12 tanh units for each of ${COUNTS.length} collocation counts from ${COUNTS[0]} to ${COUNTS[COUNTS.length - 1]}, one worker run after another with the same seed and the same condition points. Left: the relative L2 error against the exact solution on a 41 × 21 grid; the five runs take about two seconds together. Right: the mean squared residual at the collocation points (the loss the network was trained on) and on the grid. A gap between the two means the network obeys the equation at its points and not between them.`}
      readouts={
        <>
          <Readout
            label="runs done"
            value={`${points.length} of ${COUNTS.length}${next === null ? '' : ', training…'}`}
          />
          {run.error && <Readout label="error" value={run.error} />}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={nA} y={eA} title="error against the count">
            {xs.length > 0 && (
              <Curve name="relative error" x={xs} y={points.map((p) => p.error)} slot={0} emphasis showPoints />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={nB} y={rB} title="residual at the points and between them">
            {xs.length > 0 && (
              <Curve
                name="at the collocation points"
                x={xs}
                y={points.map((p) => Math.max(1e-12, p.atPoints))}
                slot={1}
                showPoints
              />
            )}
            {xs.length > 0 && (
              <Curve
                name="on the grid"
                x={xs}
                y={points.map((p) => Math.max(1e-12, p.onGrid))}
                slot={0}
                emphasis
                showPoints
              />
            )}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
