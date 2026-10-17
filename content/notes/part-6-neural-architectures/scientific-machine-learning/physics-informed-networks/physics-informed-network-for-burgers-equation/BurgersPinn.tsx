import { useMemo, useState } from 'react'
import {
  call,
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { PinnRun } from 'aifn-methods/neural/pinn'
import { FixedHeight, rowsOf } from '../_shared/useSlicedRuns'
import { useStreamedRun } from '../_shared/useStreamedRun'

const PANEL = 230
const NU = 0.01 / Math.PI

/**
 * A physics-informed network trained on viscous Burgers' equation u_t + uu_x = νu_xx from u(x, 0) = −sin πx with zero
 * ends (the engine's `pinnRun` on `burgers`, streamed from the worker), against the Cole–Hopf exact solution.
 */
export function BurgersPinn() {
  const state = useFigureState({
    nu: float(NU, { ge: 0.001, le: 0.5, suggestions: [NU, 0.01, 0.05, 0.1], label: 'viscosity ν' }),
    width: int(16, { ge: 4, le: 64, suggestions: [16, 20, 40], label: 'units per hidden layer' }),
    depth: int(2, { ge: 1, le: 8, suggestions: [2, 3, 5], label: 'hidden layers' }),
    collocation: int(256, { ge: 64, le: 8192, suggestions: [256, 1024, 4096], label: 'collocation points' }),
    weight: float(1, { ge: 0.01, le: 100, suggestions: [0.1, 1, 10], label: 'condition weight w' }),
    adamSteps: int(200, { ge: 0, le: 20000, suggestions: [200, 1000, 5000], label: 'Adam steps' }),
    rate: float(0.005, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.005, 0.02], label: 'Adam learning rate' }),
    lbfgsSteps: int(100, { ge: 0, le: 10000, suggestions: [0, 100, 500, 2000], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed' }),
  })
  const { nu, width, depth, collocation, weight, adamSteps, rate, lbfgsSteps, seed } = state
  const task = useMemo(
    () =>
      call<PinnRun>('neural/pinn/pinnRun', call('neural/pinn/burgers', nu), {
        hidden: Array.from({ length: depth }, () => width),
        collocation,
        weight,
        adamSteps,
        learningRate: rate,
        lbfgsSteps,
        seed,
        initialPoints: 40,
        boundaryPoints: 20,
        grid: [61, 31],
        every: 50,
      }),
    [nu, width, depth, collocation, weight, adamSteps, rate, lbfgsSteps, seed],
  )
  const { snapshots, running, error } = useStreamedRun(task)
  const count = Math.max(1, snapshots.length)
  const [pos, setPos] = useState({ task, at: 0 })
  const following = pos.task !== task || pos.at >= count - 1
  const at = following ? count - 1 : pos.at
  const s = snapshots[Math.min(at, snapshots.length - 1)]
  const [slice, setSlice] = useState(0.75)

  const view = useMemo(() => {
    if (!s || !s.solution) return null
    const nx = s.xs.length
    const prediction = Array.from(s.prediction)
    const solution = Array.from(s.solution)
    const error = prediction.map((v, i) => Math.abs(v - solution[i]))
    const row = Math.round(slice * (s.ts.length - 1)) * nx
    const steps = Array.from({ length: s.loss.length }, (_, i) => i + 1)
    const floor = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(1e-8, v))
    return {
      xs: Array.from(s.xs),
      ts: Array.from(s.ts),
      exact: rowsOf(solution, nx),
      network: rowsOf(prediction, nx),
      error: rowsOf(error, nx),
      maxError: Math.max(...error),
      sliceExact: solution.slice(row, row + nx),
      sliceNetwork: prediction.slice(row, row + nx),
      steps,
      residualLoss: floor(s.residualLoss),
      initialLoss: floor(s.initialLoss),
      boundaryLoss: floor(s.boundaryLoss),
      errorSteps: Array.from(s.errorSteps),
      error2: floor(s.error2),
    }
  }, [s, slice])

  const xA = useAxis({ label: 'x', range: [-1, 1] })
  const tA = useAxis({ label: 't', range: [0, 1] })
  const xB = useAxis({ label: 'x', range: [-1, 1] })
  const tB = useAxis({ label: 't', range: [0, 1] })
  const xC = useAxis({ label: 'x', range: [-1, 1] })
  const tC = useAxis({ label: 't', range: [0, 1] })
  const xD = useAxis({ label: 'x', range: [-1, 1] })
  const uD = useAxis({ label: 'u', range: [-1.1, 1.1] })
  const sE = useAxis({ label: 'step', range: [0, Math.max(10, adamSteps + lbfgsSteps)] })
  const lE = useAxis({ label: 'mean squared value', log: true, range: [1e-8, 10] })
  const percent = (e?: number) => (e === undefined || !Number.isFinite(e) ? '…' : `${(100 * e).toFixed(1)}%`)
  const handle = <Handle kind="y" at={slice} label="slice" onDrag={(t) => setSlice(Math.min(1, Math.max(0, t)))} />

  return (
    <Figure
      title="Burgers' equation and its front"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`Viscous Burgers' equation u_t + uu_x = νu_xx on [−1, 1] × [0, 1] from u(x, 0) = −sin πx with u(±1, t) = 0. A multilayer perceptron of tanh units is trained in the compute worker on the mean squared residual at a scrambled Sobol set of collocation points plus w times the misfits at 40 initial and 2 × 20 boundary points, by Adam, then L-BFGS. The defaults take about two seconds and leave a rough fit; the text quotes longer runs. The player scrubs the snapshots as they arrive and follows the newest. Top: the Cole–Hopf exact solution and the network. Middle left: the pointwise error |u_θ − u|, which concentrates at the front at x = 0. Drag the horizontal line on any map to move the time of the slice drawn middle right. Bottom: the mean squared residual, initial misfit and boundary misfit (dashed) against the step, and the relative L2 error at the snapshots (emphasised).`}
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
          <Readout label="relative error" value={percent(s?.error2.at(-1))} />
          <Readout label="largest pointwise error" value={view ? view.maxError.toFixed(3) : '…'} />
          <Readout label="parameters" value={s ? String(s.parameters) : '…'} />
          {(error || s?.error) && <Readout label="error" value={error ?? s?.error ?? ''} />}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xA} y={tA} title="exact (Cole–Hopf)">
            {view && <Raster x={view.xs} y={view.ts} z={view.exact} scale="diverging" range={[-1, 1]} valueLabel="u" />}
            {handle}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xB} y={tB} title="network">
            {view && (
              <Raster x={view.xs} y={view.ts} z={view.network} scale="diverging" range={[-1, 1]} valueLabel="u_θ" />
            )}
            {handle}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xC} y={tC} title="pointwise error">
            {view && <Raster x={view.xs} y={view.ts} z={view.error} scale="sequential" valueLabel="|u_θ − u|" />}
            {handle}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xD} y={uD} title={`u(x, t) at t = ${slice.toFixed(2)}`}>
            {view && <Curve name="exact" x={view.xs} y={view.sliceExact} muted dashed />}
            {view && <Curve name="network" x={view.xs} y={view.sliceNetwork} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
        <div className="md:col-span-2">
          <FixedHeight height={PANEL}>
            <Plot x={sE} y={lE} title="loss terms and error">
              {view && <Curve name="residual" x={view.steps} y={view.residualLoss} slot={1} />}
              {view && <Curve name="initial misfit" x={view.steps} y={view.initialLoss} slot={2} />}
              {view && <Curve name="boundary misfit" x={view.steps} y={view.boundaryLoss} slot={2} dashed />}
              {view && (
                <Curve name="relative L2 error" x={view.errorSteps} y={view.error2} slot={0} emphasis showPoints />
              )}
            </Plot>
          </FixedHeight>
        </div>
      </div>
    </Figure>
  )
}
