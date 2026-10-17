import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { heatProblem, pinnVariantRun, type PinnVariantSnapshot } from 'aifn-methods/neural/pinn'
import { FixedHeight, pick, rowsOf, useSlicedRuns } from '../_shared/useSlicedRuns'

const PANEL = 240
/** Floor of the log axis: a hard-constrained network's misfit is zero up to rounding. */
const FLOOR = 1e-8

type Setup = {
  kappa: number
  collocation: number
  adamSteps: number
  rate: number
  lbfgsSteps: number
  seed: number
}

/**
 * The heat equation solved twice by the same network from the same weights: once with the initial and boundary values
 * as penalty terms, once with them built into the trial function u = sin πx + t(1 − x²)N(x, t).
 */
export function SoftAndHardConditions() {
  const state = useFigureState({
    kappa: slider(0.02, 0.4, 0.1, { step: 0.02, label: 'diffusivity κ' }),
    collocation: int(64, { ge: 16, le: 1024, suggestions: [32, 64, 128, 256], label: 'collocation points' }),
    adamSteps: int(300, { ge: 0, le: 5000, suggestions: [300, 800, 2000], label: 'Adam steps' }),
    rate: float(0.005, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.005, 0.02], label: 'Adam learning rate' }),
    lbfgsSteps: int(100, { ge: 0, le: 2000, suggestions: [0, 100, 200, 500], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed' }),
  })
  const { kappa, collocation, adamSteps, rate, lbfgsSteps, seed } = state
  const setup = useMemo<Setup>(
    () => ({ kappa, collocation, adamSteps, rate, lbfgsSteps, seed }),
    [kappa, collocation, adamSteps, rate, lbfgsSteps, seed],
  )
  const { snapshots, done } = useSlicedRuns<PinnVariantSnapshot>(setup, () =>
    (['soft', 'hard'] as const).map((constraint) =>
      pinnVariantRun(heatProblem(setup.kappa), {
        constraint,
        hidden: [12, 12],
        collocation: setup.collocation,
        adamSteps: setup.adamSteps,
        learningRate: setup.rate,
        lbfgsSteps: setup.lbfgsSteps,
        seed: setup.seed,
        every: 25,
        grid: [41, 21],
      }),
    ),
  )
  const count = Math.max(1, ...snapshots.map((s) => s.length))
  const [pos, setPos] = useState({ setup, at: 0 })
  const at = pos.setup === setup ? Math.min(pos.at, count - 1) : 0
  const soft = pick(snapshots[0], at)
  const hard = pick(snapshots[1], at)
  const [slice, setSlice] = useState(0.5)

  const view = useMemo(() => {
    if (!soft || !hard || !soft.solution) return null
    const nx = soft.xs.length
    const error = (s: PinnVariantSnapshot) => s.prediction.map((v, i) => v - s.solution![i])
    const softError = error(soft)
    const hardError = error(hard)
    const scale = Math.max(1e-6, ...softError.map(Math.abs), ...hardError.map(Math.abs))
    const row = Math.round(slice * (soft.ts.length - 1)) * nx
    const kept = (run: readonly PinnVariantSnapshot[] | undefined) => (run ?? []).slice(0, at + 1)
    const trace = (run: readonly PinnVariantSnapshot[] | undefined, f: (s: PinnVariantSnapshot) => number) =>
      kept(run).map((s) => Math.max(FLOOR, f(s)))
    return {
      xs: soft.xs,
      ts: soft.ts,
      softError: rowsOf(softError, nx),
      hardError: rowsOf(hardError, nx),
      scale,
      exact: soft.solution.slice(row, row + nx),
      softSlice: soft.prediction.slice(row, row + nx),
      hardSlice: hard.prediction.slice(row, row + nx),
      steps: kept(snapshots[0]).map((s) => s.done),
      hardSteps: kept(snapshots[1]).map((s) => s.done),
      softL2: trace(snapshots[0], (s) => s.error2.at(-1) ?? NaN),
      hardL2: trace(snapshots[1], (s) => s.error2.at(-1) ?? NaN),
      softCondition: trace(snapshots[0], (s) => s.conditionError),
      hardCondition: trace(snapshots[1], (s) => s.conditionError),
    }
  }, [soft, hard, slice, snapshots, at])

  const xA = useAxis({ label: 'x', range: [-1, 1] })
  const tA = useAxis({ label: 't', range: [0, 1] })
  const xB = useAxis({ label: 'x', range: [-1, 1] })
  const tB = useAxis({ label: 't', range: [0, 1] })
  const xC = useAxis({ label: 'x', range: [-1, 1] })
  const uC = useAxis({ label: 'u', range: [-1.1, 1.1] })
  const sD = useAxis({ label: 'step', range: [0, Math.max(10, adamSteps + lbfgsSteps)] })
  const eD = useAxis({ label: 'error', log: true, range: [FLOOR, 10] })
  const percent = (e?: number) => (e === undefined || !Number.isFinite(e) ? '…' : `${(100 * e).toFixed(2)}%`)

  return (
    <Figure
      title="Penalised against built-in conditions"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`The heat equation u_t = κu_xx on [−1, 1] × [0, 1] from u(x, 0) = sin πx with u(±1, t) = 0, solved by two networks with two hidden layers of 12 tanh units that start from the same weights. The soft network penalises the misfit to the initial and boundary values next to the residual, both with weight 1. The hard network is u = sin πx + t(1 − x²)N(x, t), which meets the conditions for every weight, and is trained on the residual alone. Training is Adam, then L-BFGS; the player scrubs the checkpoints. Top: the error u_θ − u of each network over the domain, on one colour scale. Drag the horizontal line to choose the time of the slice drawn bottom left. Bottom right: the relative L2 error (solid) and the largest misfit to the conditions (dashed) against the step, on a log scale; the hard network's misfit sits at rounding level, drawn at ${FLOOR}.`}
      controls={
        <Player
          value={at}
          onChange={(a) => setPos({ setup, at: a })}
          count={count}
          label="checkpoint"
          format={(a) => {
            const s = pick(snapshots[0], a)
            return s ? `step ${s.done} (${s.phase === 'adam' ? 'Adam' : 'L-BFGS'})` : 'step 0'
          }}
        />
      }
      readouts={
        <>
          <Readout label="step" value={soft ? `${soft.done}${done ? '' : ', training…'}` : '…'} />
          <Readout label="error, soft" value={percent(soft?.error2.at(-1))} />
          <Readout label="error, hard" value={percent(hard?.error2.at(-1))} />
          <Readout label="condition misfit, soft" value={soft ? formatNumber(soft.conditionError) : '…'} />
          <Readout label="condition misfit, hard" value={hard ? formatNumber(hard.conditionError) : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xA} y={tA} title="error, soft conditions">
            {view && (
              <Raster
                x={view.xs}
                y={view.ts}
                z={view.softError}
                scale="diverging"
                range={[-view.scale, view.scale]}
                valueLabel="u_θ − u"
              />
            )}
            <Handle kind="y" at={slice} label="slice" onDrag={(t) => setSlice(Math.min(1, Math.max(0, t)))} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xB} y={tB} title="error, hard conditions">
            {view && (
              <Raster
                x={view.xs}
                y={view.ts}
                z={view.hardError}
                scale="diverging"
                range={[-view.scale, view.scale]}
                valueLabel="u_θ − u"
              />
            )}
            <Handle kind="y" at={slice} label="slice" onDrag={(t) => setSlice(Math.min(1, Math.max(0, t)))} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xC} y={uC} title={`u(x, t) at t = ${slice.toFixed(2)}`}>
            {view && <Curve name="exact" x={view.xs} y={view.exact} muted dashed />}
            {view && <Curve name="soft" x={view.xs} y={view.softSlice} slot={1} />}
            {view && <Curve name="hard" x={view.xs} y={view.hardSlice} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sD} y={eD} title="error and condition misfit">
            {view && <Curve name="relative error, soft" x={view.steps} y={view.softL2} slot={1} />}
            {view && <Curve name="relative error, hard" x={view.hardSteps} y={view.hardL2} slot={0} emphasis />}
            {view && <Curve name="misfit, soft" x={view.steps} y={view.softCondition} slot={1} dashed />}
            {view && <Curve name="misfit, hard" x={view.hardSteps} y={view.hardCondition} slot={0} dashed />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
