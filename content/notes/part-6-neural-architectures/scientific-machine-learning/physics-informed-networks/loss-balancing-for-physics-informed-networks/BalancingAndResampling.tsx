import { useMemo, useState } from 'react'
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
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { layerProblem, pinnVariantRun, type PinnVariantSnapshot } from 'aifn-methods/neural/pinn'
import { FixedHeight, pick, useSlicedRuns } from '../_shared/useSlicedRuns'

const PANEL = 240

type Setup = {
  width: number
  collocation: number
  boundaryWeight: number
  balance: boolean
  resample: boolean
  adamSteps: number
  rate: number
  lbfgsSteps: number
  seed: number
}

/**
 * A steep interior layer, −u″ = f with u = tanh((x − ½)/ε), solved by a plain physics-informed network and by the same
 * network with gradient-norm loss weights and residual-based resampling of the collocation points, each switchable.
 */
export function BalancingAndResampling() {
  const state = useFigureState({
    width: float(0.01, { ge: 0.005, le: 0.2, suggestions: [0.01, 0.02, 0.05, 0.1], label: 'layer width ε' }),
    collocation: int(24, { ge: 8, le: 512, suggestions: [16, 24, 64, 256], label: 'collocation points' }),
    boundaryWeight: float(1, { ge: 0.01, le: 1e6, suggestions: [1, 100, 10000], label: 'boundary weight λ_b' }),
    balance: toggle(true, 'balance by gradient norms'),
    resample: toggle(true, 'resample where the residual is large'),
    adamSteps: int(800, { ge: 0, le: 10000, suggestions: [800, 2000, 5000], label: 'Adam steps' }),
    rate: float(0.005, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.005, 0.02], label: 'Adam learning rate' }),
    lbfgsSteps: int(100, { ge: 0, le: 2000, suggestions: [0, 100, 300, 1000], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed' }),
  })
  const { width, collocation, boundaryWeight, balance, resample, adamSteps, rate, lbfgsSteps, seed } = state
  const setup = useMemo<Setup>(
    () => ({ width, collocation, boundaryWeight, balance, resample, adamSteps, rate, lbfgsSteps, seed }),
    [width, collocation, boundaryWeight, balance, resample, adamSteps, rate, lbfgsSteps, seed],
  )
  const { snapshots, done } = useSlicedRuns<PinnVariantSnapshot>(setup, () =>
    [true, false].map((plainRun) =>
      pinnVariantRun(layerProblem(setup.width), {
        hidden: [20, 20],
        collocation: setup.collocation,
        weights: { boundary: setup.boundaryWeight },
        balancing: !plainRun && setup.balance ? 'gradient-norm' : 'fixed',
        balanceEvery: 50,
        // Uncapped gradient-norm weights, as when the note's numbers were measured (the engine caps them at 1e6).
        maxWeight: Infinity,
        sampling: !plainRun && setup.resample ? 'residual-adaptive' : 'fixed',
        resampleEvery: 100,
        adamSteps: setup.adamSteps,
        learningRate: setup.rate,
        lbfgsSteps: setup.lbfgsSteps,
        seed: setup.seed,
        every: 50,
        grid: [101, 1],
        measureGradients: true,
      }),
    ),
  )
  const count = Math.max(1, ...snapshots.map((s) => s.length))
  const [pos, setPos] = useState({ setup, at: 0 })
  const at = pos.setup === setup ? Math.min(pos.at, count - 1) : 0
  const plain = pick(snapshots[0], at)
  const tuned = pick(snapshots[1], at)

  const view = useMemo(() => {
    if (!plain || !tuned || !plain.solution) return null
    const floor = (v: number) => Math.max(1e-6, v)
    return {
      xs: plain.xs,
      exact: plain.solution,
      plain: plain.prediction,
      tuned: tuned.prediction,
      plainPoints: plain.collocation,
      tunedPoints: tuned.collocation,
      plainSteps: plain.errorSteps,
      plainError: plain.error2.map(floor),
      tunedSteps: tuned.errorSteps,
      tunedError: tuned.error2.map(floor),
      gradSteps: plain.gradientSteps,
      plainResidual: plain.gradientNorms.residual.map(floor),
      plainBoundary: plain.gradientNorms.boundary.map(floor),
      tunedResidual: tuned.gradientNorms.residual.map(floor),
      tunedBoundary: tuned.gradientNorms.boundary.map((v, i) =>
        floor(v * (tuned.weights.boundary[Math.max(0, tuned.gradientSteps[i] - 1)] ?? 1)),
      ),
      weightSteps: tuned.weights.boundary.map((_, i) => i + 1),
      weight: tuned.weights.boundary.map(floor),
    }
  }, [plain, tuned])

  const total = Math.max(10, adamSteps + lbfgsSteps)
  const xA = useAxis({ label: 'x', range: [0, 1] })
  const uA = useAxis({ label: 'u', range: [-1.3, 1.3] })
  const sB = useAxis({ label: 'step', range: [0, total] })
  const eB = useAxis({ label: 'relative error', log: true, range: [1e-4, 10] })
  const sC = useAxis({ label: 'step', range: [0, total] })
  const gC = useAxis({ label: 'gradient norm', log: true })
  const sD = useAxis({ label: 'step', range: [0, total] })
  const wD = useAxis({ label: 'boundary weight λ_b', log: true })
  const percent = (e?: number) => (e === undefined || !Number.isFinite(e) ? '…' : `${(100 * e).toFixed(1)}%`)
  const label = [balance && 'gradient-norm weights', resample && 'resampling'].filter(Boolean).join(' + ') || 'plain'

  return (
    <Figure
      title="Two fixes for one hard problem"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`Poisson's equation −u″ = f on [0, 1] with exact solution u = tanh((x − ½)/ε), a layer of width about ε at the centre, and its two end values. Two networks with two hidden layers of 20 tanh units start from the same weights. The plain network (orange) keeps the residual weight 1, the boundary weight λ_b and a fixed scrambled Sobol set of collocation points. The other (blue, "${label}") sets every weight to (sum of the terms' gradient norms) / (its own gradient norm), averaged with factor 0.9, every 50 Adam steps, and every 100 steps redraws the collocation points from a pool ten times larger with probability proportional to |r|/mean|r| + 1. Training is Adam, then L-BFGS with the weights and points frozen. Top left: the solutions and each run's collocation points (ticks at the bottom). Top right: the relative L2 error. Bottom left: the gradient norms of the residual term (solid) and of the weighted boundary term (dashed). Bottom right: the balanced run's boundary weight.`}
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
          <Readout label="step" value={plain ? `${plain.done}${done ? '' : ', training…'}` : '…'} />
          <Readout label="error, plain" value={percent(plain?.error2.at(-1))} />
          <Readout label={`error, ${label}`} value={percent(tuned?.error2.at(-1))} />
          <Readout label="boundary misfit, plain" value={plain ? formatNumber(plain.conditionError) : '…'} />
          <Readout label={`boundary misfit, ${label}`} value={tuned ? formatNumber(tuned.conditionError) : '…'} />
          <Readout
            label="points within 2ε of the layer"
            value={
              view
                ? `${view.plainPoints.filter((x) => Math.abs(x - 0.5) < 2 * width).length} and ${view.tunedPoints.filter((x) => Math.abs(x - 0.5) < 2 * width).length}`
                : '…'
            }
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xA} y={uA} title="solution">
            {view && <Curve name="exact" x={view.xs} y={view.exact} muted dashed />}
            {view && <Curve name="plain" x={view.xs} y={view.plain} slot={1} />}
            {view && <Curve name={label} x={view.xs} y={view.tuned} slot={0} emphasis />}
            {view && view.plainPoints.length > 0 && (
              <Points
                name="points, plain"
                x={view.plainPoints}
                y={view.plainPoints.map(() => -1.2)}
                slot={1}
                size={3}
              />
            )}
            {view && view.tunedPoints.length > 0 && (
              <Points
                name={`points, ${label}`}
                x={view.tunedPoints}
                y={view.tunedPoints.map(() => -1.1)}
                slot={0}
                size={3}
              />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sB} y={eB} title="error against the exact solution">
            {view && <Curve name="plain" x={view.plainSteps} y={view.plainError} slot={1} />}
            {view && <Curve name={label} x={view.tunedSteps} y={view.tunedError} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sC} y={gC} title="gradient norms of the terms">
            {view && <Curve name="residual, plain" x={view.gradSteps} y={view.plainResidual} slot={1} />}
            {view && <Curve name="boundary, plain" x={view.gradSteps} y={view.plainBoundary} slot={1} dashed />}
            {view && <Curve name={`residual, ${label}`} x={view.gradSteps} y={view.tunedResidual} slot={0} />}
            {view && (
              <Curve name={`weighted boundary, ${label}`} x={view.gradSteps} y={view.tunedBoundary} slot={0} dashed />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sD} y={wD} title="boundary weight">
            {view && <Curve name={`λ_b, ${label}`} x={view.weightSteps} y={view.weight} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
