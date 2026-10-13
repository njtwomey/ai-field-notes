import { useContext, useMemo, useState } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { matern52 } from 'aifn-compute/learning/kernels'
import { forrester } from 'aifn-methods/data/objectives'
import {
  multiFidelityBayesianOptimisationSteps,
  type MultiFidelityAcquisitionName,
} from 'aifn-methods/optimisation/bayesian'

const HIGH = forrester()
const LOW = forrester({ fidelity: 'low' })
const F_MIN = HIGH.minimumValue
/** The objective at fidelity 0 (cheap) or 1 (full). */
const objective = (x: Tensor, z: number) => (z === 0 ? LOW : HIGH).value(x)
/** Cheap and expensive evaluations in the initial design: nested points of one Latin hypercube. */
const INIT = [8, 4]
/**
 * Fixed kernels on the unit interval (targets scaled by their standard deviation): a Matérn 5/2 of lengthscale 0.15
 * for the cheap function and a smooth, small one for the discrepancy, so the scale ρ is estimated from the start.
 */
const KERNELS = [matern52({ lengthscale: 0.15 }), matern52({ lengthscale: 1, variance: 0.1 })]
const Y_RANGE = [-12, 20] as const

const ACQUISITIONS = [
  { value: 'mf-mes', label: 'max-value entropy search per unit cost' },
  { value: 'mf-gp-ucb', label: 'MF-GP-UCB' },
  { value: 'ei-per-cost', label: 'expected improvement per unit cost' },
] as const

/** A cost-aware run on Forrester's two fidelities, every state kept so a Player can scrub it. */
export function CostAwareRun() {
  const state = useFigureState({
    acquisition: choice<MultiFidelityAcquisitionName>([...ACQUISITIONS], 'mf-mes', { label: 'acquisition' }),
    cost: float(10, { ge: 1, le: 100, suggestions: [2, 5, 10, 20], label: 'cost of a full evaluation (cheap = 1)' }),
    steps: int(12, { ge: 1, le: 40, suggestions: [6, 12, 20, 30], label: 'evaluations after the design' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { acquisition, cost, steps, seed } = state

  const run = useMemo(() => {
    const alg = multiFidelityBayesianOptimisationSteps(
      objective,
      HIGH,
      { cost: [1, cost] },
      {
        acquisition,
        kernels: KERNELS,
        refit: false,
        init: INIT,
        grid: 101,
      },
    )
    const tr = trace(alg, undefined, steps, { stream: stream(`cost-aware-run/${seed}`), keep: 'all' })
    const grid = Array.from(toFlat(alg.grid))
    return {
      grid,
      truthHigh: grid.map((x) => HIGH.value(tensor([x]))),
      truthLow: grid.map((x) => LOW.value(tensor([x]))),
      states: tr.steps,
    }
  }, [acquisition, cost, steps, seed])

  const key = `${acquisition}|${cost}|${steps}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const at = Math.min(cursor.key === key ? cursor.step : 0, run.states.length - 1)
  const s = run.states[at]

  const view = useMemo(() => {
    const [meanLow, meanHigh] = toArray(s.mean) as number[][]
    const [, sdHigh] = toArray(s.sd) as number[][]
    const [acqLow, acqHigh] = toArray(s.acquisition) as number[][]
    const xs = Array.from(toFlat(s.x))
    const ys = Array.from(toFlat(s.y))
    const pick = (z: number) => {
      const i = s.fidelity.map((f, k) => (f === z ? k : -1)).filter((k) => k >= 0)
      return { x: i.map((k) => xs[k]), y: i.map((k) => ys[k]) }
    }
    return {
      meanLow,
      meanHigh,
      upper: meanHigh.map((m, i) => m + 2 * sdHigh[i]),
      lower: meanHigh.map((m, i) => m - 2 * sdHigh[i]),
      acqLow,
      acqHigh,
      low: pick(0),
      high: pick(1),
      next: toFlat(s.next)[0],
    }
  }, [s])

  // The incumbent at full fidelity against the cost spent, over the whole run up to this step.
  const progress = useMemo(() => {
    const shown = run.states.slice(0, at + 1)
    return { cost: shown.map((t) => t.cost), best: shown.map((t) => t.best) }
  }, [run, at])

  const frame = useContext(FrameContext)
  const lam = useAxis({ label: 'hyperparameter x', range: [0, 1] })
  const fAxis = useAxis({ label: 'loss', range: Y_RANGE })
  const aAxis = useAxis({ label: 'acquisition', hold: 'union', key })
  const cAxis = useAxis({ label: 'cost spent', range: [run.states[0].cost, run.states[run.states.length - 1].cost] })
  const bAxis = useAxis({ label: 'best full-fidelity loss', range: [-6.5, 0] })
  const counts = [s.fidelity.filter((f) => f === 0).length, s.fidelity.filter((f) => f === 1).length]
  return (
    <Figure
      title="Choosing the point and the fidelity"
      state={state}
      caption="Multi-fidelity Bayesian optimisation of the Forrester function f (ink, dashed), with its tilted cheap version (muted) available at a cost of 1 against the full evaluation's cost. The run starts from 8 cheap and 4 full evaluations and fits the two-fidelity model f = ρ·f_lo + δ with fixed kernels. Top: the model's mean and two-standard-deviation band at full fidelity (blue), its mean for the cheap version (orange, thin), the cheap evaluations (orange points) and the full ones (ink), and the next point (dashed line). Middle: the acquisition of a cheap (orange) and a full (blue) evaluation at each x; the run evaluates the pair with the highest score. Bottom: the lowest full-fidelity loss observed against the cost spent. Step through the run with the player. Max-value entropy search per unit cost spends most steps on cheap evaluations near the minimum and pays for a full evaluation once the cheap ones have located it; MF-GP-UCB moves to the full fidelity wherever the cheap one is already known to within its threshold, and pays for it; expected improvement per unit cost can spend every step cheaply and never confirm the minimum at full fidelity."
      controls={
        <Player
          value={at}
          onChange={(v) => setCursor({ key, step: v })}
          count={run.states.length}
          label="step"
          format={(v) => `${v} evaluations after the design`}
        />
      }
      readouts={{
        run: (
          <>
            <Readout label="cost spent" value={formatNumber(s.cost)} />
            <Readout label="evaluations, cheap / full" value={`${counts[0]} / ${counts[1]}`} />
            <Readout label="best full-fidelity loss" value={formatNumber(s.best)} />
            <Readout label="at x" value={formatNumber(toFlat(s.bestX)[0])} />
            <Readout label="fitted ρ (truth 2)" value={formatNumber(s.rho[0])} />
          </>
        ),
        next: (
          <>
            <Readout label="x" value={formatNumber(view.next)} />
            <Readout label="fidelity" value={s.nextFidelity === 0 ? 'cheap' : 'full'} />
          </>
        ),
        truth: <Readout label="minimum of f" value={`${formatNumber(F_MIN)} at x = 0.757`} />,
      }}
    >
      <FrameContext.Provider value={{ ...frame, height: 640 }}>
        <Plots rows={3} heights={[3, 1.6, 1.6]}>
          <Plot x={lam} y={fAxis} title="model at both fidelities">
            <Area
              name="full fidelity ± 2 sd"
              x={run.grid}
              y={view.upper}
              base={view.lower}
              slot={0}
              opacity={0.18}
              line={false}
            />
            <Curve name="cheap version f_lo" x={run.grid} y={run.truthLow} muted thin />
            <Curve name="full fidelity f" x={run.grid} y={run.truthHigh} emphasis dashed thin />
            <Curve name="model mean, full" x={run.grid} y={view.meanHigh} slot={0} />
            <Curve name="model mean, cheap" x={run.grid} y={view.meanLow} slot={1} thin />
            {view.low.x.length > 0 && (
              <Points name="cheap evaluations" x={view.low.x} y={view.low.y} slot={1} size={6} />
            )}
            {view.high.x.length > 0 && (
              <Points name="full evaluations" x={view.high.x} y={view.high.y} emphasis size={8} />
            )}
            <Segments
              name="next x"
              segments={[{ from: [view.next, Y_RANGE[0]], to: [view.next, Y_RANGE[1]] }]}
              emphasis
              dashed
            />
          </Plot>
          <Plot x={lam} y={aAxis} title="acquisition">
            <Curve name="cheap evaluation" x={run.grid} y={view.acqLow} slot={1} />
            <Curve name="full evaluation" x={run.grid} y={view.acqHigh} slot={0} />
            <Points
              name="chosen"
              x={[view.next]}
              y={[(s.nextFidelity === 0 ? view.acqLow : view.acqHigh)[s.nextIndex]]}
              emphasis
              size={9}
            />
          </Plot>
          <Plot x={cAxis} y={bAxis} title="progress">
            <Curve
              name="minimum of f"
              x={[run.states[0].cost, run.states[run.states.length - 1].cost]}
              y={[F_MIN, F_MIN]}
              muted
              dashed
            />
            <Curve name="best full-fidelity loss" x={progress.cost} y={progress.best} slot={2} showPoints />
          </Plot>
        </Plots>
      </FrameContext.Provider>
    </Figure>
  )
}
