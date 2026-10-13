import { useMemo, useState } from 'react'
import {
  Area,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import {
  bayesianOptimisationSteps,
  expectedImprovement,
  type BayesianOptimisationState,
} from 'aifn-methods/optimisation/bayesian'
import { forrester } from 'aifn-methods/data/objectives'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { FixedHeight } from '../_shared/FixedHeight'

/** The objective: the Forrester function on [0, 1]. */
const F = forrester()
/**
 * The constrained quantity c(λ): the engine's low-fidelity Forrester function, 0.5 f(λ) + 10(λ − 0.5) + 5, standing
 * in for a cost measured with each evaluation, such as memory. It is large where f has its global minimum.
 */
const C = forrester({ fidelity: 'low' })
const GRID = 201
const F_RANGE = [-12, 20] as const
const C_RANGE = [-6, 20] as const

const band = (m: ArrayLike<number>, s: ArrayLike<number>) => ({
  mean: Array.from(m),
  upper: Array.from(m, (v, i) => v + 2 * s[i]),
  lower: Array.from(m, (v, i) => v - 2 * s[i]),
})

/**
 * A constrained Bayesian optimisation run: one Gaussian process for f and one for c, expected improvement over the best
 * feasible evaluation weighted by the probability that c ≤ τ, one evaluation per step.
 */
export function FeasibleImprovement() {
  const state = useFigureState({
    tau: slider(-4, 16, 3, { onChart: true, label: 'threshold τ' }),
    init: int(6, { label: 'initial design', min: 2, max: 12, suggestions: [3, 6, 8] }),
    budget: int(10, { label: 'evaluations after the design', min: 1, max: 30, suggestions: [5, 10, 20] }),
    seed: int(2, { label: 'seed (design)', min: 0, max: 99, suggestions: [1, 2, 3, 4] }),
  })
  const { tau, init, budget, seed } = state

  // The whole run, recomputed when the threshold is let go rather than on every move of the handle.
  const runs = useComputed(
    () => {
      const alg = bayesianOptimisationSteps((x) => ({ value: F.value(x), constraints: [C.value(x)] }), F, {
        init,
        grid: GRID,
        constraints: [{ threshold: tau }],
      })
      const states = trace(alg, undefined, budget, { stream: stream(seed), keep: 'all' }).steps
      return { states: states as readonly BayesianOptimisationState[], grid: Array.from(toFlat(alg.grid)) }
    },
    [tau, init, budget, seed],
    { mode: 'release' },
  )
  const truth = useMemo(() => {
    const grid = runs.value.grid
    return { f: grid.map((x) => F.value(tensor([x]))), c: grid.map((x) => C.value(tensor([x]))) }
  }, [runs.value.grid])

  const key = `${tau}|${init}|${budget}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const states = runs.value.states
  const at = Math.min(cursor.key === key ? cursor.step : 0, states.length - 1)
  const s = states[at]
  const grid = runs.value.grid

  const view = useMemo(() => {
    const x = Array.from(toFlat(s.x))
    const fy = Array.from(toFlat(s.y))
    const cy = (toArray(s.constraintValues!) as number[][]).map((row) => row[0])
    const cMean = toArray(s.constraintMean!) as number[][]
    const cSd = toArray(s.constraintSd!) as number[][]
    const feasible = cy.map((v) => v <= tau)
    const hasFeasible = Number.isFinite(s.feasibleBest)
    // Plain expected improvement over the best feasible value, the factor the probability of feasibility weights.
    const ei = hasFeasible ? Array.from(toFlat(expectedImprovement(s.mean, s.sd, s.feasibleBest))) : null
    const eiChoice = ei ? ei.reduce((arg, v, i) => (v > ei[arg] ? i : arg), 0) : -1
    return {
      x,
      fy,
      cy,
      feasible,
      f: band(toFlat(s.mean), toFlat(s.sd)),
      c: band(cMean[0], cSd[0]),
      pof: Array.from(toFlat(s.feasibility!)),
      acq: Array.from(toFlat(s.acquisition)),
      ei,
      eiChoice,
      next: toFlat(s.next)[0],
    }
  }, [s, tau])

  const v = view
  const pick = (values: number[], keep: boolean) => ({
    x: v.x.filter((_, i) => v.feasible[i] === keep),
    y: values.filter((_, i) => v.feasible[i] === keep),
  })
  const ok = pick(v.fy, true)
  const bad = pick(v.fy, false)
  const lastX = v.x[v.x.length - 1]

  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: 'f(λ)', range: F_RANGE })
  const cAxis = useAxis({ label: 'c(λ)', range: C_RANGE })
  const pAxis = useAxis({ label: 'P(c ≤ τ)', range: [0, 1.05] })
  const aAxis = useAxis({ label: 'score', hold: 'union', key })
  return (
    <Figure
      title="Constrained Bayesian optimisation, one evaluation at a time"
      state={state}
      caption="Panel 1: the objective f (dashed), its Gaussian-process posterior (mean and ±2 sd band) and the evaluations so far, green where the constraint held and red where it failed; the dashed vertical line is the next evaluation. Panel 2: the constrained quantity c, its own posterior and the threshold τ; drag the horizontal line to move τ, and the run is recomputed when you let go. Panel 3: the posterior probability that c(λ) ≤ τ. Panel 4: expected improvement over the best feasible evaluation (dashed) and the acquisition the run maximises, EI times the probability of feasibility (filled). Step through with the player. At τ = 3 the global minimum of f near λ = 0.76 is infeasible: plain EI keeps choosing it, while the constrained run settles on the local minimum near 0.14. With no feasible evaluation the acquisition is the probability of feasibility alone."
      controls={
        <Player
          value={at}
          onChange={(step) => setCursor({ key, step })}
          count={states.length}
          label="step"
          format={(i) => `${init + i} evaluations`}
        />
      }
      readouts={
        <>
          <Readout label="feasible evaluations" value={`${ok.x.length} of ${s.evaluations}`} />
          <Readout
            label="best feasible f*ₜ"
            value={Number.isFinite(s.feasibleBest) ? formatNumber(s.feasibleBest) : 'none yet'}
          />
          <Readout label="at λ" value={s.feasibleBestIndex >= 0 ? formatNumber(v.x[s.feasibleBestIndex]) : '—'} />
          <Readout label="EI alone would choose λ" value={v.eiChoice >= 0 ? formatNumber(grid[v.eiChoice]) : '—'} />
          <Readout label="EI × P(feasible) chooses λ" value={formatNumber(v.next)} />
          <Readout label="P(feasible) there" value={formatNumber(v.pof[s.nextIndex])} />
        </>
      }
    >
      <FixedHeight height={840}>
        <Plots rows={4} heights={[3, 3, 2, 2]}>
          <Plot x={lam} y={fAxis} title="objective">
            <Area name="μₜ ± 2σₜ" x={grid} y={v.f.upper} base={v.f.lower} slot={0} opacity={0.18} line={false} />
            <Curve name="objective f" x={grid} y={truth.f} muted dashed />
            <Curve name="posterior mean" x={grid} y={v.f.mean} slot={0} stale={runs.stale} />
            {ok.x.length > 0 && <Points name="feasible evaluation" x={ok.x} y={ok.y} tone="success" size={8} />}
            {bad.x.length > 0 && (
              <Points name="infeasible evaluation" x={bad.x} y={bad.y} tone="destructive" size={8} />
            )}
            {at > 0 && <Points name="latest evaluation" x={[lastX]} y={[v.fy[v.fy.length - 1]]} emphasis size={12} />}
            <Segments
              name="next λ"
              segments={[{ from: [v.next, F_RANGE[0]], to: [v.next, F_RANGE[1]] }]}
              emphasis
              dashed
            />
          </Plot>
          <Plot x={lam} y={cAxis} title="constraint">
            <Area name="μ ± 2σ of c" x={grid} y={v.c.upper} base={v.c.lower} slot={5} opacity={0.18} line={false} />
            <Curve name="constrained quantity c" x={grid} y={truth.c} muted dashed />
            <Curve name="posterior mean of c" x={grid} y={v.c.mean} slot={5} stale={runs.stale} />
            <Points name="evaluations of c" x={v.x} y={v.cy} slot={5} size={7} />
            <Handle {...state.handle('tau', { axis: 'y', label: 'τ' })} />
          </Plot>
          <Plot x={lam} y={pAxis} title="probability of feasibility">
            <Area name="P(c ≤ τ)" x={grid} y={v.pof} slot={5} />
          </Plot>
          <Plot x={lam} y={aAxis} title="acquisition">
            {v.ei && <Curve name="EI over best feasible" x={grid} y={v.ei} muted dashed />}
            <Area name="EI × P(feasible)" x={grid} y={v.acq} slot={3} />
            <Points name="its maximum" x={[v.next]} y={[v.acq[s.nextIndex]]} emphasis size={9} />
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
