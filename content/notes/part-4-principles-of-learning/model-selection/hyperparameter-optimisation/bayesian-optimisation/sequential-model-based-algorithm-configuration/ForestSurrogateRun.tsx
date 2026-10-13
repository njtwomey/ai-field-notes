import { useMemo, useState } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  bayesianOptimisationSteps,
  type BayesianOptimisationOptions,
  type BayesianOptimisationState,
  type NamedObjective,
} from 'aifn-methods/optimisation/bayesian'
import { forrester } from 'aifn-methods/data/objectives'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { FixedHeight } from '../_shared/FixedHeight'

/**
 * A positive cost: the Forrester function plus 7, the engine's low-fidelity form a f(λ) + b(λ − 0.5) − c with a = 1,
 * b = 0 and c = −7. Its minimum is 0.979 at λ = 0.757, with a shallow local minimum of 6.01 near 0.14.
 */
const SPEC: NamedObjective = { name: 'forrester', fidelity: 'low', a: 1, b: 0, c: -7 }
const COST = forrester({ fidelity: 'low', a: 1, b: 0, c: -7 })
const GRID = 201
const INIT = 3

type Surrogate = 'random-forest' | 'gp'

/** Rows of a B × m matrix as one polyline, the rows separated by NaN, for a single thin layer. */
function polyline(grid: readonly number[], rows: readonly (readonly number[])[]) {
  const x: number[] = []
  const y: number[] = []
  rows.forEach((row) => {
    x.push(...grid, NaN)
    y.push(...row, NaN)
  })
  return { x, y }
}

/**
 * SMAC's loop on one dimension: a random forest (or a Gaussian process) fitted to the evaluations, its trees'
 * predictions, mean and spread, expected improvement (log-normal with a log cost), and a random point every k-th choice.
 */
export function ForestSurrogateRun() {
  const state = useFigureState({
    surrogate: choice<Surrogate>(
      [
        { value: 'random-forest', label: 'random forest (SMAC)' },
        { value: 'gp', label: 'Gaussian process' },
      ],
      'random-forest',
      { label: 'surrogate' },
    ),
    trees: int(10, {
      label: 'trees B',
      min: 1,
      max: 200,
      suggestions: [3, 10, 30, 100],
      when: (v) => v.surrogate === 'random-forest',
    }),
    every: int(2, { label: 'random point every k-th choice (0: never)', min: 0, max: 10, suggestions: [0, 2, 3, 5] }),
    logCost: setting(false, { label: 'model ln f (log-normal EI)' }),
    gp: setting(false, {
      label: 'GP posterior on the same evaluations',
      when: (v) => v.surrogate === 'random-forest',
    }),
    budget: int(20, { label: 'evaluations after the design', min: 1, max: 40, suggestions: [10, 20, 40] }),
    seed: int(1, { label: 'seed', min: 0, max: 99, suggestions: [0, 1, 2, 3] }),
  })
  const { surrogate, trees, every, logCost, budget, seed } = state
  const showGp = state.gp && surrogate === 'random-forest'

  const options: BayesianOptimisationOptions = useMemo(
    () => ({
      surrogate,
      grid: GRID,
      logCost,
      ...(surrogate === 'random-forest' ? { forest: { trees } } : {}),
    }),
    [surrogate, trees, logCost],
  )

  const runs = useMemo(() => {
    const alg = bayesianOptimisationSteps(SPEC, { ...options, init: INIT, interleaveRandom: every })
    const states = trace(alg, undefined, budget, { stream: stream(seed), keep: 'all' })
      .steps as readonly BayesianOptimisationState[]
    const grid = Array.from(toFlat(alg.grid))
    const cost = grid.map((x) => COST.value(tensor([x])))
    return { states, grid, cost, minimum: alg.minimum }
  }, [options, every, budget, seed])

  const key = `${surrogate}|${trees}|${every}|${logCost}|${budget}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const at = Math.min(cursor.key === key ? cursor.step : 0, runs.states.length - 1)
  const s = runs.states[at]
  const grid = runs.grid
  // With a log cost the surrogate, its trees and its band are on the scale of ln f, so the data are drawn there too.
  const scale = (v: number) => (logCost ? Math.log(v) : v)

  const view = useMemo(() => {
    const mean = Array.from(toFlat(s.mean))
    const sd = toFlat(s.sd)
    const xs = Array.from(toFlat(s.x))
    const ys = Array.from(toFlat(s.y))
    // Which evaluations were random: the choice made in state i is evaluation INIT + i + 1.
    const random = xs.map((_, i) => i >= INIT && runs.states[i - INIT].random)
    return {
      mean,
      upper: mean.map((m, i) => m + 2 * sd[i]),
      lower: mean.map((m, i) => m - 2 * sd[i]),
      trees: s.treePredictions ? polyline(grid, toArray(s.treePredictions) as number[][]) : null,
      acq: Array.from(toFlat(s.acquisition)),
      xs,
      ys,
      random,
      next: toFlat(s.next)[0],
    }
  }, [s, grid, runs.states])

  // The Gaussian process fitted to exactly the evaluations the forest has seen, for comparison.
  const gp = useMemo(() => {
    if (!showGp) return null
    const g = run(bayesianOptimisationSteps(SPEC, { grid: GRID, logCost, init: s.x }), undefined, 0, {
      stream: stream(seed),
    })
    const mean = Array.from(toFlat(g.mean))
    const sd = toFlat(g.sd)
    return { mean, upper: mean.map((m, i) => m + 2 * sd[i]), lower: mean.map((m, i) => m - 2 * sd[i]) }
  }, [showGp, logCost, s, seed])

  const progress = useMemo(() => {
    const ys = Array.from(toFlat(runs.states[runs.states.length - 1].y)).slice(0, INIT + at)
    let best = Infinity
    return { n: ys.map((_, i) => i + 1), best: ys.map((v) => (best = Math.min(best, v))) }
  }, [runs.states, at])

  const v = view
  const chosen = v.xs.map((_, i) => i).filter((i) => !v.random[i])
  const random = v.xs.map((_, i) => i).filter((i) => v.random[i])
  const truth = runs.cost.map(scale)
  const yRange: [number, number] = logCost ? [-0.5, 3.5] : [-2, 25]
  const evalsRandom = random.filter((i) => i < progress.n.length)

  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: logCost ? 'ln f(λ)' : 'cost f(λ)', range: yRange })
  const aAxis = useAxis({ label: logCost ? 'log-normal EI' : 'EI', hold: 'union', key })
  const nAxis = useAxis({ label: 'evaluations', range: [1, INIT + budget], integer: true })
  const bAxis = useAxis({ label: 'best cost so far', range: [0, 12] })
  return (
    <Figure
      title="A random-forest surrogate, one evaluation at a time"
      state={state}
      caption="Top: a positive cost, the Forrester function plus 7 (dashed), the evaluations so far (orange, or green when chosen at random), the surrogate's mean with its ±2 sd band, and for the forest each tree's prediction (thin lines). Middle: expected improvement over the 201 candidates; its maximum is the next evaluation unless the choice is a random one. Bottom: the best cost observed, with random evaluations marked. Step through with the player. The forest's mean is a step function and its spread is the disagreement of trees grown on different bootstrap samples: largest where few evaluations constrain the splits, and flat beyond the outermost evaluation. Switch on the Gaussian process on the same evaluations to compare its smooth mean and its band, which widens with distance from the data. Set the random interleaving to 0 to let EI choose every point, and model ln f to score candidates by the log-normal expected improvement in the cost itself."
      controls={
        <Player
          value={at}
          onChange={(step) => setCursor({ key, step })}
          count={runs.states.length}
          label="step"
          format={(i) => `${INIT + i} evaluations`}
        />
      }
      readouts={
        <>
          <Readout label="evaluations" value={String(s.evaluations)} />
          <Readout label="incumbent cost" value={formatNumber(s.best)} />
          <Readout label="at λ" value={formatNumber(toFlat(s.bestX)[0])} />
          <Readout label="next λ" value={`${formatNumber(v.next)}${s.random ? ' (random)' : ''}`} />
          <Readout label="EI at next λ" value={formatNumber(v.acq[s.nextIndex])} />
          <Readout label="true minimum" value={`${formatNumber(runs.minimum)} at 0.757`} />
        </>
      }
    >
      <FixedHeight height={660}>
        <Plots rows={3} heights={[3, 2, 2]}>
          <Plot x={lam} y={fAxis} title={surrogate === 'gp' ? 'Gaussian-process surrogate' : 'forest surrogate'}>
            <Area name="mean ± 2 sd" x={grid} y={v.upper} base={v.lower} slot={0} opacity={0.15} line={false} />
            {v.trees && <Curve name="tree predictions" x={v.trees.x} y={v.trees.y} slot={0} thin silent />}
            <Curve name="cost f" x={grid} y={truth} muted dashed />
            <Curve name="surrogate mean" x={grid} y={v.mean} slot={0} />
            {gp && (
              <>
                <Area name="GP ± 2 sd" x={grid} y={gp.upper} base={gp.lower} slot={6} opacity={0.12} line={false} />
                <Curve name="GP mean" x={grid} y={gp.mean} slot={6} dashed />
              </>
            )}
            {chosen.length > 0 && (
              <Points
                name="evaluations"
                x={chosen.map((i) => v.xs[i])}
                y={chosen.map((i) => scale(v.ys[i]))}
                slot={1}
                size={8}
              />
            )}
            {random.length > 0 && (
              <Points
                name="random evaluations"
                x={random.map((i) => v.xs[i])}
                y={random.map((i) => scale(v.ys[i]))}
                slot={2}
                size={8}
              />
            )}
            <Segments
              name="next λ"
              segments={[{ from: [v.next, yRange[0]], to: [v.next, yRange[1]] }]}
              emphasis
              dashed
            />
          </Plot>
          <Plot x={lam} y={aAxis} title={s.random ? 'acquisition (next choice is random)' : 'acquisition'}>
            <Area name={logCost ? 'log-normal EI' : 'EI'} x={grid} y={v.acq} slot={3} />
            {!s.random && <Points name="its maximum" x={[v.next]} y={[v.acq[s.nextIndex]]} emphasis size={9} />}
          </Plot>
          <Plot x={nAxis} y={bAxis} title="progress">
            <Curve name="true minimum" x={[1, INIT + budget]} y={[runs.minimum, runs.minimum]} muted dashed />
            <Curve name="best cost so far" x={progress.n} y={progress.best} slot={1} showPoints />
            {evalsRandom.length > 0 && (
              <Points
                name="after a random evaluation"
                x={evalsRandom.map((i) => i + 1)}
                y={evalsRandom.map((i) => progress.best[i])}
                slot={2}
                size={8}
              />
            )}
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
