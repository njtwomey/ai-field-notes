import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  bayesianOptimisationSteps,
  expectedImprovement,
  probabilityOfImprovement,
} from 'aifn-methods/optimisation/bayesian'
import { forrester } from 'aifn-methods/data/objectives'
import { stream } from 'aifn-compute/foundation/random'
import { mul, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
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

/** The largest entry of a score and where it is. */
function argmax(v: readonly number[]) {
  const i = v.reduce((arg, x, k) => (x > v[arg] ? k : arg), 0)
  return { i, value: v[i] }
}

/**
 * Expected improvement over the best feasible evaluation, the posterior probability of feasibility, and their product,
 * from two Gaussian processes fitted to the same evaluations of f and c.
 */
export function FeasibleImprovement() {
  const state = useFigureState({
    tau: slider(-4, 16, 3, { onChart: true, label: 'threshold τ' }),
    n: int(6, { label: 'evaluations t', min: 2, max: 15, suggestions: [4, 6, 8, 12] }),
    seed: int(2, { label: 'seed (design)', min: 0, max: 99, suggestions: [1, 2, 3, 4] }),
  })
  const { tau, n, seed } = state

  const fits = useMemo(() => {
    // One design, evaluated for both functions; each gets its own GP with hyperparameters by marginal likelihood.
    const fAlg = bayesianOptimisationSteps(F.value, F, { init: n, grid: GRID })
    const fs = run(fAlg, undefined, 0, { stream: stream(seed) })
    const cAlg = bayesianOptimisationSteps(C.value, C, { init: fs.x, grid: GRID })
    const cs = run(cAlg, undefined, 0, { stream: stream(seed) })
    const grid = Array.from(toFlat(fAlg.grid))
    const band = (m: ArrayLike<number>, s: ArrayLike<number>) => ({
      mean: Array.from(m),
      upper: Array.from(m, (v, i) => v + 2 * s[i]),
      lower: Array.from(m, (v, i) => v - 2 * s[i]),
    })
    return {
      grid,
      fTruth: grid.map((x) => F.value(tensor([x]))),
      cTruth: grid.map((x) => C.value(tensor([x]))),
      x: Array.from(toFlat(fs.x)),
      fy: Array.from(toFlat(fs.y)),
      cy: Array.from(toFlat(cs.y)),
      fPost: { m: fs.mean, s: fs.sd, ...band(toFlat(fs.mean), toFlat(fs.sd)) },
      cPost: { m: cs.mean, s: cs.sd, ...band(toFlat(cs.mean), toFlat(cs.sd)) },
    }
  }, [n, seed])

  const scores = useMemo(() => {
    const feasible = fits.cy.map((c) => c <= tau)
    const feasibleValues = fits.fy.filter((_, i) => feasible[i])
    const incumbent = feasibleValues.length > 0 ? Math.min(...feasibleValues) : null
    // P(c(λ) ≤ τ) under the constraint's posterior: the probability that c "improves" on the threshold τ.
    const pof = toFlat(probabilityOfImprovement(fits.cPost.m, fits.cPost.s, tau))
    if (incumbent === null) {
      return {
        feasible,
        incumbent,
        pof: Array.from(pof),
        ei: null,
        product: Array.from(pof),
        choice: argmax(Array.from(pof)),
      }
    }
    const eiT = expectedImprovement(fits.fPost.m, fits.fPost.s, incumbent)
    const product = Array.from(toFlat(mul(eiT, tensor(pof))))
    return { feasible, incumbent, pof: Array.from(pof), ei: Array.from(toFlat(eiT)), product, choice: argmax(product) }
  }, [fits, tau])

  const r = scores
  const unconstrained = r.ei ? argmax(r.ei) : null
  const nextX = fits.grid[r.choice.i]
  const ok = fits.x.filter((_, i) => r.feasible[i])
  const bad = fits.x.filter((_, i) => !r.feasible[i])
  const pick = (v: number[], xs: number[]) => xs.map((x) => v[fits.x.indexOf(x)])

  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: 'f(λ)', range: F_RANGE })
  const cAxis = useAxis({ label: 'c(λ)', range: C_RANGE })
  const pAxis = useAxis({ label: 'P(c ≤ τ)', range: [0, 1.05] })
  const aAxis = useAxis({ label: 'score', hold: 'union', key: `${n}|${seed}` })
  return (
    <Figure
      title="Expected improvement times the probability of feasibility"
      state={state}
      caption="Panel 1: the objective f (dashed), its Gaussian-process posterior (mean and ±2 sd band) and t evaluations, green where the constraint held and red where it failed. Panel 2: the constrained quantity c, its own posterior and the threshold τ; drag the horizontal line to move τ. Panel 3: the posterior probability that c(λ) ≤ τ. Panel 4: expected improvement over the best feasible evaluation (dashed) and the constrained acquisition, EI times the probability of feasibility (filled), with their maxima. At τ = 3 the global minimum of f near λ = 0.76 is infeasible; plain EI still goes there, and the constrained acquisition does not. With no feasible evaluation the acquisition is the probability of feasibility alone."
      readouts={
        <>
          <Readout label="feasible evaluations" value={`${ok.length} of ${n}`} />
          <Readout label="best feasible f*ₜ" value={r.incumbent === null ? 'none yet' : formatNumber(r.incumbent)} />
          <Readout label="EI chooses λ" value={unconstrained ? formatNumber(fits.grid[unconstrained.i]) : '—'} />
          <Readout label="EI × P(feasible) chooses λ" value={formatNumber(nextX)} />
          <Readout label="P(feasible) there" value={formatNumber(r.pof[r.choice.i])} />
        </>
      }
    >
      <FixedHeight height={720}>
        <Plots rows={4} heights={[3, 3, 2, 2]}>
          <Plot x={lam} y={fAxis} title="objective">
            <Area
              name="μₜ ± 2σₜ"
              x={fits.grid}
              y={fits.fPost.upper}
              base={fits.fPost.lower}
              slot={0}
              opacity={0.18}
              line={false}
            />
            <Curve name="objective f" x={fits.grid} y={fits.fTruth} muted dashed />
            <Curve name="posterior mean" x={fits.grid} y={fits.fPost.mean} slot={0} />
            {ok.length > 0 && (
              <Points name="feasible evaluation" x={ok} y={pick(fits.fy, ok)} tone="success" size={8} />
            )}
            {bad.length > 0 && (
              <Points name="infeasible evaluation" x={bad} y={pick(fits.fy, bad)} tone="destructive" size={8} />
            )}
            <Segments
              name="next λ"
              segments={[{ from: [nextX, F_RANGE[0]], to: [nextX, F_RANGE[1]] }]}
              emphasis
              dashed
            />
          </Plot>
          <Plot x={lam} y={cAxis} title="constraint">
            <Area
              name="μ ± 2σ of c"
              x={fits.grid}
              y={fits.cPost.upper}
              base={fits.cPost.lower}
              slot={5}
              opacity={0.18}
              line={false}
            />
            <Curve name="constrained quantity c" x={fits.grid} y={fits.cTruth} muted dashed />
            <Curve name="posterior mean of c" x={fits.grid} y={fits.cPost.mean} slot={5} />
            <Points name="evaluations of c" x={fits.x} y={fits.cy} slot={5} size={7} />
            <Handle {...state.handle('tau', { axis: 'y', label: 'τ' })} />
          </Plot>
          <Plot x={lam} y={pAxis} title="probability of feasibility">
            <Area name="P(c ≤ τ)" x={fits.grid} y={r.pof} slot={5} />
          </Plot>
          <Plot x={lam} y={aAxis} title="acquisition">
            {r.ei && <Curve name="EI over best feasible" x={fits.grid} y={r.ei} muted dashed />}
            <Area name="EI × P(feasible)" x={fits.grid} y={r.product} slot={3} />
            <Points name="its maximum" x={[nextX]} y={[r.choice.value]} emphasis size={9} />
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
