import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  bayesianOptimisationSteps,
  expectedImprovement,
  maxValueEntropySearch,
  probabilityOfImprovement,
  upperConfidenceBound,
} from 'aifn-methods/optimisation/bayesian'
import { forrester } from 'aifn-methods/data/objectives'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { FixedHeight } from '../_shared/FixedHeight'

const F = forrester()
const GRID = 201

/** A score rescaled to [0, 1] over the grid, for drawing several acquisitions on one axis, and its argmax. */
function rescaled(score: Tensor) {
  const a = toFlat(score)
  let lo = Infinity
  let hi = -Infinity
  let arg = 0
  a.forEach((v, i) => {
    lo = Math.min(lo, v)
    if (v > hi) {
      hi = v
      arg = i
    }
  })
  const span = hi - lo || 1
  return { y: a.map((v) => (v - lo) / span), arg }
}

/** Five acquisition functions scoring the same Gaussian-process posterior on the Forrester function. */
export function AcquisitionGallery() {
  const state = useFigureState({
    n: int(4, { label: 'evaluations t', min: 2, max: 15, suggestions: [3, 4, 6, 10] }),
    seed: int(3, { label: 'seed (design and draws)', min: 0, max: 99, suggestions: [1, 2, 3, 4] }),
    xi: float(0, { label: 'margin ξ of PI and EI', ge: 0, le: 5, suggestions: [0, 0.1, 0.5, 1] }),
    beta: float(4, { label: 'confidence weight β', ge: 0, le: 100, suggestions: [0.25, 1, 4, 16] }),
  })

  const result = useMemo(() => {
    // Thompson sampling's state also carries the posterior mean and sd, and the draw that chose its point.
    const alg = bayesianOptimisationSteps(F.value, F, { acquisition: 'thompson', init: state.n, grid: GRID })
    const s = run(alg, undefined, 0, { stream: stream(state.seed) })
    const grid = toFlat(alg.grid)
    const options = { xi: state.xi }
    const scores = {
      pi: rescaled(probabilityOfImprovement(s.mean, s.sd, s.best, options)),
      ei: rescaled(expectedImprovement(s.mean, s.sd, s.best, options)),
      lcb: rescaled(upperConfidenceBound(s.mean, s.sd, { beta: state.beta })),
      mes: rescaled(
        maxValueEntropySearch(s.mean, s.sd, { samples: 10, stream: stream(`mes-${state.seed}`), best: s.best }),
      ),
    }
    const mean = toFlat(s.mean)
    const sd = toFlat(s.sd)
    return {
      grid,
      truth: grid.map((x) => F.value(tensor([x]))),
      mean,
      upper: mean.map((m, i) => m + 2 * sd[i]),
      lower: mean.map((m, i) => m - 2 * sd[i]),
      draw: s.draw ? toFlat(s.draw) : mean,
      drawArg: s.nextIndex,
      xs: toFlat(s.x),
      ys: toFlat(s.y),
      best: s.best,
      scores,
    }
  }, [state.n, state.seed, state.xi, state.beta])

  const r = result
  const lam = useAxis({ label: 'hyperparameter λ', range: [0, 1] })
  const fAxis = useAxis({ label: 'f(λ)', range: [-12, 20] })
  const sAxis = useAxis({ label: 'score, rescaled to [0, 1]', range: [0, 1.05] })
  const lines = [
    { key: 'pi', name: 'probability of improvement', slot: 6 },
    { key: 'ei', name: 'expected improvement', slot: 3 },
    { key: 'lcb', name: 'confidence bound', slot: 4 },
    { key: 'mes', name: 'max-value entropy search', slot: 7 },
  ] as const
  return (
    <Figure
      title="Four acquisitions and a Thompson draw on one posterior"
      state={state}
      caption="Top: the Forrester objective (dashed), the Gaussian-process posterior after t evaluations (mean and ±2 sd band), the incumbent, and one joint draw from the posterior with its minimum marked: Thompson sampling would evaluate there. Bottom: probability of improvement, expected improvement, the confidence bound −(μ − √β σ) and max-value entropy search over the 201 candidates, each rescaled to [0, 1] so their shapes can be compared; the dots mark each one's maximum, the point it would evaluate next. Raise ξ to move PI and EI away from the incumbent, raise β to widen the confidence bound, and change the seed for another design and another draw."
      readouts={
        <>
          <Readout label="incumbent f*ₜ" value={formatNumber(r.best)} />
          <Readout label="PI chooses λ" value={formatNumber(r.grid[r.scores.pi.arg])} />
          <Readout label="EI chooses λ" value={formatNumber(r.grid[r.scores.ei.arg])} />
          <Readout label="bound chooses λ" value={formatNumber(r.grid[r.scores.lcb.arg])} />
          <Readout label="MES chooses λ" value={formatNumber(r.grid[r.scores.mes.arg])} />
          <Readout label="Thompson chooses λ" value={formatNumber(r.grid[r.drawArg])} />
        </>
      }
    >
      <FixedHeight height={520}>
        <Plots rows={2} heights={[3, 2]}>
          <Plot x={lam} y={fAxis} title="posterior">
            <Area name="μₜ ± 2σₜ" x={r.grid} y={r.upper} base={r.lower} slot={0} opacity={0.18} line={false} />
            <Curve name="objective f" x={r.grid} y={r.truth} muted dashed />
            <Curve name="posterior mean μₜ" x={r.grid} y={r.mean} slot={0} />
            <Curve name="posterior draw" x={r.grid} y={r.draw} slot={2} width={1.5} />
            <Curve name="incumbent f*ₜ" x={[0, 1]} y={[r.best, r.best]} muted />
            <Points name="evaluations" x={r.xs} y={r.ys} slot={1} size={8} />
            <Points name="minimum of the draw" x={[r.grid[r.drawArg]]} y={[r.draw[r.drawArg]]} emphasis size={9} />
          </Plot>
          <Plot x={lam} y={sAxis} title="acquisitions">
            {lines.map((l) => (
              <Curve key={l.key} name={l.name} x={r.grid} y={r.scores[l.key].y} slot={l.slot} />
            ))}
            {lines.map((l) => (
              <Points
                key={`${l.key}-max`}
                name={l.name}
                x={[r.grid[r.scores[l.key].arg]]}
                y={[1]}
                slot={l.slot}
                size={9}
              />
            ))}
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
