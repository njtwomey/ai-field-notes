import {
  choice,
  Curve,
  Figure,
  float,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { permutation, stream } from 'aifn-compute/foundation/random'
import { fromData, mean, stack, take, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { psisLoo, pseudoBmaPlusWeights, pseudoBmaWeights, stackingWeights } from 'aifn-compute/learning/selection'
import { softmax } from 'aifn-compute/numerics/special'
import { drawData, fmt, logLikDraws, polynomialEvidence, predictiveLogDensity, type Target } from '../_shared/blr'

const CANDIDATES = [1, 3, 5] as const
const SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
const LOG_SIZES = SIZES.map(Math.log10)
const TRUTHS = [
  { value: 'cubic', label: 'cubic x³ − x (degree 3 is true)' },
  { value: 'sine', label: 'sin(πx) (no candidate is true)' },
] as const
const WEIGHTS = [
  { value: 'bma', label: 'BMA (evidence)' },
  { value: 'pseudo', label: 'pseudo-BMA (PSIS-LOO)' },
  { value: 'pseudoPlus', label: 'pseudo-BMA+ (Bayesian bootstrap)' },
] as const
type Rule = (typeof WEIGHTS)[number]['value']
/** Bayesian-bootstrap replicates of pseudo-BMA+. */
const BOOTSTRAP = 1000
const flatOf = (t: Tensor) => Array.from(toFlat(t))

/**
 * Posterior model probabilities (or pseudo-BMA weights) and stacking weights of three polynomial regressions as the
 * sample grows, when one candidate generated the data and when none did, beside each candidate's held-out log
 * predictive density per point.
 */
export function ModelWeightsAgainstN() {
  const state = useFigureState({
    target: choice(TRUTHS, 'sine', { label: 'true regression function' }),
    weights: choice(WEIGHTS, 'bma', { label: 'weights' }),
    noise: float(0.2, { min: 0.05, max: 1, step: 0.05, label: 'noise σ (known to the models)' }),
    alpha: float(1, {
      min: 0.001,
      max: 100,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1, 10],
      label: 'prior precision α of each weight',
    }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    logN: float(Math.log10(64), {
      min: LOG_SIZES[0],
      max: LOG_SIZES[LOG_SIZES.length - 1],
      step: 0.01,
      label: 'log₁₀ n shown',
    }),
  })
  const { target, weights, noise, alpha, seed } = state
  const run = useComputed(
    () => {
      const all = drawData(`bma/${target}/${seed}`, SIZES[SIZES.length - 1], noise, target as Target)
      const fresh = drawData(`bma/${target}/fresh`, 2000, noise, target as Target)
      const beta = 1 / (noise * noise)
      // regression1d sorts its inputs, so a random order makes the first n points a uniform sample.
      const order = Array.from(toFlat(permutation(stream(`bma/order/${seed}`), SIZES[SIZES.length - 1])))
      return SIZES.map((n) => {
        const x = take(all.x, order.slice(0, n)) as Tensor
        const y = take(all.y, order.slice(0, n)) as Tensor
        const fits = CANDIDATES.map((d) => polynomialEvidence(x, y, d, alpha, beta))
        // The n × J matrix of leave-one-out log predictive densities, one PSIS-LOO column per model.
        const loo = stack(
          fits.map((f, k) => psisLoo(logLikDraws(f.ev, f.X, y, 1000, `bma/draws/${seed}/${n}/${k}`)).pointwise.elpd),
          1,
        )
        const evidence = fits.map((f) => f.ev.logEvidence)
        const w: Record<Rule, number[]> = {
          bma: flatOf(softmax(fromData(Float64Array.from(evidence), [evidence.length])) as Tensor),
          pseudo: flatOf(pseudoBmaWeights(loo).weights),
          pseudoPlus: flatOf(
            pseudoBmaPlusWeights(loo, { B: BOOTSTRAP, stream: stream(`bma/bootstrap/${seed}/${n}`) }).weights,
          ),
        }
        return {
          w,
          stacking: flatOf(stackingWeights(loo).weights),
          heldOut: fits.map((f) => Number(mean(predictiveLogDensity(f.ev, f.map(fresh.x), fresh.y)))),
        }
      })
    },
    [target, noise, alpha, seed],
    { mode: 'release' },
  )
  const rows = run.value
  const j = LOG_SIZES.reduce((b, l, i) => (Math.abs(l - state.logN) < Math.abs(LOG_SIZES[b] - state.logN) ? i : b), 0)
  const r = rows[j]
  const key = `${target}/${noise}/${alpha}/${seed}`
  const nx = useAxis({
    label: 'log₁₀ data points n',
    range: [LOG_SIZES[0] - 0.05, LOG_SIZES[LOG_SIZES.length - 1] + 0.05],
    nice: false,
  })
  const wy = useAxis({ label: 'model weight', range: [0, 1] })
  const hy = useAxis({ label: 'held-out log density per point', hold: 'union', key })
  const handle = (
    <Handle
      kind="x"
      at={LOG_SIZES[j]}
      label="n"
      onDrag={(x) => state.set('logN', Math.max(LOG_SIZES[0], Math.min(LOG_SIZES[LOG_SIZES.length - 1], x)))}
    />
  )
  const best = r.heldOut.reduce((b, x, i) => (x > r.heldOut[b] ? i : b), 0)
  const rule = weights as Rule
  const ruleLabel = WEIGHTS.find((w) => w.value === rule)?.label ?? ''
  const weightCurves = (of: (row: (typeof rows)[number]) => number[]) =>
    CANDIDATES.map((d, k) => (
      <Curve key={d} name={`degree ${d}`} x={LOG_SIZES} y={rows.map((row) => of(row)[k])} slot={k} stale={run.stale} />
    ))
  return (
    <Figure
      title="Model weights as the data grow"
      state={state}
      hoverReadout={false}
      defaultSize="L"
      caption="Three Bayesian polynomial regressions, of degree 1, 3 and 5, each with a N(0, 1/α) prior on its coefficients and the noise known, fitted to the first n points of one growing sample. Left: the weight of each model, by BMA (proportional to the evidence, with equal prior probabilities), by pseudo-BMA (proportional to exp of the PSIS-LOO elpd) or by pseudo-BMA+ (pseudo-BMA averaged over 1,000 Bayesian-bootstrap replicates of the elpd). Middle: the stacking weights, which maximise the PSIS-LOO log score of the mixture of the three predictive distributions. Right: the truth the weights should follow, each model's log predictive density per point on 2,000 fresh points. When the cubic generated the data, BMA moves towards degree 3, the simplest true model, even though degree 5 contains it; the pseudo-BMA weights keep a share on degree 5, which predicts almost as well. When the truth is a sine, no candidate is right, but degree 5 predicts best at large n, and every rule, stacking included, ends with nearly all its weight on it. Drag the vertical line to read one sample size."
      readouts={
        <>
          <Readout label="n" value={`${SIZES[j]}`} />
          {CANDIDATES.map((d, k) => (
            <Readout
              key={d}
              label={`degree ${d}: ${ruleLabel} / stacking / held-out log density`}
              value={`${fmt(r.w[rule][k])} / ${fmt(r.stacking[k])} / ${fmt(r.heldOut[k])}`}
            />
          ))}
          <Readout label="best held-out model" value={`degree ${CANDIDATES[best]}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Plot x={nx} y={wy} title={ruleLabel}>
          {weightCurves((row) => row.w[rule])}
          {handle}
        </Plot>
        <Plot x={nx} y={wy} title="stacking (PSIS-LOO)">
          {weightCurves((row) => row.stacking)}
          {handle}
        </Plot>
        <Plot x={nx} y={hy} title="truth: held-out log density">
          {CANDIDATES.map((d, k) => (
            <Curve
              key={d}
              name={`degree ${d}`}
              x={LOG_SIZES}
              y={rows.map((row) => row.heldOut[k])}
              slot={k}
              stale={run.stale}
            />
          ))}
          {handle}
        </Plot>
      </div>
    </Figure>
  )
}
