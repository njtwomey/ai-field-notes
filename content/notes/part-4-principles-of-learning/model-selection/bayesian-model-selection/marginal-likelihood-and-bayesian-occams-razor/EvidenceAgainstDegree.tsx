import {
  Curve,
  Figure,
  float,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linspace, reshape, type Tensor } from 'aifn-compute/foundation/tensor'
import { drawData, flat, fmt, polynomialEvidence, posteriorMean, trueElpd } from '../_shared/blr'

const DEGREES = Array.from({ length: 10 }, (_, d) => d)
const GRID = reshape(linspace(-1, 1, 161), [161, 1]) as Tensor
const GRID_X = flat(GRID)

/**
 * The log evidence of Bayesian polynomial regressions of degree 0 to 9, split into the best-fit log likelihood and the
 * log Occam factor, beside the elpd of each posterior predictive measured on a large fresh sample.
 */
export function EvidenceAgainstDegree() {
  const state = useFigureState({
    n: int(20, { ge: 5, le: 200, suggestions: [10, 20, 50, 100], label: 'data points n' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ (known to the model)' }),
    alpha: float(1, {
      min: 0.001,
      max: 100,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1, 10],
      label: 'prior precision α of each weight',
    }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    degree: int(3, { ge: 0, le: 9, label: 'degree shown' }),
  })
  const { n, noise, alpha, seed } = state
  const run = useComputed(
    () => {
      const data = drawData(`occam/${seed}`, n, noise)
      const fresh = drawData('occam/fresh', 2000, noise)
      const beta = 1 / (noise * noise)
      const rows = DEGREES.map((d) => {
        const { ev, map } = polynomialEvidence(data.x, data.y, d, alpha, beta)
        return {
          evidence: ev.logEvidence,
          fit: ev.bestFitLogLikelihood,
          occam: ev.logOccamFactor,
          gamma: ev.gamma,
          truth: trueElpd(ev, map(fresh.x), fresh.y, n),
          curve: flat(posteriorMean(ev, map(GRID))),
        }
      })
      return { x: flat(data.x), y: flat(data.y), rows }
    },
    [n, noise, alpha, seed],
    { mode: 'release' },
  )
  const v = run.value
  const pick = (k: 'evidence' | 'fit' | 'occam' | 'truth') => v.rows.map((r) => r[k])
  const argmax = (a: number[]) => a.reduce((b, x, i) => (x > a[b] ? i : b), 0)
  const bestEvidence = argmax(pick('evidence'))
  const bestTruth = argmax(pick('truth'))
  const d = Math.max(0, Math.min(9, state.degree))
  const row = v.rows[d]
  const dx = useAxis({ label: 'polynomial degree', range: [-0.3, 9.3], nice: false })
  const ly = useAxis({ label: 'log density (nats)', hold: 'union', key: `${n}/${noise}/${alpha}/${seed}` })
  const ey = useAxis({ label: 'log density (nats)', hold: 'union', key: `e/${n}/${noise}/${alpha}/${seed}` })
  const xx = useAxis({ label: 'x', range: [-1.05, 1.05], nice: false })
  const yy = useAxis({ label: 'y', range: [-2, 2] })
  const degreeHandle = (
    <Handle
      kind="x"
      at={d}
      label="degree"
      onDrag={(x) => state.set('degree', Math.max(0, Math.min(9, Math.round(x))))}
    />
  )
  return (
    <Figure
      title="The evidence against polynomial degree"
      state={state}
      hoverReadout={false}
      caption="Bayesian polynomial regression of degree 0 to 9 on n points of y = sin(πx) + σε, with a N(0, 1/α) prior on every coefficient and the noise level known. Left: the log evidence (thick) is the best-fit log likelihood, which rises with the degree, plus the log Occam factor, which falls as each new coefficient is pinned down by the data. Middle: the log evidence beside the truth it competes with, the elpd of each posterior predictive on n new points, measured on 2,000 fresh points. Right: the data and the posterior mean at the degree marked. Drag the vertical line to change the degree. A larger α (a narrower prior) makes extra coefficients cheaper, because the prior already concentrates where the data put them."
      readouts={
        <>
          <Readout label="degree with the largest evidence" value={`${bestEvidence}`} />
          <Readout label="degree with the best held-out elpd" value={`${bestTruth}`} />
          <Readout
            label={`degree ${d}: log evidence = fit + Occam`}
            value={`${fmt(row.evidence)} = ${fmt(row.fit)} + (${fmt(row.occam)})`}
          />
          <Readout label={`degree ${d}: effective parameters γ of ${d + 1}`} value={fmt(row.gamma)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Plot x={dx} y={ly} title="evidence = best fit + Occam factor">
          <Curve name="best-fit log likelihood" x={DEGREES} y={pick('fit')} slot={0} stale={run.stale} />
          <Curve name="log Occam factor" x={DEGREES} y={pick('occam')} slot={1} stale={run.stale} />
          <Curve name="log evidence" x={DEGREES} y={pick('evidence')} emphasis stale={run.stale} />
          {degreeHandle}
        </Plot>
        <Plot x={dx} y={ey} title="evidence against the truth">
          <Curve name="log evidence" x={DEGREES} y={pick('evidence')} emphasis stale={run.stale} />
          <Curve name="held-out elpd (truth)" x={DEGREES} y={pick('truth')} slot={2} dashed stale={run.stale} />
          {degreeHandle}
        </Plot>
        <Plot x={xx} y={yy} title={`posterior mean, degree ${d}`}>
          <Points name="data" x={v.x} y={v.y} slot={0} />
          <Curve name="posterior mean" x={GRID_X} y={row.curve} slot={1} stale={run.stale} />
        </Plot>
      </div>
    </Figure>
  )
}
