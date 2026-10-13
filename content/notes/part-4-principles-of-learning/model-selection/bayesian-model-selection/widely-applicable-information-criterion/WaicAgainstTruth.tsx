import { Curve, Figure, float, Handle, int, Plot, Readout, useAxis, useComputed, useFigureState } from 'aifn-render'
import { psisLoo, waic } from 'aifn-compute/learning/selection'
import { drawData, fmt, logLikDraws, polynomialEvidence, trueElpd } from '../_shared/blr'

const DEGREES = Array.from({ length: 10 }, (_, d) => d)

/**
 * WAIC and PSIS-LOO from posterior draws of Bayesian polynomial regressions, beside the lppd they correct and the elpd
 * measured on a large fresh sample; and the effective numbers of parameters beside the coefficient count.
 */
export function WaicAgainstTruth() {
  const state = useFigureState({
    n: int(30, { ge: 8, le: 200, suggestions: [10, 30, 100], label: 'data points n' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ (known to the model)' }),
    alpha: float(1, {
      min: 0.001,
      max: 100,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1, 10],
      label: 'prior precision α of each weight',
    }),
    draws: int(1000, { ge: 100, le: 4000, suggestions: [500, 1000, 4000], label: 'posterior draws S' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    degree: int(3, { ge: 0, le: 9, label: 'degree shown' }),
  })
  const { n, noise, alpha, draws, seed } = state
  const run = useComputed(
    () => {
      const data = drawData(`waic/${seed}`, n, noise)
      const fresh = drawData('waic/fresh', 2000, noise)
      const beta = 1 / (noise * noise)
      return DEGREES.map((d) => {
        const { X, map, ev } = polynomialEvidence(data.x, data.y, d, alpha, beta)
        const L = logLikDraws(ev, X, data.y, draws, `waic/draws/${seed}/${d}`)
        const w = waic(L)
        const loo = psisLoo(L)
        return {
          lppd: w.lppd,
          waic: w.elpd,
          loo: loo.elpd,
          se: w.se,
          pWaic: w.p,
          pLoo: loo.p,
          gamma: ev.gamma,
          bad: loo.bad,
          truth: trueElpd(ev, map(fresh.x), fresh.y, n),
        }
      })
    },
    [n, noise, alpha, draws, seed],
    { mode: 'release' },
  )
  const rows = run.value
  const pick = (k: keyof (typeof rows)[number]) => rows.map((r) => r[k])
  const d = Math.max(0, Math.min(9, state.degree))
  const r = rows[d]
  const key = `${n}/${noise}/${alpha}/${seed}`
  const dx = useAxis({ label: 'polynomial degree', range: [-0.3, 9.3], nice: false })
  const ey = useAxis({ label: 'elpd (nats, higher is better)', hold: 'union', key })
  const py = useAxis({ label: 'effective number of parameters', range: [0, undefined], hold: 'union', key: `p/${key}` })
  const handle = (
    <Handle
      kind="x"
      at={d}
      label="degree"
      onDrag={(x) => state.set('degree', Math.max(0, Math.min(9, Math.round(x))))}
    />
  )
  return (
    <Figure
      title="WAIC and PSIS-LOO against the truth"
      state={state}
      hoverReadout={false}
      caption="Bayesian polynomial regression of degree 0 to 9 on n points of y = sin(πx) + σε, with a N(0, 1/α) prior on each coefficient and the noise known. From S posterior draws, the lppd scores the posterior predictive on the training points; WAIC and PSIS-LOO subtract their estimates of its optimism. The truth is the elpd of each posterior predictive on n new points, measured on 2,000 fresh points. Left: the three estimates and the truth. Right: the effective numbers of parameters p_WAIC and p_loo, beside MacKay's γ and the number of coefficients d + 1; once the degree fits the data and the prior is weak (small α) they track d + 1, and a strong prior pulls them below it. At degrees 0 and 1 the model misfits the sine badly, and they exceed d + 1: the penalty grows with the misfit, because a point the model cannot fit has a log-likelihood that varies a lot across the posterior. Drag the vertical line to read one degree."
      readouts={
        <>
          <Readout
            label={`degree ${d}: lppd / WAIC / PSIS-LOO / truth`}
            value={`${fmt(r.lppd)} / ${fmt(r.waic)} / ${fmt(r.loo)} / ${fmt(r.truth)}`}
          />
          <Readout label="standard error of the WAIC elpd" value={fmt(r.se)} />
          <Readout
            label={`degree ${d}: p_WAIC / p_loo / γ`}
            value={`${fmt(r.pWaic)} / ${fmt(r.pLoo)} / ${fmt(r.gamma)}`}
          />
          <Readout label={`degree ${d}: points with k̂ above the threshold`} value={`${r.bad}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={dx} y={ey} title="estimates of the elpd">
          <Curve name="lppd (no correction)" x={DEGREES} y={pick('lppd')} slot={0} stale={run.stale} />
          <Curve name="WAIC" x={DEGREES} y={pick('waic')} slot={1} stale={run.stale} />
          <Curve name="PSIS-LOO" x={DEGREES} y={pick('loo')} slot={2} stale={run.stale} />
          <Curve name="truth (fresh data)" x={DEGREES} y={pick('truth')} emphasis dashed stale={run.stale} />
          {handle}
        </Plot>
        <Plot x={dx} y={py} title="effective number of parameters">
          <Curve name="p_WAIC" x={DEGREES} y={pick('pWaic')} slot={1} stale={run.stale} />
          <Curve name="p_loo" x={DEGREES} y={pick('pLoo')} slot={2} stale={run.stale} />
          <Curve name="γ (MacKay)" x={DEGREES} y={pick('gamma')} slot={0} dashed stale={run.stale} />
          <Curve name="coefficients d + 1" x={DEGREES} y={DEGREES.map((k) => k + 1)} emphasis dashed />
          {handle}
        </Plot>
      </div>
    </Figure>
  )
}
