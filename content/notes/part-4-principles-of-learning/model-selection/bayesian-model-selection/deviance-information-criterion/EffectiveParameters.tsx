import { Curve, Figure, float, Handle, int, Plot, Readout, useAxis, useComputed, useFigureState } from 'aifn-render'
import { dic, waic } from 'aifn-compute/learning/selection'
import { drawData, fmt, logLikAtMean, logLikDraws, polynomialEvidence } from '../_shared/blr'

const LOG_ALPHAS = Array.from({ length: 25 }, (_, i) => -3 + i * 0.25)

/**
 * The effective number of parameters of a Bayesian polynomial regression as the prior tightens: DIC's p_D and p_V and
 * WAIC's p_WAIC from posterior draws, beside the exact γ of the linear-Gaussian model and the coefficient count.
 */
export function EffectiveParameters() {
  const state = useFigureState({
    degree: int(7, { ge: 1, le: 12, suggestions: [3, 7, 10], label: 'polynomial degree' }),
    n: int(30, { ge: 10, le: 200, suggestions: [15, 30, 100], label: 'data points n' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ (known to the model)' }),
    draws: int(1000, { ge: 100, le: 4000, suggestions: [500, 1000, 4000], label: 'posterior draws S' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    logAlpha: float(0, { min: -3, max: 3, step: 0.25, label: 'log₁₀ α shown' }),
  })
  const { degree, n, noise, draws, seed } = state
  const run = useComputed(
    () => {
      const data = drawData(`dic/${seed}`, n, noise)
      const beta = 1 / (noise * noise)
      return LOG_ALPHAS.map((la) => {
        const { X, ev } = polynomialEvidence(data.x, data.y, degree, 10 ** la, beta)
        const L = logLikDraws(ev, X, data.y, draws, `dic/draws/${seed}/${la}`)
        const d = dic(L, logLikAtMean(ev, X, data.y))
        return { pD: d.pD, pV: d.pV, dic: d.dic, pWaic: waic(L).p, gamma: ev.gamma }
      })
    },
    [degree, n, noise, draws, seed],
    { mode: 'release' },
  )
  const rows = run.value
  const pick = (k: 'pD' | 'pV' | 'pWaic' | 'gamma') => rows.map((r) => r[k])
  const j = Math.max(0, Math.min(LOG_ALPHAS.length - 1, Math.round((state.logAlpha + 3) * 4)))
  const r = rows[j]
  const ax = useAxis({ label: 'log₁₀ prior precision α', range: [-3, 3], nice: false })
  const py = useAxis({
    label: 'effective number of parameters',
    range: [0, undefined],
    hold: 'union',
    key: `${degree}`,
  })
  return (
    <Figure
      title="Effective parameters as the prior tightens"
      state={state}
      hoverReadout={false}
      caption="A Bayesian polynomial regression of fixed degree on n points of y = sin(πx) + σε, with a N(0, 1/α) prior on each of its d + 1 coefficients. Against the prior precision α: DIC's p_D (posterior mean deviance minus the deviance at the posterior mean) and p_V (half the posterior variance of the deviance), WAIC's p_WAIC, all from S posterior draws, and the exact effective number γ of the linear-Gaussian model. A vague prior (left) leaves every coefficient to the data and the counts approach d + 1; a tight prior (right) fixes the coefficients and the counts fall towards zero. p_D tracks γ closely here, because the posterior is Gaussian and the posterior mean is a good summary; p_V does not, because the prior pulls the posterior mean away from the best fit. Drag the vertical line to read one value of α."
      readouts={
        <>
          <Readout label="log₁₀ α" value={`${LOG_ALPHAS[j]}`} />
          <Readout
            label="p_D / p_V / p_WAIC / γ"
            value={`${fmt(r.pD)} / ${fmt(r.pV)} / ${fmt(r.pWaic)} / ${fmt(r.gamma)}`}
          />
          <Readout label="DIC" value={fmt(r.dic, 4)} />
        </>
      }
    >
      <Plot x={ax} y={py}>
        <Curve name="p_D" x={LOG_ALPHAS} y={pick('pD')} slot={0} stale={run.stale} />
        <Curve name="p_V" x={LOG_ALPHAS} y={pick('pV')} slot={1} stale={run.stale} />
        <Curve name="p_WAIC" x={LOG_ALPHAS} y={pick('pWaic')} slot={2} stale={run.stale} />
        <Curve name="γ (exact)" x={LOG_ALPHAS} y={pick('gamma')} emphasis dashed stale={run.stale} />
        <Curve name="coefficients d + 1" x={[-3, 3]} y={[degree + 1, degree + 1]} emphasis dashed />
        <Handle
          kind="x"
          at={LOG_ALPHAS[j]}
          label="α"
          onDrag={(x) => state.set('logAlpha', Math.max(-3, Math.min(3, Math.round(x * 4) / 4)))}
        />
      </Plot>
    </Figure>
  )
}
