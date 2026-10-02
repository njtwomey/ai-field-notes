import { grad } from 'aifn/foundation/autodiff'
import {
  approxNdcg,
  lambdaRank,
  listMle,
  listNet,
  listwiseSoftmax,
  pairwiseHinge,
  pointwiseBce,
  pointwiseSquaredError,
  rankNet,
} from 'aifn-applied/retrieval/losses'
import {
  huber,
  logCosh,
  meanAbsoluteErrorLoss,
  meanSquaredErrorLoss,
  pinball,
  surrogates,
  type SurrogateName,
} from 'aifn/learning/losses'
import { linspace, tensor, toFlat, unwrap, zeros, sum, type Tensor, type Value } from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Figure } from '@lab/layout'
import { choice, row, setting, slider, useFigureState } from '@lab/state'
import { Bars, Curve, Handle, Plot, Plots, Points, Readout, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)
const flat = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const num = (v: Value) => flat(v)[0]

// ---------------------------------------------------------------------------------------------------------------------
// 1. Classification surrogates as functions of the margin.

const SURROGATES: { key: SurrogateName; label: string; slot: number }[] = [
  { key: 'zeroOne', label: '0–1', slot: 0 },
  { key: 'hinge', label: 'hinge', slot: 1 },
  { key: 'squaredHinge', label: 'squared hinge', slot: 2 },
  { key: 'logistic', label: 'logistic (bits)', slot: 3 },
  { key: 'exponential', label: 'exponential', slot: 4 },
  { key: 'modifiedHuber', label: 'modified Huber', slot: 5 },
]
const MARGINS = linspace(-2.5, 2.5, 401)
const MX = toFlat(MARGINS)

export function MarginLossesSpecimen() {
  const state = useFigureState({
    shown: row('1 · surrogates', {
      zeroOne: setting(true, '0–1'),
      hinge: setting(true, 'hinge'),
      squaredHinge: setting(false, 'squared hinge'),
      logistic: setting(true, 'logistic (bits)'),
      exponential: setting(true, 'exponential'),
      modifiedHuber: setting(false, 'modified Huber'),
    }),
    m: slider(-2.5, 2.5, -0.5, { onChart: true, label: 'margin m' }),
  })
  const shown: Record<SurrogateName, boolean> = state.shown
  const m = state.m
  const curves = useMemo(
    () =>
      SURROGATES.map((s) => ({
        ...s,
        y: flat(surrogates[s.key](MARGINS)),
        dy: flat(grad((v: Value) => sum(surrogates[s.key](v)))(MARGINS) as Tensor),
      })),
    [],
  )
  const active = curves.filter((c) => shown[c.key])
  const mx = useAxis({ label: 'margin m', range: [-2.5, 2.5] })
  const phi = useAxis({ label: 'φ(m)', range: [0, 4] })
  const dphi = useAxis({ label: 'φ′(m)', range: [-3, 0.5] })
  return (
    <Figure
      title="Classification losses as functions of the margin"
      purpose="Every convex surrogate lies above the 0–1 loss and passes through (0, 1); they differ in how hard they push on badly misclassified points (m ≪ 0) and whether they stop pushing once m > 1."
      defaultSize="L"
      state={state}
      readouts={{
        'at m': (
          <>
            <Readout label="margin m = y·f(x)" value={fmt(m)} />
            {active.map((c) => (
              <Readout
                key={c.key}
                label={`${c.label}: φ, φ′`}
                value={`${fmt(num(surrogates[c.key](m)))}, ${fmt(num(grad((v: Value) => surrogates[c.key](v))(m) as number))}`}
              />
            ))}
          </>
        ),
      }}
      caption="aifn/learning/losses surrogates, φ(m) above and its derivative from aifn/autodiff below. Drag the margin. The exponential loss's slope grows without bound for m < 0, so outliers dominate it; the hinge's slope is −1 there and 0 beyond m = 1; the logistic loss never quite stops pushing."
    >
      <Plots rows={2} heights={[3, 2]} hoverGroup>
        <Plot x={mx} y={phi}>
          {active.map((c) => (
            <Curve key={c.key} name={c.label} x={MX} y={c.y} slot={c.slot} />
          ))}
          <Handle {...state.handle('m', { label: 'm' })} />
        </Plot>
        <Plot x={mx} y={dphi}>
          {active.map((c) => (
            <Curve key={c.key} name={c.label} x={MX} y={c.dy} slot={c.slot} />
          ))}
          <Handle {...state.handle('m', { label: 'm' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Regression losses of the residual.

const RESIDUALS = linspace(-4, 4, 401)
const RX = toFlat(RESIDUALS)
const ZERO = zeros([401])

export function RegressionLossesSpecimen() {
  const state = useFigureState({
    shape: row('1 · shape parameters', {
      delta: slider(0.1, 3, 1, { label: 'Huber δ', step: 0.1 }),
      tau: slider(0.05, 0.95, 0.8, { label: 'pinball τ', step: 0.05 }),
    }),
  })
  const { delta, tau } = state.shape
  const series = useMemo(() => {
    const none = { reduction: 'none' as const }
    // Predictions ŷ = r against targets 0, so the residual ŷ − y is r.
    const curves: [string, Value][] = [
      ['squared', meanSquaredErrorLoss(RESIDUALS, ZERO, none)],
      ['absolute', meanAbsoluteErrorLoss(RESIDUALS, ZERO, none)],
      [`Huber (δ = ${fmt(delta)})`, huber(RESIDUALS, ZERO, { ...none, delta })],
      ['log-cosh', logCosh(RESIDUALS, ZERO, none)],
      [`pinball (τ = ${fmt(tau)})`, pinball(RESIDUALS, ZERO, { ...none, quantile: tau })],
    ]
    return curves.map(([name, y], slot) => ({ name, y: flat(y), slot }))
  }, [delta, tau])
  const rx = useAxis({ label: 'residual ŷ − y', range: [-4, 4] })
  const ly = useAxis({ label: 'loss', range: [0, 4] })
  return (
    <Figure
      title="Regression losses of the residual"
      purpose="Squared loss grows quadratically, so one outlier can dominate it; Huber and log-cosh are quadratic near zero and linear beyond; the pinball loss tilts the absolute loss so that its minimiser is a quantile."
      state={state}
      caption="aifn/learning/losses with reduction 'none', as functions of the prediction minus the target. The pinball loss costs τ per unit of under-prediction (left) and 1 − τ per unit of over-prediction (right)."
    >
      <Plot x={rx} y={ly}>
        {series.map((c) => (
          <Curve key={c.slot} name={c.name} x={RX} y={c.y} slot={c.slot} />
        ))}
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Ranking losses on a toy list.

const GRADES = [3, 2, 2, 1, 0, 0]
type RankingId =
  | 'pointwiseBce'
  | 'pointwiseSquaredError'
  | 'rankNet'
  | 'pairwiseHinge'
  | 'lambdaRank'
  | 'listwiseSoftmax'
  | 'listNet'
  | 'listMle'
  | 'approxNdcg'
const RANKING: { value: RankingId; label: string; f: (s: Value, r: number[]) => Value }[] = [
  { value: 'pointwiseBce', label: 'pointwise: binary cross-entropy', f: pointwiseBce },
  { value: 'pointwiseSquaredError', label: 'pointwise: squared error', f: pointwiseSquaredError },
  { value: 'rankNet', label: 'pairwise: RankNet', f: rankNet },
  { value: 'pairwiseHinge', label: 'pairwise: hinge (RankSVM)', f: pairwiseHinge },
  { value: 'lambdaRank', label: 'pairwise: LambdaRank', f: lambdaRank },
  { value: 'listwiseSoftmax', label: 'listwise: softmax cross-entropy', f: listwiseSoftmax },
  { value: 'listNet', label: 'listwise: ListNet', f: listNet },
  { value: 'listMle', label: 'listwise: ListMLE', f: listMle },
  { value: 'approxNdcg', label: 'listwise: ApproxNDCG', f: approxNdcg },
]
const POSITIONS = GRADES.map((_, i) => i)
const ITEM_NAMES = GRADES.map((g, i) => `${i + 1} (grade ${g})`)

export function RankingLossesSpecimen() {
  const state = useFigureState({
    loss: row('1 · loss', {
      id: choice(
        RANKING.map(({ value, label }) => ({ value, label })),
        'lambdaRank',
        { label: 'loss' },
      ),
    }),
    // The six scores: each moved by its handle.
    s0: slider(-3, 3, 0.2, { onChart: true }),
    s1: slider(-3, 3, 1.1, { onChart: true }),
    s2: slider(-3, 3, -0.3, { onChart: true }),
    s3: slider(-3, 3, 0.8, { onChart: true }),
    s4: slider(-3, 3, 0.5, { onChart: true }),
    s5: slider(-3, 3, -0.9, { onChart: true }),
  })
  const scores = useMemo(
    () => [state.s0, state.s1, state.s2, state.s3, state.s4, state.s5],
    [state.s0, state.s1, state.s2, state.s3, state.s4, state.s5],
  )
  const id: RankingId = state.loss.id
  const loss = RANKING.find((r) => r.value === id)!
  const s = useMemo(() => tensor(scores), [scores])
  const push = useMemo(() => toFlat(grad((v: Value) => loss.f(v, GRADES))(s) as Tensor).map((g) => -g), [s, loss])
  const values = useMemo(() => RANKING.map((r) => ({ label: r.label, v: num(r.f(s, GRADES)) })), [s])
  const item = useAxis({ label: 'item', categories: ITEM_NAMES })
  const sy = useAxis({ label: 'score (bars: grade)', range: [-3, 3.5] })
  const py = useAxis({ label: 'push', hold: 'union', key: id })
  return (
    <Figure
      title="Ranking losses on a toy list"
      purpose="Each ranking loss pushes item scores in the direction that lowers it: pointwise losses toward each item's own target, pairwise losses apart for misordered pairs, listwise losses by the softmax of the whole list; LambdaRank weights each pair by the NDCG it would gain."
      defaultSize="L"
      state={state}
      readouts={{
        'every loss at these scores': values.map((v) => <Readout key={v.label} label={v.label} value={fmt(v.v)} />),
      }}
      caption={`Six items with relevance grades ${GRADES.join(', ')} (grey bars). Top: their scores; drag any score up or down. Bottom: minus the gradient of the chosen loss (aifn/autodiff through aifn-applied/retrieval/losses), the push each score gets from one gradient step.`}
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={item} y={sy}>
          <Bars name="relevance grade" x={POSITIONS} y={GRADES} muted />
          <Points name="score" x={POSITIONS} y={scores} slot={0} live />
          {POSITIONS.map((i) => (
            <Handle
              key={i}
              kind="point"
              at={[i, scores[i]]}
              label={`item ${i + 1}`}
              onDrag={([, y]) => state.set(`s${i}`, y)}
            />
          ))}
        </Plot>
        <Plot x={item} y={py}>
          <Bars name="−∂loss/∂score" x={POSITIONS} y={push} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
