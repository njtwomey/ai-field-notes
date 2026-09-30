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
import { useMemo, useState } from 'react'
import { ControlRow, Figure } from '@lab/layout'
import { Select, Slider, Switch } from '@lab/controls'
import { Panel, Readout, Subplots, XYChart, formatNumber, type Handle, type XYSeries } from '@lab/viz'

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
  const [shown, setShown] = useState<Record<SurrogateName, boolean>>({
    zeroOne: true,
    hinge: true,
    squaredHinge: false,
    logistic: true,
    exponential: true,
    modifiedHuber: false,
  })
  const [m, setM] = useState(-0.5)
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
  const values: XYSeries[] = active.map((c) => ({ name: c.label, type: 'line', x: MX, y: c.y, slot: c.slot }))
  const slopes: XYSeries[] = active.map((c) => ({ name: c.label, type: 'line', x: MX, y: c.dy, slot: c.slot }))
  const handle: Handle[] = [{ kind: 'x', at: m, label: 'm', onDrag: (x) => setM(Math.max(-2.5, Math.min(2.5, x))) }]
  return (
    <Figure
      title="Classification losses as functions of the margin"
      description="Every convex surrogate lies above the 0–1 loss and passes through (0, 1); they differ in how hard they push on badly misclassified points (m ≪ 0) and whether they stop pushing once m > 1."
      defaultSize="L"
      controls={
        <ControlRow label="1 · surrogates">
          {SURROGATES.map((s) => (
            <Switch
              key={s.key}
              label={s.label}
              checked={shown[s.key]}
              onChange={(v) => setShown((o) => ({ ...o, [s.key]: v }))}
            />
          ))}
        </ControlRow>
      }
      readouts={
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
      }
      caption="aifn/losses surrogates, φ(m) above and its derivative from aifn/autodiff below. Drag the margin. The exponential loss's slope grows without bound for m < 0, so outliers dominate it; the hinge's slope is −1 there and 0 beyond m = 1; the logistic loss never quite stops pushing."
    >
      <Subplots rows={2} sharex heightRatios={[3, 2]} hoverGroup>
        <Panel>
          <XYChart series={values} yLabel="φ(m)" yRange={[0, 4]} handles={handle} />
        </Panel>
        <Panel>
          <XYChart series={slopes} xLabel="margin m" yLabel="φ′(m)" yRange={[-3, 0.5]} handles={handle} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Regression losses of the residual.

const RESIDUALS = linspace(-4, 4, 401)
const RX = toFlat(RESIDUALS)
const ZERO = zeros([401])

export function RegressionLossesSpecimen() {
  const [delta, setDelta] = useState(1)
  const [tau, setTau] = useState(0.8)
  const series = useMemo<XYSeries[]>(() => {
    const none = { reduction: 'none' as const }
    // Predictions ŷ = r against targets 0, so the residual ŷ − y is r.
    const curves: [string, Value][] = [
      ['squared', meanSquaredErrorLoss(RESIDUALS, ZERO, none)],
      ['absolute', meanAbsoluteErrorLoss(RESIDUALS, ZERO, none)],
      [`Huber (δ = ${fmt(delta)})`, huber(RESIDUALS, ZERO, { ...none, delta })],
      ['log-cosh', logCosh(RESIDUALS, ZERO, none)],
      [`pinball (τ = ${fmt(tau)})`, pinball(RESIDUALS, ZERO, { ...none, quantile: tau })],
    ]
    return curves.map(([name, y], slot) => ({ name, type: 'line', x: RX, y: flat(y), slot }))
  }, [delta, tau])
  return (
    <Figure
      title="Regression losses of the residual"
      description="Squared loss grows quadratically, so one outlier can dominate it; Huber and log-cosh are quadratic near zero and linear beyond; the pinball loss tilts the absolute loss so that its minimiser is a quantile."
      controls={
        <ControlRow label="1 · shape parameters">
          <Slider label="Huber δ" value={delta} min={0.1} max={3} step={0.1} onChange={setDelta} />
          <Slider label="pinball τ" value={tau} min={0.05} max={0.95} step={0.05} onChange={setTau} />
        </ControlRow>
      }
      caption="aifn/losses with reduction 'none', as functions of the prediction minus the target. The pinball loss costs τ per unit of under-prediction (left) and 1 − τ per unit of over-prediction (right)."
    >
      <XYChart series={series} xLabel="residual ŷ − y" yLabel="loss" yRange={[0, 4]} rescaleOnChange={false} />
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
const ITEMS = GRADES.map((_, i) => i + 1)

export function RankingLossesSpecimen() {
  const [scores, setScores] = useState([0.2, 1.1, -0.3, 0.8, 0.5, -0.9])
  const [id, setId] = useState<RankingId>('lambdaRank')
  const loss = RANKING.find((r) => r.value === id)!
  const s = useMemo(() => tensor(scores), [scores])
  const push = useMemo(() => toFlat(grad((v: Value) => loss.f(v, GRADES))(s) as Tensor).map((g) => -g), [s, loss])
  const values = useMemo(() => RANKING.map((r) => ({ label: r.label, v: num(r.f(s, GRADES)) })), [s])
  const handles: Handle[] = scores.map((v, i) => ({
    kind: 'point',
    at: [i + 1, v],
    label: `item ${i + 1}`,
    onDrag: ([, y]) => setScores((old) => old.map((o, k) => (k === i ? Math.max(-3, Math.min(3, y)) : o))),
  }))
  const scoreSeries: XYSeries[] = [
    { name: 'relevance grade', type: 'bar', x: ITEMS, y: GRADES, muted: true },
    { name: 'score', type: 'scatter', x: ITEMS, y: scores, slot: 0 },
  ]
  const pushSeries: XYSeries[] = [{ name: '−∂loss/∂score', type: 'bar', x: ITEMS, y: push, slot: 1 }]
  return (
    <Figure
      title="Ranking losses on a toy list"
      description="Each ranking loss pushes item scores in the direction that lowers it: pointwise losses toward each item's own target, pairwise losses apart for misordered pairs, listwise losses by the softmax of the whole list; LambdaRank weights each pair by the NDCG it would gain."
      defaultSize="L"
      controls={
        <ControlRow label="1 · loss">
          <Select
            label="loss"
            value={id}
            onChange={setId}
            options={RANKING.map(({ value, label }) => ({ value, label }))}
          />
        </ControlRow>
      }
      readouts={values.map((v) => (
        <Readout key={v.label} label={v.label} value={fmt(v.v)} />
      ))}
      caption={`Six items with relevance grades ${GRADES.join(', ')} (grey bars). Top: their scores; drag any score up or down. Bottom: minus the gradient of the chosen loss (aifn/autodiff through aifn/losses), the push each score gets from one gradient step.`}
    >
      <Subplots rows={2} heightRatios={[3, 2]}>
        <Panel>
          <XYChart
            series={scoreSeries}
            integerX
            yLabel="score (bars: grade)"
            xRange={[0.5, 6.5]}
            yRange={[-3, 3.5]}
            handles={handles}
          />
        </Panel>
        <Panel>
          <XYChart series={pushSeries} integerX xLabel="item" yLabel="push" xRange={[0.5, 6.5]} />
        </Panel>
      </Subplots>
    </Figure>
  )
}
