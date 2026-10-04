/**
 * Resampling for imbalanced classes: two overlapping Gaussian classes with a rare positive class, resampled by random
 * over- or under-sampling, SMOTE, borderline-SMOTE, ADASYN or Tomek-link cleaning (`aifn-methods/learning/preprocessing`),
 * then a logistic regression fitted to the original and to the resampled data, scored on a large held-out sample
 * with the same imbalance (`aifn/learning/metrics`).
 */
import { useMemo } from 'react'
import { gaussians } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import {
  adasyn,
  borderlineSmote,
  borderStatus,
  randomOverSample,
  randomUnderSample,
  removeTomekLinks,
  smote,
  type Resampled,
} from 'aifn-methods/learning/preprocessing'
import { child, stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { dataset } from 'aifn/learning/estimators'
import { auroc, averagePrecision, balancedAccuracy, precision, recall } from 'aifn/learning/metrics'
import { Figure } from '@lab/layout'
import { choice, float, int, row, setting, slider, useComputed, useFigureState } from '@lab/state'
import { Contours, formatNumber, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const METHODS = [
  { value: 'none', label: 'none (original data)' },
  { value: 'over', label: 'random over-sampling' },
  { value: 'under', label: 'random under-sampling' },
  { value: 'smote', label: 'SMOTE' },
  { value: 'borderline', label: 'borderline-SMOTE' },
  { value: 'adasyn', label: 'ADASYN' },
  { value: 'tomek', label: 'Tomek-link cleaning' },
] as const
type Method = (typeof METHODS)[number]['value']

const BOX = 4.5
const G = 60
const GRID_X = Array.from({ length: G }, (_, i) => -BOX + (2 * BOX * i) / (G - 1))
const GRID = fromData(
  Float64Array.from({ length: G * G * 2 }, (_, k) => {
    const cell = k >> 1
    return k % 2 === 0 ? GRID_X[cell % G] : GRID_X[Math.floor(cell / G)]
  }),
  [G * G, 2],
)

function resample(method: Method, s: ReturnType<typeof stream>, x: Tensor, y: Tensor, k: number): Resampled | null {
  switch (method) {
    case 'none':
      return null
    case 'over':
      return randomOverSample(s, x, y)
    case 'under':
      return randomUnderSample(s, x, y)
    case 'smote':
      return smote(s, x, y, { k })
    case 'borderline':
      return borderlineSmote(s, x, y, { k })
    case 'adasyn':
      return adasyn(s, x, y, { k })
    case 'tomek':
      return removeTomekLinks(x, y)
  }
}

/** Points of a row-major [n × 2] tensor at the given rows, as columns. */
const pick = (x: Float64Array, rows: readonly number[]) => ({
  x: rows.map((i) => x[2 * i]),
  y: rows.map((i) => x[2 * i + 1]),
})

export function ResamplingSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      majority: int(400, { ge: 10, le: 5000, suggestions: [200, 400, 1000], label: 'majority points' }),
      minority: int(25, { ge: 2, le: 2000, suggestions: [10, 25, 50, 100], label: 'minority points' }),
      separation: slider(0.5, 5, 2, { step: 0.1, label: 'class separation d′' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    method: row('2 · resampling', {
      method: choice(METHODS, 'smote', { label: 'method' }),
      k: int(5, { ge: 1, le: 30, suggestions: [1, 3, 5, 10], label: 'neighbours k (SMOTE family)' }),
      l2: float(1, { ge: 0, suggestions: [0, 0.1, 1, 10], label: 'logistic L2 penalty' }),
    }),
    show: row('3 · show', {
      parents: setting(true, 'segments to the parents'),
      border: setting(false, 'borderline status'),
    }),
  })
  const { majority, minority, separation, seed } = state.data
  const { method, k, l2 } = state.method
  const data = useMemo(() => {
    const train = gaussians(stream(`imbalance/${seed}`), { n: [majority, minority], separation })
    const test = gaussians(child(stream(`imbalance/${seed}`), 'test'), {
      n: [20 * majority, 20 * minority],
      separation,
    })
    return { train, test }
  }, [majority, minority, separation, seed])
  const result = useComputed(
    () => {
      const { x, y } = data.train
      const r = resample(method as Method, stream(`resample/${seed}`), x, y as Tensor, k)
      const fitOn = (xs: Tensor, ys: Tensor) => logisticRegression({ l2 }).fit(dataset(xs, ys))
      const base = fitOn(x, y as Tensor)
      const model = r ? fitOn(r.x, r.y) : base
      const yTest = Array.from(toFlat(data.test.y as Tensor))
      const score = (m: typeof base) => {
        const p = toFlat(m.expect(data.test.x))
        const decided = Array.from(p, (v) => (v > 0.5 ? 1 : 0))
        return {
          recall: recall(yTest, decided, { positive: 1, zeroDivision: 0 }),
          precision: precision(yTest, decided, { positive: 1, zeroDivision: 0 }),
          balanced: balancedAccuracy(yTest, decided),
          auroc: auroc(yTest, Array.from(p)),
          ap: averagePrecision(yTest, Array.from(p)),
        }
      }
      const field = (m: typeof base) => {
        const p = toFlat(m.expect(GRID))
        return Array.from({ length: G }, (_, i): number[] => Array.from(p.slice(i * G, (i + 1) * G)) as number[])
      }
      return { r, before: score(base), after: score(model), field: field(model), baseField: field(base) }
    },
    [data, method, k, l2, seed],
    { mode: 'release' },
  )
  const { r, before, after, field, baseField } = result.value
  const X = useMemo(() => Float64Array.from(toFlat(data.train.x)), [data])
  const Y = useMemo(() => Array.from(toFlat(data.train.y as Tensor)), [data])
  const rowsOf = (c: number) => Y.flatMap((v, i) => (v === c ? [i] : []))
  const kept = useMemo(() => (r ? new Set(Array.from(r.origin).filter((i) => i >= 0)) : null), [r])
  const majorityRows = rowsOf(0)
  const minorityRows = rowsOf(1)
  const removed = kept ? Y.flatMap((_, i) => (kept.has(i) ? [] : [i])) : []
  const synthetic = useMemo(() => {
    if (!r) return null
    const RX = toFlat(r.x)
    const n0 = r.origin.length - r.synthetic.base.length
    const pts = { x: [] as number[], y: [] as number[] }
    const segs: { from: readonly [number, number]; to: readonly [number, number] }[] = []
    for (let s = 0; s < r.synthetic.base.length; s++) {
      pts.x.push(RX[2 * (n0 + s)])
      pts.y.push(RX[2 * (n0 + s) + 1])
      if (s < 150) {
        const b = r.synthetic.base[s]
        const nb = r.synthetic.neighbour[s]
        segs.push({ from: [X[2 * b], X[2 * b + 1]], to: [X[2 * nb], X[2 * nb + 1]] })
      }
    }
    return { pts, segs }
  }, [r, X])
  const border = useMemo(() => {
    if (!state.show.border) return null
    const b = borderStatus(data.train.x, data.train.y as Tensor, 1, 10)
    const of = (s: string) => Array.from(b.rows).filter((_, i) => b.status[i] === s)
    return { danger: of('danger'), noise: of('noise') }
  }, [data, state.show.border])
  const x1 = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const x2 = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: x1 })
  const counts = r ? [0, 1].map((c) => Array.from(toFlat(r.y)).filter((v) => v === c).length) : [majority, minority]
  const minorityShown = pick(X, kept ? minorityRows.filter((i) => kept.has(i)) : minorityRows)
  const majorityShown = pick(X, kept ? majorityRows.filter((i) => kept.has(i)) : majorityRows)
  const metric = (label: string, key: keyof typeof before) => (
    <Readout label={label} value={`${fmt(before[key])} → ${fmt(after[key])}`} />
  )
  return (
    <Figure
      title="Resampling a rare class"
      purpose="Over-sampling and SMOTE add minority points and under-sampling removes majority ones, which moves a classifier's boundary towards the majority class: recall of the rare class rises and precision falls, while the ranking measures (AUROC, average precision) barely move, because resampling mostly shifts the threshold."
      state={state}
      defaultSize="L"
      readouts={{
        'training counts (majority / minority)': (
          <>
            <Readout label="before" value={`${majority} / ${minority}`} />
            <Readout label="after" value={`${counts[0]} / ${counts[1]}`} />
            <Readout label="synthetic" value={r ? r.synthetic.base.length : 0} />
            <Readout label="removed" value={removed.length} />
          </>
        ),
        'held-out, original → resampled fit': (
          <>
            {metric('minority recall', 'recall')}
            {metric('precision', 'precision')}
            {metric('balanced accuracy', 'balanced')}
            {metric('AUROC', 'auroc')}
            {metric('average precision', 'ap')}
          </>
        ),
      }}
      caption="aifn gaussians draws two Gaussian classes d′ apart; the resampler of the method row is applied to the training set and logisticRegression is fitted to it. Left: the training data after resampling, majority (slot 0) and minority (slot 1) points, synthetic points (their own marker) with segments to their two parents (the first 150), removed points in grey, and optionally the borderline status of minority points (danger and noise, ink markers). Right: P(minority | x) of the resampled fit with its ½ contour (ink) and the original fit's ½ contour (coloured line). Scores are on a held-out sample twenty times larger with the same imbalance, at threshold ½."
    >
      <Plots cols={2}>
        <Plot x={x1} y={x2} title="training data after resampling">
          <Points name="majority" x={majorityShown.x} y={majorityShown.y} slot={0} thin />
          <Points name="minority" x={minorityShown.x} y={minorityShown.y} slot={1} size={6} />
          {removed.length > 0 && <Points name="removed" {...pick(X, removed)} muted shape={4} size={7} />}
          {synthetic && state.show.parents && <Segments segments={synthetic.segs} width={0.6} />}
          {synthetic && synthetic.pts.x.length > 0 && (
            <Points name="synthetic" x={synthetic.pts.x} y={synthetic.pts.y} slot={1} shape={3} size={5} />
          )}
          {border && <Points name="danger" {...pick(X, border.danger)} emphasis shape={1} size={9} />}
          {border && <Points name="noise" {...pick(X, border.noise)} emphasis shape={2} size={9} />}
        </Plot>
        <Plot x={x1} y={x2} title="P(minority | x): resampled fit and the original boundary">
          <Raster
            x={GRID_X}
            y={GRID_X}
            z={field}
            scale="sequential"
            range={[0, 1]}
            valueLabel="P(minority)"
            fillOpacity={0.75}
            boundary={0.5}
          />
          <Contours x={GRID_X} y={GRID_X} z={baseField} levels={[0.5]} labels={false} slot={2} />
          <Points name="minority" x={minorityShown.x} y={minorityShown.y} slot={1} size={4} />
        </Plot>
      </Plots>
    </Figure>
  )
}
