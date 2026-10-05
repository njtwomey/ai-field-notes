import { useMemo, useState } from 'react'
import { gaussians } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import {
  adasyn,
  borderlineSmote,
  randomOverSample,
  randomUnderSample,
  removeTomekLinks,
  smote,
  type Resampled,
} from 'aifn-methods/learning/preprocessing'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { auroc, averagePrecision, balancedAccuracy, precision, recall } from 'aifn-compute/learning/metrics'
import {
  Contours,
  ControlRow,
  Figure,
  formatNumber,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Segments,
  Select,
  Slider,
  useAxis,
} from 'aifn-render'

const METHODS = [
  { value: 'none', label: 'None (original uncorrected data)' },
  { value: 'smote', label: 'SMOTE' },
  { value: 'borderline', label: 'Borderline-SMOTE' },
  { value: 'adasyn', label: 'ADASYN' },
  { value: 'over', label: 'Random over-sampling' },
  { value: 'under', label: 'Random under-sampling' },
  { value: 'tomek', label: 'Tomek-link cleaning' },
] as const

type Method = (typeof METHODS)[number]['value']

const BOX = 4.5
const G = 50
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

const pick = (x: Float64Array, rows: readonly number[]) => ({
  x: rows.map((i) => x[2 * i]),
  y: rows.map((i) => x[2 * i + 1]),
})

export function DoesSmoteHelpExplorer() {
  const [method, setMethod] = useState<Method>('smote')
  const [minorityCount, setMinorityCount] = useState(25)
  const [separation, setSeparation] = useState(2.0)
  const majorityCount = 400

  const data = useMemo(() => {
    const s = stream('does-smote-help-seed-1')
    const train = gaussians(s, { n: [majorityCount, minorityCount], separation })
    const test = gaussians(child(s, 'test'), {
      n: [20 * majorityCount, 20 * minorityCount],
      separation,
    })
    return { train, test }
  }, [minorityCount, separation])

  const { r, before, after, field, baseField } = useMemo(() => {
    const { x, y } = data.train
    const sResample = stream('resample-stream')
    const res = resample(method, sResample, x, y as Tensor, 5)

    const fitOn = (xs: Tensor, ys: Tensor) => logisticRegression({ l2: 1 }).fit(dataset(xs, ys))
    const base = fitOn(x, y as Tensor)
    const model = res ? fitOn(res.x, res.y) : base

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

    const computeField = (m: typeof base) => {
      const p = toFlat(m.expect(GRID))
      return Array.from({ length: G }, (_, i): number[] => Array.from(p.slice(i * G, (i + 1) * G)) as number[])
    }

    return {
      r: res,
      before: score(base),
      after: score(model),
      field: computeField(model),
      baseField: computeField(base),
    }
  }, [data, method])

  const X = useMemo(() => Float64Array.from(toFlat(data.train.x)), [data])
  const Y = useMemo(() => Array.from(toFlat(data.train.y as Tensor)), [data])
  const rowsOf = (c: number) => Y.flatMap((v, i) => (v === c ? [i] : []))
  const kept = useMemo(() => (r ? new Set(Array.from(r.origin).filter((i) => i >= 0)) : null), [r])
  const majorityRows = rowsOf(0)
  const minorityRows = rowsOf(1)
  const majorityShown = pick(X, kept ? majorityRows.filter((i) => kept.has(i)) : majorityRows)
  const minorityShown = pick(X, kept ? minorityRows.filter((i) => kept.has(i)) : minorityRows)

  const synthetic = useMemo(() => {
    if (!r) return null
    const RX = toFlat(r.x)
    const n0 = r.origin.length - r.synthetic.base.length
    const pts = { x: [] as number[], y: [] as number[] }
    const segs: { from: readonly [number, number]; to: readonly [number, number] }[] = []
    for (let s = 0; s < r.synthetic.base.length; s++) {
      pts.x.push(RX[2 * (n0 + s)])
      pts.y.push(RX[2 * (n0 + s) + 1])
      if (s < 120) {
        const b = r.synthetic.base[s]
        const nb = r.synthetic.neighbour[s]
        segs.push({ from: [X[2 * b], X[2 * b + 1]], to: [X[2 * nb], X[2 * nb + 1]] })
      }
    }
    return { pts, segs }
  }, [r, X])

  const x1 = useAxis({ label: 'feature x₁', range: [-BOX, BOX] })
  const x2 = useAxis({ label: 'feature x₂', range: [-BOX, BOX], equal: x1 })

  return (
    <Figure
      title="Empirical test: Does SMOTE or rebalancing help?"
      purpose="Evaluate whether synthetic resampling or class rebalancing improves classification discrimination or merely shifts the operating decision threshold."
      caption="Left: training dataset after resampling (majority class in blue, minority class in green, synthetic points joined by grey segments to parent pairs). Right: P(minority | x) probability surface from the resampled logistic fit (fill and dashed boundary) compared to the uncorrected baseline decision boundary (solid cyan line). Observe the readouts: while minority recall increases, precision drops sharply, and the ranking metrics (AUROC and Average Precision) remain virtually identical, proving that rebalancing functions primarily as an uncalibrated threshold shift."
    >
      <ControlRow>
        <Select
          label="Resampling technique"
          value={method}
          onChange={(v) => setMethod(v as Method)}
          options={METHODS.map((m) => ({ value: m.value, label: m.label }))}
        />
        <Slider
          label="Minority sample size (400 majority)"
          value={minorityCount}
          min={10}
          max={80}
          step={5}
          onChange={setMinorityCount}
        />
        <Slider
          label="Class separation d′"
          value={separation}
          min={1.0}
          max={3.5}
          step={0.1}
          onChange={setSeparation}
        />
      </ControlRow>

      <div className="my-2 flex flex-wrap gap-4 font-mono text-xs text-muted-foreground">
        <Readout label="Minority Recall" value={`${formatNumber(before.recall)} → ${formatNumber(after.recall)}`} />
        <Readout label="Precision" value={`${formatNumber(before.precision)} → ${formatNumber(after.precision)}`} />
        <Readout label="AUROC (Ranking)" value={`${formatNumber(before.auroc)} → ${formatNumber(after.auroc)}`} />
        <Readout label="Avg Precision (PR)" value={`${formatNumber(before.ap)} → ${formatNumber(after.ap)}`} />
        <Readout label="Balanced Acc" value={`${formatNumber(before.balanced)} → ${formatNumber(after.balanced)}`} />
      </div>

      <Plots cols={2}>
        <Plot x={x1} y={x2} title="Resampled training data">
          <Points name="Majority class (0)" x={majorityShown.x} y={majorityShown.y} slot={0} />
          <Points name="Minority class (1)" x={minorityShown.x} y={minorityShown.y} slot={1} size={6} />
          {synthetic && <Segments segments={synthetic.segs} width={0.5} />}
          {synthetic && synthetic.pts.x.length > 0 && (
            <Points name="Synthetic samples" x={synthetic.pts.x} y={synthetic.pts.y} slot={2} size={5} />
          )}
        </Plot>

        <Plot x={x1} y={x2} title="Predicted P(minority | x) & decision boundary">
          <Raster
            x={GRID_X}
            y={GRID_X}
            z={field}
            scale="sequential"
            range={[0, 1]}
            valueLabel="P(minority)"
            fillOpacity={0.7}
          />
          <Contours x={GRID_X} y={GRID_X} z={baseField} levels={[0.5]} labels={false} slot={1} />
          <Contours x={GRID_X} y={GRID_X} z={field} levels={[0.5]} labels={false} slot={3} />
          <Points name="Minority observations" x={minorityShown.x} y={minorityShown.y} slot={1} size={4} />
        </Plot>
      </Plots>
    </Figure>
  )
}
