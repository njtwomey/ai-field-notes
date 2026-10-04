import type { Dataset } from 'aifn-methods/data'
import { useMemo, useState } from 'react'
import { andrewsCurves } from 'aifn-methods/unsupervised/embedding/linear'
import { zScores } from 'aifn/probability/stats'
import { fromData, mean, tensor, toFlat } from 'aifn/foundation/tensor'
import { Switch } from '@lab/controls'
import { ControlRow, PanelSlot } from '@lab/layout'
import { Curve, Plot, Readout, useAxis } from '@lab/viz'
import { toggled, useClassTable, type ClassTableInput } from './class-table'
import { ClassLegend } from './ClassLegend'
import { registerView } from './registry'

export type AndrewsCurvesPanelProps = ClassTableInput & {
  /** Standardise each feature before the projection (default true). */
  standardise?: boolean
  /** Rows drawn; above this a class-stratified sample is drawn (default 1000). */
  maxRows?: number
}

const POINTS = 121
const T_RANGE = [-Math.PI, Math.PI] as const

/**
 * Andrews curves (aifn/embed `andrewsCurves`): each row becomes the finite Fourier series
 * f(t) = x₁/√2 + x₂ sin t + x₃ cos t + x₄ sin 2t + …, drawn thin in its class's colour, with each class's mean curve
 * bold. Distances between curves (in L²) are proportional to distances between rows, so classes that separate in the
 * features separate as bands of curves. The first features set the lowest frequencies, so the order of features
 * changes the picture but not the distances.
 */
export function AndrewsCurvesPanel({
  data,
  x,
  y,
  featureNames,
  labelNames,
  standardise: initialStandardise = true,
  maxRows = 1000,
}: AndrewsCurvesPanelProps) {
  const table = useClassTable({ data, x, y, featureNames, labelNames }, maxRows)
  const { n, d, k, labels, columns } = table
  const [scaled, setScaled] = useState(initialStandardise)
  const [hidden, setHidden] = useState<ReadonlySet<number>>(() => new Set())

  const { t, rows, means } = useMemo(() => {
    const cols = scaled ? columns.map((c) => toFlat(zScores(c))) : columns
    const flat = new Float64Array(n * d)
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) flat[i * d + j] = cols[j][i]
    const at = Array.from({ length: POINTS }, (_, j) => -Math.PI + (2 * Math.PI * j) / (POINTS - 1))
    const { t, curves } = andrewsCurves(fromData(flat, [n, d]), at)
    // The curve of the class mean is the mean of the class's curves: the map is linear.
    const centres = Array.from({ length: k }, (_, c) => {
      const mine = cols.map((v) => mean(tensor(Array.from(v).filter((_, i) => labels[i] === c))))
      return toFlat(andrewsCurves(fromData(Float64Array.from(mine), [1, d]), at).curves)
    })
    return { t: Array.from(t), rows: toFlat(curves), means: centres }
  }, [columns, scaled, n, d, k, labels])

  const series = useMemo(() => {
    type Line = { name: string; x: number[]; y: number[]; slot: number; thin: boolean }
    const out: Line[] = []
    const m = t.length
    for (let c = 0; c < k; c++) {
      if (hidden.has(c)) continue
      // Every row of the class as one line series, rows separated by a NaN gap.
      const xs: number[] = []
      const ys: number[] = []
      for (let i = 0; i < n; i++) {
        if (labels[i] !== c) continue
        for (let j = 0; j < m; j++) {
          xs.push(t[j])
          ys.push(rows[i * m + j])
        }
        xs.push(NaN)
        ys.push(NaN)
      }
      out.push({ name: table.labelNames[c], x: xs, y: ys, thin: true, slot: c })
    }
    for (let c = 0; c < k; c++)
      if (!hidden.has(c)) out.push({ name: `${table.labelNames[c]} mean`, x: t, y: means[c], slot: c, thin: false })
    return out
  }, [t, rows, means, n, k, labels, hidden, table.labelNames])

  const tAxis = useAxis({ label: 't', range: T_RANGE })
  const fAxis = useAxis({ label: 'f(t)' })

  const counts = useMemo(() => {
    const out = new Array<number>(k).fill(0)
    for (const c of labels) out[c]++
    return out
  }, [labels, k])

  return (
    <>
      <PanelSlot slot="controls">
        <>
          <ControlRow label="features">
            <Switch label="standardise each feature first" checked={scaled} onChange={setScaled} />
          </ControlRow>
          {table.labelled && (
            <ControlRow label="classes (click to hide or show)">
              <ClassLegend
                names={table.labelNames}
                counts={counts}
                hidden={hidden}
                onToggle={(c) => setHidden((h) => toggled(h, c))}
              />
            </ControlRow>
          )}
        </>
      </PanelSlot>
      <PanelSlot slot="readouts">
        <>
          <Readout
            label="rows"
            value={table.total > n ? `${n} drawn of ${table.total} (stratified sample)` : String(n)}
          />
        </>
      </PanelSlot>
      <Plot x={tAxis} y={fAxis} legend={false} renderer="canvas">
        {series.map((s) => (
          <Curve key={s.name} id={s.name} name={s.name} x={s.x} y={s.y} slot={s.slot} thin={s.thin} silent={s.thin} />
        ))}
      </Plot>
    </>
  )
}

registerView<Dataset>({
  key: 'dataset/andrews',
  kind: 'dataset',
  description: 'Andrews curves: each row as a Fourier series, thin per row and bold per class mean.',
  title: (d) => `Andrews curves: ${d.meta.name ?? 'data'}`,
  render: (d) => <AndrewsCurvesPanel data={d} />,
})
