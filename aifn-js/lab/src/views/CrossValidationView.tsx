import { useMemo, useState } from 'react'
import { toRows } from 'aifn/foundation/tensor'
import type { CrossValidation } from 'aifn/learning/validate'
import { Select } from '@lab/controls'
import { PanelSlot } from '@lab/layout'
import { Bars, Curve, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue } from './format'
import { registerKind, registerView } from './registry'

export type CrossValidationPanelProps = {
  /** The result of `crossValidate` (or the outer level of `nested`). */
  result: CrossValidation<unknown>
  /** The metric drawn first (default: the first). */
  metric?: string
}

const ROLES = ['train', 'test'] as const

/**
 * A cross-validation at a glance: the fold assignment matrix (one row per fold, one column per data row; train, test
 * or unused) above the chosen metric on each fold's test rows, with its mean and ± one standard deviation.
 */
export function CrossValidationPanel({ result, metric }: CrossValidationPanelProps) {
  const names = Object.keys(result.scores)
  const [chosen, setChosen] = useState(metric ?? names[0])
  const shown = names.includes(chosen) ? chosen : names[0]
  const { x, y, z } = useMemo(() => {
    const rows = toRows(result.assignment)
    const n = rows[0]?.length ?? 0
    return {
      x: Array.from({ length: n }, (_, i) => i),
      y: rows.map((_, f) => f + 1),
      z: rows,
    }
  }, [result])
  const bars = useMemo(() => {
    const scores = Array.from(result.scores[shown]?.data ?? [])
    return {
      folds: scores.map((_, f) => f),
      scores,
      labels: scores.map((_, f) => String(f + 1)),
      ends: [-0.5, scores.length - 0.5],
      mean: result.mean[shown],
      sd: result.std[shown],
    }
  }, [result, shown])
  const rowAxis = useAxis({ label: 'row' })
  const foldRows = useAxis({ label: 'fold', format: (v) => v.toFixed(0) })
  const foldAxis = useAxis({ label: 'fold', categories: bars.labels })
  const scoreAxis = useAxis({ label: shown })
  const { mean, sd, ends } = bars
  return (
    <>
      {names.length > 1 && (
        <PanelSlot slot="controls">
          <Select label="metric" value={shown} onChange={setChosen} options={names} />
        </PanelSlot>
      )}
      <PanelSlot slot="readouts">
        <>
          <Readout label="folds" value={result.folds.length} />
          <Readout label={`mean ${shown}`} value={formatValue(result.mean[shown])} />
          <Readout label="sd over folds" value={formatValue(result.std[shown])} />
          <Readout label="direction" value={`${result.directions[shown]} is better`} />
        </>
      </PanelSlot>
      <Plots rows={2} heights={[55, 45]}>
        <Plot x={rowAxis} y={foldRows}>
          <Raster x={x} y={y} z={z} scale="categorical" categoryNames={ROLES} valueLabel="role" />
        </Plot>
        <Plot x={foldAxis} y={scoreAxis}>
          <Bars name={shown} x={bars.folds} y={bars.scores} slot={1} />
          <Curve name="mean" x={ends} y={[mean, mean]} emphasis />
          {Number.isFinite(sd) && <Curve name="± 1 sd" x={ends} y={[mean - sd, mean - sd]} muted dashed />}
          {Number.isFinite(sd) && <Curve name="± 1 sd" x={ends} y={[mean + sd, mean + sd]} muted dashed />}
        </Plot>
      </Plots>
    </>
  )
}

registerKind(
  'cross-validation',
  (o) =>
    typeof o === 'object' &&
    o !== null &&
    'assignment' in o &&
    'scores' in o &&
    'folds' in o &&
    typeof (o as CrossValidation<unknown>).splitter === 'string',
)
registerView<CrossValidation<unknown>>({
  key: 'cross-validation/folds',
  kind: 'cross-validation',
  description: 'The fold assignment matrix above a metric on each fold, with its mean and ± one standard deviation.',
  title: (r) => `Cross-validation: ${r.splitter}`,
  render: (r) => <CrossValidationPanel result={r} />,
})
