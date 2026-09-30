import { useMemo, useState } from 'react'
import { toRows } from 'aifn/tensor'
import type { CrossValidation } from 'aifn/validate'
import { Select } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Heatmap, Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from './format'
import type { FrameProps } from './frame'

export type CrossValidationViewProps = FrameProps & {
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
export function CrossValidationView({ result, metric, title, controls, readouts, ...frame }: CrossValidationViewProps) {
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
  const bars = useMemo((): XYSeries[] => {
    const scores = Array.from(result.scores[shown]?.data ?? [])
    const folds = scores.map((_, f) => f + 1)
    const mean = result.mean[shown]
    const sd = result.std[shown]
    const ends = [0.5, scores.length + 0.5]
    const out: XYSeries[] = [
      { name: shown, type: 'bar', x: folds, y: scores, slot: 1 },
      { name: 'mean', type: 'line', x: ends, y: [mean, mean], emphasis: true },
    ]
    if (Number.isFinite(sd)) {
      out.push({ name: '± 1 sd', type: 'line', x: ends, y: [mean - sd, mean - sd], muted: true, dashed: true })
      out.push({ name: '± 1 sd', type: 'line', x: ends, y: [mean + sd, mean + sd], muted: true, dashed: true })
    }
    return out
  }, [result, shown])
  return (
    <Figure
      title={title ?? `Cross-validation: ${result.splitter}`}
      defaultSize="L"
      {...frame}
      controls={
        <>
          {controls}
          {names.length > 1 && <Select label="metric" value={shown} onChange={setChosen} options={names} />}
        </>
      }
      readouts={
        <>
          {readouts}
          <Readout label="folds" value={result.folds.length} />
          <Readout label={`mean ${shown}`} value={formatValue(result.mean[shown])} />
          <Readout label="sd over folds" value={formatValue(result.std[shown])} />
          <Readout label="direction" value={`${result.directions[shown]} is better`} />
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <ChartSize scale={0.55}>
          <Heatmap x={x} y={y} z={z} scale="categorical" categoryNames={ROLES} xLabel="row" yLabel="fold" />
        </ChartSize>
        <ChartSize scale={0.45}>
          <XYChart series={bars} xLabel="fold" yLabel={shown} integerX />
        </ChartSize>
      </div>
    </Figure>
  )
}
