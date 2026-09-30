import { useCallback, useMemo, useState } from 'react'
import { correlation, extent, histogram, kde } from 'aifn/stats'
import { seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { Button, MultiCombobox, Select, Switch } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import {
  formatNumber,
  Panel,
  Readout,
  Subplots,
  useScaleColor,
  XYChart,
  type PlotPointer,
  type Range,
  type Segment,
  type XYSeries,
} from '@lab/viz'
import { toggled, useClassTable, type ClassTable, type ClassTableInput } from './class-table'
import { ClassLegend } from './ClassLegend'
import type { FrameProps } from './frame'

export type PairPlotViewProps = FrameProps &
  ClassTableInput & {
    /** The features drawn first (column indices); default the first `maxFeatures`. */
    features?: readonly number[]
    /** The most features drawn at once (default 8); with more, a picker chooses them. */
    maxFeatures?: number
    /** Rows drawn; above this a class-stratified sample is drawn and the readouts say so (default 2000). */
    maxRows?: number
    /** The diagonal at first: per-class histograms on shared bins, or per-class KDE curves (default histogram). */
    diagonal?: 'histogram' | 'kde'
    /** The upper triangle at first: the mirrored scatter, or Pearson's r overall and per class (default correlation). */
    upper?: 'scatter' | 'correlation'
  }

type Selection = { x: number; y: number; rx: Range; ry: Range }

/** A hovered point is found within this fraction of each axis's span. */
const HOVER_RADIUS = 0.04
const KDE_POINTS = 80

/**
 * A scatter matrix of a labelled dataset: per-class histograms (or KDE curves) of each feature down the diagonal, the
 * pairwise scatters below it, and above it either the mirrored scatters or the correlations. Columns share x and rows
 * share y, so a feature has one scale down its column and across its row. Drag a rectangle in any scatter to select the
 * rows inside it: they keep their class colour in every panel and the rest fade. Hovering a point marks the same row
 * everywhere; the class chips hide or show a class in every panel.
 */
export function PairPlotView({
  data,
  x,
  y,
  featureNames,
  labelNames,
  features: initialFeatures,
  maxFeatures = 8,
  maxRows = 2000,
  diagonal = 'histogram',
  upper = 'correlation',
  title,
  description,
  controls,
  readouts,
  caption,
  id,
  defaultSize = 'L',
}: PairPlotViewProps) {
  const table = useClassTable({ data, x, y, featureNames, labelNames }, maxRows)
  const { resolved: mode } = useTheme()
  const [chosen, setChosen] = useState<number[]>(() =>
    (initialFeatures ?? Array.from({ length: table.d }, (_, j) => j)).slice(0, maxFeatures),
  )
  const features = useMemo(() => chosen.filter((j) => j < table.d), [chosen, table.d])
  const [diag, setDiag] = useState(diagonal)
  const [top, setTop] = useState(upper)
  const [hidden, setHidden] = useState<ReadonlySet<number>>(() => new Set())
  const [selection, setSelection] = useState<Selection | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  const { columns, labels, n, k } = table
  // Rows shown (class not hidden) and selected (inside the brushed rectangle); selected is null with no brush.
  const shown = useMemo(() => Uint8Array.from(labels, (c) => (hidden.has(c) ? 0 : 1)), [labels, hidden])
  const selected = useMemo(() => {
    if (!selection) return null
    const [a, b] = [columns[selection.x], columns[selection.y]]
    return Uint8Array.from(shown, (s, i) =>
      s && a[i] >= selection.rx[0] && a[i] <= selection.rx[1] && b[i] >= selection.ry[0] && b[i] <= selection.ry[1]
        ? 1
        : 0,
    )
  }, [selection, columns, shown])
  const active = selected ?? shown
  const counts = useMemo(() => {
    const out = new Array<number>(k).fill(0)
    labels.forEach((c, i) => active[i] && out[c]++)
    return out
  }, [labels, active, k])

  // Shared bins per feature, from every drawn row, so hiding a class or brushing never moves them.
  const edges = useMemo(() => columns.map((c) => histogram(c, { bins: 'sturges' }).edges), [columns])
  const spans = useMemo(() => columns.map((c) => extent(c)), [columns])

  const diagonalSeries = useMemo(
    () =>
      features.map((f) => {
        const e = edges[f]
        const centres = Array.from({ length: e.length - 1 }, (_, b) => (e[b] + e[b + 1]) / 2)
        const width = e[1] - e[0]
        const values = (keep: (i: number) => boolean) => columns[f].filter((_, i) => keep(i))
        const out: XYSeries[] = []
        // Under a brush, every shown row as a muted histogram, so the selection reads as a part of the whole.
        if (selected)
          out.push({
            name: 'all shown',
            type: 'bar',
            x: centres,
            y: Array.from(
              histogram(
                values((i) => !!shown[i]),
                { bins: e },
              ).counts,
            ),
            histogram: true,
            muted: true,
          })
        for (let c = 0; c < k; c++) {
          if (hidden.has(c)) continue
          const v = values((i) => labels[i] === c && !!active[i])
          const name = table.labelNames[c]
          if (diag === 'histogram')
            out.push({
              name,
              type: 'bar',
              x: centres,
              y: Array.from(histogram(v, { bins: e }).counts),
              histogram: true,
              slot: c,
            })
          else if (v.length > 1) {
            // Scaled to counts per bin (density × rows × bin width), so the curves read on the histogram's axis.
            const [lo, hi] = [e[0] - width, e[e.length - 1] + width]
            const at = Array.from({ length: KDE_POINTS }, (_, j) => lo + ((hi - lo) * j) / (KDE_POINTS - 1))
            const { density, degenerate } = kde(v, at)
            if (!degenerate)
              out.push({
                name,
                type: 'line',
                x: at,
                y: Array.from(density, (p) => p * v.length * width),
                area: true,
                slot: c,
              })
          }
        }
        return out
      }),
    [features, edges, columns, selected, shown, hidden, labels, active, diag, k, table.labelNames],
  )

  const scatterSeries = useCallback(
    (fx: number, fy: number): XYSeries[] => {
      const pick = (keep: (i: number) => boolean) => {
        const idx: number[] = []
        for (let i = 0; i < n; i++) if (keep(i)) idx.push(i)
        return {
          x: idx.map((i) => columns[fx][i]),
          y: idx.map((i) => columns[fy][i]),
          group: idx.map((i) => labels[i]),
        }
      }
      const out: XYSeries[] = []
      if (selected) {
        const rest = pick((i) => !!shown[i] && !selected[i])
        out.push({ name: 'not selected', type: 'scatter', x: rest.x, y: rest.y, muted: true, thin: true })
      }
      const on = pick((i) => !!active[i])
      out.push({ name: 'rows', type: 'scatter', ...on, groupNames: table.labelNames, thin: true })
      return out
    },
    [n, columns, labels, selected, shown, active, table.labelNames],
  )

  const cells = useMemo(
    () =>
      features.flatMap((fy, r) =>
        features.map((fx, c) =>
          r === c ? null : r < c && top === 'correlation' ? null : { fx, fy, series: scatterSeries(fx, fy) },
        ),
      ),
    [features, top, scatterSeries],
  )

  const onPointer = useCallback(
    (fx: number, fy: number) => (e: PlotPointer) => {
      if (e.type === 'leave') return setHovered(null)
      if (e.type !== 'move') return
      const [sx, sy] = [spans[fx], spans[fy]]
      const [wx, wy] = [sx[1] - sx[0] || 1, sy[1] - sy[0] || 1]
      let best = -1
      let bestD = HOVER_RADIUS ** 2
      for (let i = 0; i < n; i++) {
        if (!shown[i]) continue
        const d = ((columns[fx][i] - e.point[0]) / wx) ** 2 + ((columns[fy][i] - e.point[1]) / wy) ** 2
        if (d < bestD) [best, bestD] = [i, d]
      }
      setHovered((h) => (h === (best < 0 ? null : best) ? h : best < 0 ? null : best))
    },
    [spans, n, shown, columns],
  )
  const onBrush = useCallback(
    (fx: number, fy: number) => (rect: { x: Range; y: Range } | null) =>
      setSelection(rect ? { x: fx, y: fy, rx: rect.x, ry: rect.y } : null),
    [],
  )

  const hoverLive = (fx: number, fy: number): XYSeries[] => [
    {
      name: 'hovered',
      type: 'scatter',
      x: hovered === null ? [] : [columns[fx][hovered]],
      y: hovered === null ? [] : [columns[fy][hovered]],
      emphasis: true,
      thin: true,
    },
  ]
  // The committed brush, drawn as a rectangle on the panel it was drawn in (memoised: a new array redraws the panel).
  const brushSegments = useMemo((): readonly Segment[] => {
    if (!selection) return NO_SEGMENTS
    const [[x0, x1], [y0, y1]] = [selection.rx, selection.ry]
    return [
      { from: [x0, y0], to: [x1, y0] },
      { from: [x1, y0], to: [x1, y1] },
      { from: [x1, y1], to: [x0, y1] },
      { from: [x0, y1], to: [x0, y0] },
    ]
  }, [selection])
  const brushBox = (fx: number, fy: number) =>
    selection && selection.x === fx && selection.y === fy ? brushSegments : NO_SEGMENTS

  const m = features.length
  const renderer = n * m * m > 40000 ? 'canvas' : 'svg'
  const axisKey = `${features.join(',')}|${diag}|${top}`
  const featureOptions = useMemo(
    () => table.featureNames.map((name, j) => ({ value: String(j), label: name })),
    [table.featureNames],
  )
  const selectedCount = selected ? selected.reduce((a, b) => a + b, 0) : null

  return (
    <Figure
      id={id}
      title={title ?? `Pair plot: ${data?.meta.name ?? 'data'}`}
      description={description}
      defaultSize={defaultSize}
      controls={
        <>
          {controls}
          {table.d > maxFeatures && (
            <MultiCombobox
              label={`features (up to ${maxFeatures})`}
              value={features.map(String)}
              onChange={(v) => {
                setChosen(v.map(Number))
                setSelection(null)
              }}
              options={featureOptions}
              max={maxFeatures}
              min={2}
            />
          )}
          <ControlRow label="panels">
            <Switch
              label="KDE curves on the diagonal"
              checked={diag === 'kde'}
              onChange={(on) => setDiag(on ? 'kde' : 'histogram')}
            />
            <Select
              label="upper triangle"
              value={top}
              onChange={setTop}
              options={[
                { value: 'correlation', label: 'correlation (r)' },
                { value: 'scatter', label: 'scatter (mirrored)' },
              ]}
            />
            <Button variant="outline" size="sm" disabled={!selection} onClick={() => setSelection(null)}>
              Clear selection
            </Button>
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
      }
      readouts={
        <>
          {readouts}
          <Readout
            label="rows"
            value={table.total > n ? `${n} drawn of ${table.total} (stratified sample)` : String(n)}
          />
          {selectedCount !== null && <Readout label="selected" value={`${selectedCount} of ${n}`} />}
          {hovered !== null && (
            <Readout
              label={`row ${hovered}`}
              color={seriesColor(mode, labels[hovered])}
              value={`${table.labelled ? `${table.labelNames[labels[hovered]]} · ` : ''}${features
                .map((f) => `${table.featureNames[f]} ${formatNumber(columns[f][hovered])}`)
                .join(', ')}`}
            />
          )}
        </>
      }
      caption={
        caption ??
        'Drag a rectangle in any scatter to select the rows inside it; they keep their colour in every panel and the rest fade. A click without a drag clears it. Hover a point to mark the same row everywhere.'
      }
      hoverReadout={false}
    >
      <Subplots
        rows={m}
        cols={m}
        sharex="col"
        sharey="row"
        dense
        toolbar={false}
        rescaleOnChange={false}
        axisKey={axisKey}
      >
        {features.flatMap((fy, r) =>
          features.map((fx, c) => {
            const key = `${fy}-${fx}`
            if (r === c)
              return (
                <Panel key={key} share={{ y: false }} ticks={{ y: false }}>
                  <XYChart
                    series={diagonalSeries[r]}
                    live={diagonalLive(hovered === null ? null : columns[fx][hovered], diagonalSeries[r])}
                    xLabel={table.featureNames[fx]}
                    yLabel={table.featureNames[fy]}
                    zoom={false}
                    legend={false}
                  />
                </Panel>
              )
            const cell = cells[r * m + c]
            if (!cell)
              return (
                <Panel key={key}>
                  <CorrelationCell table={table} fx={fx} fy={fy} active={active} hidden={hidden} />
                </Panel>
              )
            return (
              <Panel key={key}>
                <XYChart
                  series={cell.series}
                  live={hoverLive(fx, fy)}
                  segments={brushBox(fx, fy)}
                  xLabel={table.featureNames[fx]}
                  yLabel={table.featureNames[fy]}
                  zoom={false}
                  legend={false}
                  renderer={renderer}
                  onBrush={onBrush(fx, fy)}
                  onPointer={onPointer(fx, fy)}
                />
              </Panel>
            )
          }),
        )}
      </Subplots>
    </Figure>
  )
}

const NO_SEGMENTS: readonly Segment[] = []

/** A dashed ink line at the hovered row's value on a diagonal panel, as tall as its tallest bar or curve. */
function diagonalLive(at: number | null, series: readonly XYSeries[]): XYSeries[] {
  let peak = 0
  for (const s of series) for (const v of s.y) peak = Math.max(peak, v)
  return [
    {
      name: 'hovered',
      type: 'line',
      x: at === null ? [] : [at, at],
      y: at === null ? [] : [0, peak],
      emphasis: true,
      dashed: true,
    },
  ]
}

/** Pearson's r of two features over the shown (or selected) rows: overall, then per class in its colour. */
function CorrelationCell({
  table,
  fx,
  fy,
  active,
  hidden,
}: {
  table: ClassTable
  fx: number
  fy: number
  active: Uint8Array
  hidden: ReadonlySet<number>
}) {
  const { resolved: mode } = useTheme()
  const tint = useScaleColor('diverging')
  const { overall, perClass } = useMemo(() => {
    const r = (keep: (i: number) => boolean) => {
      const a: number[] = []
      const b: number[] = []
      for (let i = 0; i < table.n; i++)
        if (keep(i)) {
          a.push(table.columns[fx][i])
          b.push(table.columns[fy][i])
        }
      return a.length > 2 ? correlation(a, b) : NaN
    }
    return {
      overall: r((i) => !!active[i]),
      perClass: table.labelled
        ? table.labelNames.map((_, c) => (hidden.has(c) ? null : r((i) => !!active[i] && table.labels[i] === c)))
        : [],
    }
  }, [table, fx, fy, active, hidden])
  const show = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '–')
  const background = Number.isFinite(overall)
    ? `color-mix(in srgb, ${tint((overall + 1) / 2)} ${Math.round(25 * Math.abs(overall))}%, transparent)`
    : undefined
  return (
    <div
      className="@container flex h-full flex-col items-center justify-center gap-0.5 overflow-hidden rounded-sm border border-border/60 p-1 text-center"
      style={{ background, margin: 5, height: 'calc(100% - 10px)' }}
      title={`Pearson's r between ${table.featureNames[fx]} and ${table.featureNames[fy]}`}
    >
      <div className="font-mono text-xs font-medium tabular-nums @[7rem]:text-base">r = {show(overall)}</div>
      {perClass.map((v, c) =>
        v === null ? null : (
          <div
            key={c}
            className="max-w-full truncate font-mono text-[10px] tabular-nums @[7rem]:text-xs"
            style={{ color: seriesColor(mode, c) }}
          >
            <span className="hidden @[9rem]:inline">{table.labelNames[c]} </span>
            {show(v)}
          </div>
        ),
      )}
    </div>
  )
}
