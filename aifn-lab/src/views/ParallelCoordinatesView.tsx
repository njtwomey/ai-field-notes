import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { ArrowUpDown, GripVertical } from 'lucide-react'
import { extent, median, zScores } from 'aifn/probability/stats'
import { toFlat } from 'aifn/foundation/tensor'
import { chrome, seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { Button, Select, Switch } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { cn } from '@lab/lib/utils'
import { EChart, formatNumber, Readout, useChartHeight, useElementSize, type Range } from '@lab/viz'
import { toggled, useClassTable, type ClassTableInput } from './class-table'
import { ClassLegend } from './ClassLegend'
import type { FrameProps } from './frame'

export type ParallelCoordinatesViewProps = FrameProps &
  ClassTableInput & {
    /** Axis scaling at first: each axis over its own range (`minmax`, default), or z-scores on one common range. */
    scaling?: 'minmax' | 'standard'
    /** Draw each class's per-axis medians as a bold line (default false). */
    medians?: boolean
    /** Rows drawn; above this a class-stratified sample is drawn and the readouts say so (default 3000). */
    maxRows?: number
  }

/** The header strip over the axes: a draggable chip per axis with a flip button. */
const HEADER = 30
/** Axes this far apart or more carry their names in the header; closer, the names are drawn rotated under the axes. */
const NAMED_STEP = 76
const SIDE = 36

type Areas = Readonly<Record<string, readonly (readonly [number, number])[]>>

const axisId = (j: number) => `f${j}`

/**
 * Parallel coordinates: one vertical axis per feature and one polyline per row, coloured by class. Each axis is scaled
 * to its own range (min–max) or all share one z-score range (standardised), so spreads compare across axes. Drag along
 * an axis to brush an interval: rows outside every brushed interval fade. Drag an axis's chip in the header to move the
 * axis, and use its arrow to flip it; the class chips hide or show a class. Lines are thin and translucent, lighter
 * the more rows there are; per-class median lines summarise each class.
 */
export function ParallelCoordinatesView({
  data,
  x,
  y,
  featureNames,
  labelNames,
  scaling: initialScaling = 'minmax',
  medians: initialMedians = false,
  maxRows = 3000,
  title,
  description,
  controls,
  readouts,
  caption,
  id,
  defaultSize = 'L',
}: ParallelCoordinatesViewProps) {
  const table = useClassTable({ data, x, y, featureNames, labelNames }, maxRows)
  const { n, d, k, labels, columns } = table
  const { resolved: mode } = useTheme()
  const [order, setOrder] = useState<number[]>(() => Array.from({ length: d }, (_, j) => j))
  const [flipped, setFlipped] = useState<ReadonlySet<number>>(() => new Set())
  const [scaling, setScaling] = useState(initialScaling)
  const [showMedians, setShowMedians] = useState(initialMedians)
  const [hidden, setHidden] = useState<ReadonlySet<number>>(() => new Set())
  const [areas, setAreas] = useState<Areas>({})

  // The plotted values: raw on per-axis ranges, or z-scores (aifn/stats) on one range shared by every axis.
  const scaled = useMemo(() => {
    const values = scaling === 'standard' ? columns.map((c) => Float64Array.from(toFlat(zScores(c)))) : columns
    const ranges = values.map((v) => extent(v) as Range)
    const common: Range = [Math.min(...ranges.map((r) => r[0])), Math.max(...ranges.map((r) => r[1]))]
    return { values, ranges: scaling === 'standard' ? ranges.map(() => common) : ranges }
  }, [columns, scaling])

  // Rows inside every brushed interval (an axis with none passes everything).
  const inside = useMemo(() => {
    const brushed = Object.entries(areas).filter(([, v]) => v.length)
    if (!brushed.length) return null
    const cols = brushed.map(([key, v]) => ({ v: scaled.values[Number(key.slice(1))], intervals: v }))
    return Uint8Array.from({ length: n }, (_, i) =>
      cols.every(({ v, intervals }) => intervals.some(([a, b]) => v[i] >= Math.min(a, b) && v[i] <= Math.max(a, b)))
        ? 1
        : 0,
    )
  }, [areas, scaled, n])
  const counts = useMemo(() => {
    const out = new Array<number>(k).fill(0)
    labels.forEach((c, i) => (!inside || inside[i]) && out[c]++)
    return out
  }, [labels, inside, k])
  const perClassMedians = useMemo(
    () =>
      Array.from({ length: k }, (_, c) =>
        scaled.values.map((v) => {
          const mine = v.filter((_, i) => labels[i] === c)
          return mine.length ? median(mine) : NaN
        }),
      ),
    [scaled, labels, k],
  )

  const [box, size] = useElementSize<HTMLDivElement>()
  const height = useChartHeight()
  const step = d > 1 ? (size.width - 2 * SIDE) / (d - 1) : 0
  const named = step >= NAMED_STEP || d === 1
  // Lighter the more rows there are, so the density of lines, not the last class drawn, sets the colour.
  const opacity = Math.min(0.5, Math.max(0.04, 25 / Math.max(n, 1) ** 0.75))

  const option = useMemo(() => {
    const c = chrome(mode)
    const rowsOf = (cls: number) => {
      const out: number[][] = []
      for (let i = 0; i < n; i++) if (labels[i] === cls) out.push(scaled.values.map((v) => v[i]))
      return out
    }
    const series: Record<string, unknown>[] = []
    for (let cls = 0; cls < k; cls++) {
      if (hidden.has(cls)) continue
      series.push({
        name: table.labelNames[cls],
        type: 'parallel',
        data: rowsOf(cls),
        smooth: false,
        lineStyle: { width: 1, color: seriesColor(mode, cls), opacity },
        // Brushed lines stand out against the faded rest even when few are left.
        activeOpacity: Math.min(0.8, opacity * 3),
        inactiveOpacity: 0.025,
        emphasis: { lineStyle: { width: 2, opacity: 1 } },
        progressive: 0,
        z: 2,
      })
    }
    if (showMedians)
      for (let cls = 0; cls < k; cls++) {
        if (hidden.has(cls)) continue
        const row = [perClassMedians[cls]]
        // A surface-coloured casing under each median line separates it from the thin lines of its own class.
        for (const [width, color, z] of [
          [5, c.surface, 3],
          [2.5, seriesColor(mode, cls), 4],
        ] as const)
          series.push({
            name: `${table.labelNames[cls]} median`,
            type: 'parallel',
            data: row,
            silent: true,
            lineStyle: { width, color, opacity: 1 },
            activeOpacity: 1,
            inactiveOpacity: 1,
            z,
          })
      }
    const labelled = step >= 56 || scaling === 'standard'
    return {
      legend: { show: false },
      tooltip: {
        trigger: 'item',
        formatter: (p: { seriesName: string; value: number[]; marker: string }) =>
          `${p.marker}${p.seriesName}<br/>` +
          order
            .map((j) => `${table.featureNames[j]}: ${formatNumber(p.value[j])}`)
            .slice(0, 12)
            .join('<br/>'),
      },
      parallel: {
        left: SIDE,
        right: SIDE,
        top: 12,
        bottom: named ? 26 : 64,
        parallelAxisDefault: {
          type: 'value',
          realtime: n * d <= 40000,
          nameLocation: 'start',
          nameGap: 14,
          nameRotate: named ? 0 : 40,
          nameTextStyle: { color: c.inkSecondary, fontSize: 11, align: named ? 'center' : 'right' },
          axisLine: { lineStyle: { color: c.axis } },
          axisTick: { show: false },
          axisLabel: {
            color: c.muted,
            fontSize: 10,
            formatter: (v: number) => formatNumber(v),
            // The ends of a min–max axis are the data's extremes, not round ticks: their labels only crowd the ticks.
            showMinLabel: scaling === 'standard',
            showMaxLabel: scaling === 'standard',
          },
          splitLine: { show: false },
          areaSelectStyle: { width: 16, borderWidth: 1, borderColor: c.ink, color: c.ink, opacity: 0.12 },
        },
      },
      parallelAxis: order.map((j, p) => ({
        id: axisId(j),
        dim: j,
        name: named ? '' : table.featureNames[j],
        min: scaled.ranges[j][0],
        max: scaled.ranges[j][1],
        inverse: flipped.has(j),
        axisLabel: { show: scaling === 'standard' ? p === 0 : labelled },
      })),
      series,
    }
  }, [
    mode,
    n,
    d,
    k,
    labels,
    scaled,
    hidden,
    table,
    opacity,
    showMedians,
    perClassMedians,
    order,
    flipped,
    scaling,
    named,
    step,
  ])

  // Axis reordering: a chip follows the pointer and drops at the nearest axis position.
  const [drag, setDrag] = useState<{ j: number; dx: number } | null>(null)
  const start = useRef<{ j: number; x: number } | null>(null)
  const onDown = (j: number) => (e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) return
    start.current = { j, x: e.clientX }
    e.currentTarget.setPointerCapture(e.pointerId)
    setDrag({ j, dx: 0 })
  }
  const onMove = (e: ReactPointerEvent<HTMLElement>) => {
    if (start.current) setDrag({ j: start.current.j, dx: e.clientX - start.current.x })
  }
  const onUp = (e: ReactPointerEvent<HTMLElement>) => {
    const s = start.current
    start.current = null
    setDrag(null)
    if (!s || !step) return
    const from = order.indexOf(s.j)
    const to = Math.max(0, Math.min(d - 1, from + Math.round((e.clientX - s.x) / step)))
    if (to === from) return
    const next = order.filter((j) => j !== s.j)
    next.splice(to, 0, s.j)
    setOrder(next)
  }

  const selected = inside ? inside.reduce((a, b) => a + b, 0) : null
  const brushedAxes = Object.values(areas).filter((v) => v.length).length

  return (
    <Figure
      id={id}
      title={title ?? `Parallel coordinates: ${data?.meta.name ?? 'data'}`}
      description={description}
      defaultSize={defaultSize}
      hoverReadout={false}
      controls={
        <>
          {controls}
          <ControlRow label="axes">
            <Select
              label="scaling"
              value={scaling}
              onChange={(v) => {
                setScaling(v)
                setAreas({})
              }}
              options={[
                { value: 'minmax', label: 'min–max (each axis its own range)' },
                { value: 'standard', label: 'standardised (z-scores, one range)' },
              ]}
            />
            <Switch label="class medians" checked={showMedians} onChange={setShowMedians} />
            <Button
              variant="outline"
              size="sm"
              disabled={!brushedAxes && !flipped.size && order.every((j, p) => j === p)}
              onClick={() => {
                setAreas({})
                setFlipped(new Set())
                setOrder(Array.from({ length: d }, (_, j) => j))
              }}
            >
              Reset axes and brushes
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
          <Readout label="axes" value={d} />
          {selected !== null && (
            <Readout
              label="inside the brushes"
              value={`${selected} of ${n} (${brushedAxes} ${brushedAxes === 1 ? 'axis' : 'axes'} brushed)`}
            />
          )}
        </>
      }
      caption={
        caption ??
        'Drag along an axis to brush an interval (drag again to add one; click the axis outside it to clear). Drag a chip in the header to move its axis; its arrow flips the axis.'
      }
    >
      <div ref={box} className="flex w-full flex-col" style={{ height }}>
        <div className="relative shrink-0 select-none" style={{ height: HEADER }}>
          {order.map((j, p) => {
            const at = SIDE + p * step
            const moving = drag?.j === j
            const width = Math.max(28, Math.min(named ? step - 6 : 40, 160))
            return (
              <div
                key={j}
                className={cn(
                  'absolute top-0 flex h-6 items-center gap-0.5 rounded-md border bg-background text-xs shadow-xs',
                  moving && 'z-10 ring-2 ring-ring/50',
                )}
                style={{
                  // Edge chips stay inside the figure; they sit off-centre over their axis rather than clip.
                  left: Math.max(0, Math.min(size.width - width, at - width / 2)),
                  width,
                  transform: moving ? `translateX(${drag.dx}px)` : undefined,
                }}
                title={table.featureNames[j]}
              >
                <span
                  role="button"
                  tabIndex={-1}
                  aria-label={`Move axis ${table.featureNames[j]}`}
                  className="flex h-full min-w-0 flex-1 cursor-grab touch-none items-center gap-0.5 pl-0.5 active:cursor-grabbing"
                  onPointerDown={onDown(j)}
                  onPointerMove={onMove}
                  onPointerUp={onUp}
                  onPointerCancel={onUp}
                >
                  <GripVertical className="size-3 shrink-0 text-muted-foreground" />
                  {named && <span className="truncate">{table.featureNames[j]}</span>}
                </span>
                {(named || width >= 40) && (
                  <button
                    type="button"
                    aria-pressed={flipped.has(j)}
                    aria-label={`Flip axis ${table.featureNames[j]}`}
                    className={cn(
                      'flex size-5 shrink-0 items-center justify-center rounded-sm hover:bg-muted',
                      flipped.has(j) ? 'text-foreground' : 'text-muted-foreground',
                    )}
                    onClick={() => setFlipped((f) => toggled(f, j))}
                  >
                    <ArrowUpDown className="size-3" />
                  </button>
                )}
              </div>
            )
          })}
        </div>
        <div className="relative min-h-0 flex-1">
          <EChart
            option={option}
            height="fill"
            className="absolute inset-0"
            cartesian={false}
            renderer="canvas"
            axisAreas={areas}
            onAxisAreaSelect={(axis, intervals) => setAreas((a) => ({ ...a, [axis]: intervals }))}
            ariaLabel={`Parallel coordinates of ${d} features, one line per row`}
          />
        </div>
      </div>
    </Figure>
  )
}
