import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Trace } from 'aifn-compute/foundation/trace'
import { decimate, seriesComponents } from 'aifn-compute/foundation/trace'
import { Check, Copy } from 'lucide-react'
import { useCallback, useId, useMemo, useState, type ReactNode } from 'react'
import { Button, Player, Switch } from 'aifn-render/controls'
import { PanelSlot } from 'aifn-render/layout'
import { cn } from 'aifn-render/lib/utils'
import { Badge } from 'aifn-render/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from 'aifn-render/ui/table'
import { Curve, Handle, niceRange, Plot, useAxis, type AxisModel, type Range } from 'aifn-render/viz'
import { formatValue } from './format'
import { registerView } from './registry'
import { StateTree } from './StateTree'

export type TracePanelProps<S> = {
  trace: Trace<S>
  /** Series plotted at first; default the first three. */
  show?: string[]
  /** A chart of the current state (e.g. a path drawn up to the current step), shown above the series. */
  renderState?: (state: S, at: { position: number; step: number; trace: Trace<S> }) => ReactNode
  /** Most points drawn per line; longer traces are decimated for drawing (the cursor stays exact). Default 1500. */
  maxPoints?: number
  /** Components drawn per series; further components are listed but not drawn. Default 8 (the palette's slots). */
  maxComponents?: number
  /**
   * Open at the last kept step, saying why (the finished run is the point). Without it the player opens at the first
   * kept step, as every walk-through does.
   */
  startReason?: string
}

/** The kept position whose step number is nearest `step` (index is ascending). */
function nearestPosition(index: ArrayLike<number>, step: number): number {
  let lo = 0
  let hi = index.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (index[mid] <= step) lo = mid
    else hi = mid
  }
  return Math.abs(index[hi] - step) < Math.abs(index[lo] - step) ? hi : lo
}

/** JSON with typed arrays as plain arrays and functions dropped, for copying a state. */
function toJson(value: unknown): string {
  const seen = new WeakSet()
  return JSON.stringify(
    value,
    (_k, v: unknown) => {
      if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v as unknown as ArrayLike<number>)
      if (typeof v === 'function') return undefined
      if (typeof v === 'object' && v !== null) {
        if (seen.has(v)) return '[circular]'
        seen.add(v)
      }
      if (typeof v === 'number' && !Number.isFinite(v)) return String(v)
      return v
    },
    2,
  )
}

/**
 * The generic view of any `Trace`, as one figure: play or step through the kept steps, plot any recorded series
 * against the step (a vector series draws one line per component), inspect the current step's state as a typed tree
 * with the changes since the previous kept step highlighted, and see where the time went, per step and per phase.
 *
 * The series and timing panels share one hover (the hovered step reads out on every panel), and every panel's step
 * cursor can be dragged to scrub. When the trace grows (a time-sliced or extended run) and the cursor is at the end,
 * it follows the end.
 */
export function TracePanel<S>({
  trace,
  show,
  renderState,
  maxPoints = 1500,
  maxComponents = 8,
  startReason,
}: TracePanelProps<S>) {
  const kept = trace.index.length
  const last = kept - 1
  // The cursor is a kept position, or 'end', which follows the end of a growing (time-sliced or extended) trace.
  const [position, setPositionRaw] = useState<number | 'end'>(startReason ? 'end' : 0)
  const [logScale, setLogScale] = useState(false)
  const names = useMemo(() => Object.keys(trace.series), [trace.series])
  const [chosen, setChosen] = useState<string[]>(() => show ?? names.slice(0, 3))
  // Names that this trace does not record (after the trace changes) are ignored rather than dropped.
  const selected = chosen.filter((n) => n in trace.series)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['state']))
  const group = useId()

  const pos = position === 'end' ? last : Math.min(position, last)
  const setPosition = useCallback(
    (p: number) => {
      const clamped = Math.max(0, Math.min(last, Math.round(p)))
      setPositionRaw(clamped === last ? 'end' : clamped)
    },
    [last],
  )
  const step = trace.index[pos]
  const state = trace.steps[pos]
  const previousState = pos > 0 ? trace.steps[pos - 1] : undefined

  // Drawing copy of the trace, decimated once per trace.
  const drawn = useMemo(() => decimate(trace, maxPoints), [trace, maxPoints])
  const drawnIndex = useMemo(() => Array.from(drawn.index), [drawn])
  const onCursor = useCallback((x: number) => setPosition(nearestPosition(trace.index, x)), [setPosition, trace.index])
  // One step axis for every chart: zoom one and all follow. Its range is whole ticks over the kept steps, so a trace
  // that grows by a few steps keeps the axis (and the charts take the new data as a patch).
  const stepRange = niceRange([trace.index[0] ?? 0, Math.max(trace.index[last] ?? 1, (trace.index[0] ?? 0) + 1)])
  const stepAxis = useAxis({ label: 'step', range: stepRange })

  const toggleSeries = (name: string) =>
    setChosen((s) => (s.includes(name) ? s.filter((n) => n !== name) : [...s, name]))

  const stoppedVariant =
    trace.meta.stopped === 'diverged' ? 'destructive' : trace.meta.stopped === 'done' ? 'secondary' : 'outline'

  return (
    <>
      <PanelSlot slot="about">
        <span className="flex flex-col gap-1">
          {/* What ran and how it ended. */}
          <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="font-mono font-medium text-foreground">{trace.meta.algorithm}</span>
            <Badge variant={stoppedVariant}>{trace.meta.stopped}</Badge>
            <span>
              <Num>{trace.meta.steps}</Num> steps · <Num>{kept}</Num> kept
              {trace.meta.every > 1 && (
                <>
                  {' '}
                  (every <Num>{trace.meta.every}</Num>)
                </>
              )}
            </span>
            <span>
              <Num>{formatMs(trace.timing.totalMs)}</Num> in steps ·{' '}
              <Num>{Number.isFinite(trace.timing.perSecond) ? formatRate(trace.timing.perSecond) : '∞'}</Num> steps/s
            </span>
            {trace.checkpoints.index.length > 1 && (
              <span>
                <Num>{trace.checkpoints.index.length}</Num> checkpoints
              </span>
            )}
            <span>
              key <span className="font-mono">{trace.meta.key.path}</span>
            </span>
          </span>
        </span>
      </PanelSlot>
      <PanelSlot slot="controls">
        <>
          <div className="col-span-full">
            <Player
              label={`step (of ${trace.meta.steps}; kept ${pos + 1}/${kept})`}
              value={pos}
              onChange={setPosition}
              count={kept}
              format={(p) => `${trace.index[Math.min(last, Math.round(p))] ?? ''}`}
              startReason={startReason}
            />
          </div>
          {/* Which series to plot. */}
          <div className="col-span-full flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs text-muted-foreground">Series</span>
            {names.length === 0 && <span className="text-xs text-muted-foreground">none recorded</span>}
            {names.map((name) => (
              <Button
                key={name}
                size="xs"
                variant={selected.includes(name) ? 'secondary' : 'ghost'}
                aria-pressed={selected.includes(name)}
                onClick={() => toggleSeries(name)}
                className={cn('font-mono', !selected.includes(name) && 'text-muted-foreground')}
              >
                {selected.includes(name) && <Check />}
                {name}
                <span className="text-[10px] opacity-60">[{trace.series[name].shape.slice(1).join('×') || '·'}]</span>
              </Button>
            ))}
            <div className="ml-auto">
              <Switch label="log y" checked={logScale} onChange={setLogScale} />
            </div>
          </div>
        </>
      </PanelSlot>
      {/* The current state's picture spans the figure: a path or a fit needs the room. */}
      {renderState && <div className="min-w-0">{renderState(state, { position: pos, step, trace })}</div>}
      {/* Side by side only when the figure itself is wide (a container query, not the window): at size M the series
          take the full width and the state sits under them. */}
      <div className="@container">
        <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
          <div className="flex min-w-0 flex-col gap-4">
            {/* Series and timing share the hovered step. */}
            {selected.map((name) => (
              <SeriesChart
                key={name}
                name={name}
                full={trace.series[name]}
                drawn={drawn.series[name]}
                x={drawnIndex}
                xAxis={stepAxis}
                position={pos}
                step={step}
                onCursor={onCursor}
                logScale={logScale}
                maxComponents={maxComponents}
                hoverGroup={group}
              />
            ))}

            <TimingPanel
              trace={trace}
              xAxis={stepAxis}
              step={step}
              onCursor={onCursor}
              maxPoints={maxPoints}
              logScale={logScale}
              hoverGroup={group}
            />
          </div>

          {/* The current state. */}
          <section className="flex min-w-0 flex-col gap-2 @3xl:sticky @3xl:top-4 @3xl:max-h-[calc(100svh-2rem)] @3xl:self-start">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>
                State at step <span className="font-mono text-foreground">{step}</span>
              </span>
              {previousState !== undefined && (
                <span className="flex items-center gap-1">
                  <span className="inline-block size-2 rounded-sm bg-primary/15 ring-1 ring-primary/30" /> changed since
                  step {trace.index[pos - 1]}
                </span>
              )}
              <CopyButton text={() => toJson(state)} />
            </div>
            <div className="min-h-0 overflow-auto rounded-lg border bg-background p-2">
              <StateTree
                value={state}
                previous={previousState}
                expanded={expanded}
                onToggle={(path) =>
                  setExpanded((s) => {
                    const next = new Set(s)
                    if (next.has(path)) next.delete(path)
                    else next.add(path)
                    return next
                  })
                }
              />
            </div>
          </section>
        </div>
      </div>
    </>
  )
}

registerView<Trace<unknown>>({
  key: 'trace/series',
  kind: 'trace',
  description:
    'Play through the kept steps, plot any recorded series against the step, inspect the state at the step, and see the timing.',
  title: (t) => `Trace of ${t.meta.algorithm}`,
  render: (t) => <TracePanel trace={t} />,
})

function Num({ children }: { children: ReactNode }) {
  return <span className="font-mono text-foreground tabular-nums">{children}</span>
}

function CopyButton({ text }: { text: () => string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      size="xs"
      className="ml-auto"
      onClick={() => {
        void navigator.clipboard?.writeText(text()).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1200)
        })
      }}
    >
      {copied ? <Check /> : <Copy />} JSON
    </Button>
  )
}

const formatMs = (ms: number) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${formatValue(Number(ms.toPrecision(3)))} ms`
const formatRate = (r: number) =>
  r >= 1e6 ? `${(r / 1e6).toFixed(1)}M` : r >= 1e3 ? `${(r / 1e3).toFixed(1)}k` : r.toFixed(0)

/** A y axis over `values` (whole ticks; whole decades on a log axis), fixed so new data moves it only by a tick. */
function useValueAxis(label: string, extent: Range | undefined, log: boolean): AxisModel {
  const range = extent ? niceRange(extent, log) : undefined
  return useAxis({ label, log, range: range ? [range[0], range[1]] : undefined })
}

/** The finite extent of some arrays (positive values only for a log axis). */
function finiteExtent(arrays: readonly ArrayLike<number>[], positive: boolean): Range | undefined {
  let lo = Infinity
  let hi = -Infinity
  for (const a of arrays)
    for (let i = 0; i < a.length; i++) {
      const v = a[i]
      if (!Number.isFinite(v) || (positive && v <= 0)) continue
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  return Number.isFinite(lo) ? [lo, hi] : undefined
}

/**
 * One recorded series against the step: one line per component, with the step cursor and the current values. The
 * lines are live layers on axes fixed to whole ticks over the data, so a new trace (a dragged start, a time-sliced run)
 * reaches the chart as a patch rather than a full redraw.
 */
function SeriesChart({
  name,
  full,
  drawn,
  x,
  xAxis,
  position,
  step,
  onCursor,
  logScale,
  maxComponents,
  hoverGroup,
}: {
  name: string
  full: Tensor
  drawn: Tensor
  x: number[]
  xAxis: AxisModel
  position: number
  step: number
  onCursor: (x: number) => void
  logScale: boolean
  maxComponents: number
  hoverGroup: string
}) {
  const components = useMemo(() => seriesComponents(drawn), [drawn])
  const width = components.length
  const { lines, positive } = useMemo(() => {
    const shown = components.slice(0, maxComponents)
    const positive = shown.every((c) => c.values.every((v) => v > 0))
    // Non-finite values (a diverged run) break the axes; draw them as gaps and report them in the readout.
    const lines = shown.map((c) => ({
      name: `${name}${c.label}`,
      y: Array.from(c.values, (v) => (Number.isFinite(v) ? v : NaN)),
    }))
    return { lines, positive }
  }, [components, maxComponents, name])
  const useLog = logScale && positive
  const extent = useMemo(
    () =>
      finiteExtent(
        lines.map((l) => l.y),
        useLog,
      ),
    [lines, useLog],
  )
  const yAxis = useValueAxis(name, extent, useLog)
  // The current values come from the full series, so they are exact at any step.
  const fullValues = useMemo(() => toFlat(full), [full])
  const current = useMemo(
    () => fullValues.slice(position * width, position * width + Math.min(width, 12)).map(formatValue),
    [fullValues, position, width],
  )
  return (
    <div className="flex flex-col gap-1">
      <Plot x={xAxis} y={yAxis} hoverGroup={hoverGroup} scale={0.5} ariaLabel={`${name} against step`}>
        {lines.map((l, k) => (
          <Curve key={k} id={`line${k}`} name={l.name} x={x} y={l.y} slot={k} live />
        ))}
        <Handle kind="x" at={step} label={name} onDrag={onCursor} />
      </Plot>
      <div className="flex flex-wrap gap-x-3 font-mono text-xs text-muted-foreground tabular-nums">
        <span>
          {name} at {step}:
        </span>
        <span className="text-foreground">
          {width === 1 ? current[0] : `[${current.join(', ')}${width > 12 ? ', …' : ''}]`}
        </span>
        {width > maxComponents && (
          <span>
            (drawing {maxComponents} of {width} components)
          </span>
        )}
        {logScale && !positive && <span>(linear: values not all positive)</span>}
      </div>
    </div>
  )
}

/** Per-step time (bucketed means for long runs) and the per-phase totals. */
function TimingPanel<S>({
  trace,
  xAxis,
  step,
  onCursor,
  maxPoints,
  logScale,
  hoverGroup,
}: {
  trace: Trace<S>
  xAxis: AxisModel
  step: number
  onCursor: (x: number) => void
  maxPoints: number
  logScale: boolean
  hoverGroup: string
}) {
  const { stepMs, totalMs, phases } = trace.timing
  const line = useMemo(() => {
    const n = stepMs.length
    const buckets = Math.min(n, maxPoints)
    const x: number[] = []
    const y: number[] = []
    for (let b = 0; b < buckets; b++) {
      const from = Math.floor((b * n) / buckets)
      const to = Math.floor(((b + 1) * n) / buckets)
      let s = 0
      for (let i = from; i < to; i++) s += stepMs[i]
      // Step i takes state i to i + 1, so it is drawn at step i + 1.
      x.push((from + to) / 2 + 0.5)
      y.push(s / (to - from))
    }
    return { name: n > maxPoints ? 'ms per step (bucket mean)' : 'ms per step', x, y }
  }, [stepMs, maxPoints])
  const positive = line.y.every((v) => v > 0)
  const useLog = logScale && positive
  const extent = useMemo(() => finiteExtent([line.y], useLog), [line, useLog])
  const yAxis = useValueAxis('ms', extent && (useLog ? extent : [Math.min(0, extent[0]), extent[1]]), useLog)
  const inStep = Object.entries(phases).filter(([k]) => k !== 'init' && k !== 'record')
  return (
    <section className="flex flex-col gap-2">
      <div className="text-xs text-muted-foreground">Timing</div>
      {stepMs.length > 0 ? (
        <Plot x={xAxis} y={yAxis} hoverGroup={hoverGroup} scale={0.45} ariaLabel="Time per step">
          <Curve id="ms" name={line.name} x={line.x} y={line.y} slot={0} live />
          <Handle kind="x" at={step} label="timing" onDrag={onCursor} />
        </Plot>
      ) : (
        <p className="text-xs text-muted-foreground">No steps were run.</p>
      )}
      <Table className="text-xs">
        <TableHeader>
          <TableRow>
            <TableHead>phase</TableHead>
            <TableHead className="text-right">total</TableHead>
            <TableHead className="text-right">per step</TableHead>
            <TableHead className="text-right">share of step time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody className="font-mono tabular-nums">
          <PhaseRow name="step (all)" ms={totalMs} steps={stepMs.length} share={totalMs > 0 ? 1 : undefined} strong />
          {inStep.map(([name, ms]) => (
            <PhaseRow
              key={name}
              name={name}
              ms={ms}
              steps={stepMs.length}
              share={totalMs > 0 ? ms / totalMs : undefined}
              indent
            />
          ))}
          {'init' in phases && <PhaseRow name="init" ms={phases.init} />}
          {'record' in phases && <PhaseRow name="record" ms={phases.record} steps={trace.index.length} />}
        </TableBody>
      </Table>
    </section>
  )
}

function PhaseRow({
  name,
  ms,
  steps,
  share,
  strong,
  indent,
}: {
  name: string
  ms: number
  steps?: number
  share?: number
  strong?: boolean
  indent?: boolean
}) {
  return (
    <TableRow>
      <TableCell className={cn(strong && 'font-medium', indent && 'pl-6')}>{name}</TableCell>
      <TableCell className="text-right">{formatMs(ms)}</TableCell>
      <TableCell className="text-right">{steps ? formatMs(ms / steps) : ''}</TableCell>
      <TableCell className="text-right">
        {share !== undefined && (
          <span className="inline-flex items-center justify-end gap-2">
            <span className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
              <span className="block h-full bg-primary/60" style={{ width: `${Math.min(100, share * 100)}%` }} />
            </span>
            {(share * 100).toFixed(0)}%
          </span>
        )}
      </TableCell>
    </TableRow>
  )
}
