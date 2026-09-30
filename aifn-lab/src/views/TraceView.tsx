import { toFlat, type Tensor } from 'aifn/tensor'
import type { Trace } from 'aifn/trace'
import { decimate, seriesComponents } from 'aifn/trace'
import { Check, Copy } from 'lucide-react'
import { useCallback, useId, useMemo, useState, type ReactNode } from 'react'
import { Button, Player, Switch } from '@lab/controls'
import { Figure } from '@lab/layout'
import { cn } from '@lab/lib/utils'
import { Badge } from '@lab/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@lab/ui/table'
import { ChartSize, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { formatValue } from './format'
import type { FrameProps } from './frame'
import { StateTree } from './StateTree'

export type TraceViewProps<S> = FrameProps & {
  trace: Trace<S>
  /** Series plotted at first; default the first three. */
  show?: string[]
  /** A chart of the current state (e.g. a path drawn up to the current step), shown above the series. */
  renderState?: (state: S, at: { position: number; step: number; trace: Trace<S> }) => ReactNode
  /** Most points drawn per line; longer traces are decimated for drawing (the cursor stays exact). Default 1500. */
  maxPoints?: number
  /** Components drawn per series; further components are listed but not drawn. Default 8 (the palette's slots). */
  maxComponents?: number
  /** Start at the first kept step instead of the last. */
  startAtFirst?: boolean
}

/** The kept position whose step number is nearest `step` (index is ascending). */
function nearestPosition(index: readonly number[], step: number): number {
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
export function TraceView<S>({
  trace,
  show,
  renderState,
  maxPoints = 1500,
  maxComponents = 8,
  startAtFirst = false,
  title,
  description,
  controls,
  readouts,
  ...frame
}: TraceViewProps<S>) {
  const kept = trace.index.length
  const last = kept - 1
  // The cursor is a kept position, or 'end', which follows the end of a growing (time-sliced or extended) trace.
  const [position, setPositionRaw] = useState<number | 'end'>(startAtFirst ? 0 : 'end')
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
  const cursor = useCallback(
    (label: string): Handle[] => [
      { kind: 'x', at: step, label, onDrag: (x: number) => setPosition(nearestPosition(trace.index, x)) },
    ],
    [step, setPosition, trace.index],
  )

  const toggleSeries = (name: string) =>
    setChosen((s) => (s.includes(name) ? s.filter((n) => n !== name) : [...s, name]))

  const stoppedVariant =
    trace.meta.stopped === 'diverged' ? 'destructive' : trace.meta.stopped === 'done' ? 'secondary' : 'outline'

  return (
    <Figure
      title={title ?? `Trace of ${trace.meta.algorithm}`}
      defaultSize="full"
      {...frame}
      description={
        <span className="flex flex-col gap-1">
          {description}
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
            {trace.meta.seed !== undefined && (
              <span>
                seed <Num>{trace.meta.seed}</Num>
              </span>
            )}
          </span>
        </span>
      }
      controls={
        <>
          {controls}
          <div className="col-span-full">
            <Player
              label={`step (of ${trace.meta.steps}; kept ${pos + 1}/${kept})`}
              value={pos}
              onChange={setPosition}
              count={kept}
              format={(p) => `${trace.index[Math.min(last, Math.round(p))] ?? ''}`}
              defaultSpeed={30}
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
      }
      readouts={readouts}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
        <div className="flex min-w-0 flex-col gap-4">
          {renderState && <div>{renderState(state, { position: pos, step, trace })}</div>}

          {/* Series and timing share the hovered step. */}
          <ChartSize scale={0.5}>
            {selected.map((name) => (
              <SeriesChart
                key={name}
                name={name}
                full={trace.series[name]}
                drawn={drawn.series[name]}
                x={drawn.index}
                position={pos}
                step={step}
                handles={cursor(name)}
                logScale={logScale}
                maxComponents={maxComponents}
                hoverGroup={group}
              />
            ))}
          </ChartSize>

          <TimingPanel
            trace={trace}
            handles={cursor('timing')}
            maxPoints={maxPoints}
            logScale={logScale}
            hoverGroup={group}
          />
        </div>

        {/* The current state. */}
        <section className="flex min-w-0 flex-col gap-2 lg:sticky lg:top-4 lg:max-h-[calc(100svh-2rem)] lg:self-start">
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
    </Figure>
  )
}

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

/** One recorded series against the step: one line per component, with the step cursor and the current values. */
function SeriesChart({
  name,
  full,
  drawn,
  x,
  position,
  step,
  handles,
  logScale,
  maxComponents,
  hoverGroup,
}: {
  name: string
  full: Tensor
  drawn: Tensor
  x: number[]
  position: number
  step: number
  handles: Handle[]
  logScale: boolean
  maxComponents: number
  hoverGroup: string
}) {
  const components = useMemo(() => seriesComponents(drawn), [drawn])
  const width = components.length
  const { series, positive } = useMemo(() => {
    const drawnComponents = components.slice(0, maxComponents)
    const positive = drawnComponents.every((c) => c.values.every((v) => v > 0))
    const series: XYSeries[] = drawnComponents.map((c, k) => ({
      name: `${name}${c.label}`,
      type: 'line',
      // Non-finite values (a diverged run) break the axes; draw them as gaps and report them in the readout.
      x,
      y: Array.from(c.values, (v) => (Number.isFinite(v) ? v : NaN)),
      slot: k,
    }))
    return { series, positive }
  }, [components, maxComponents, name, x])
  // The current values come from the full series, so they are exact at any step.
  const fullValues = useMemo(() => toFlat(full), [full])
  const current = useMemo(
    () => fullValues.slice(position * width, position * width + Math.min(width, 12)).map(formatValue),
    [fullValues, position, width],
  )
  const useLog = logScale && positive
  return (
    <div className="flex flex-col gap-1">
      <XYChart
        series={series}
        xLabel="step"
        yLabel={name}
        yLog={useLog}
        handles={handles}
        hoverGroup={hoverGroup}
        ariaLabel={`${name} against step`}
      />
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
  handles,
  maxPoints,
  logScale,
  hoverGroup,
}: {
  trace: Trace<S>
  handles: Handle[]
  maxPoints: number
  logScale: boolean
  hoverGroup: string
}) {
  const { stepMs, totalMs, phases } = trace.timing
  const series = useMemo((): XYSeries[] => {
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
    return [{ name: n > maxPoints ? 'ms per step (bucket mean)' : 'ms per step', type: 'line', x, y, slot: 0 }]
  }, [stepMs, maxPoints])
  const inStep = Object.entries(phases).filter(([k]) => k !== 'init' && k !== 'record')
  const positive = series[0].y.every((v) => v > 0)
  return (
    <section className="flex flex-col gap-2">
      <div className="text-xs text-muted-foreground">Timing</div>
      {stepMs.length > 0 ? (
        <ChartSize scale={0.35}>
          <XYChart
            series={series}
            xLabel="step"
            yLabel="ms"
            yLog={logScale && positive}
            handles={handles}
            hoverGroup={hoverGroup}
            ariaLabel="Time per step"
          />
        </ChartSize>
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
