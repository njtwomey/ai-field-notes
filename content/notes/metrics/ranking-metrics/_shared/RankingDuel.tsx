import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { useTheme } from '@/components/theme-provider'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  seriesColor,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { cn } from '@/lib/utils'
import { isRelevant } from './ranking'
import { METRICS, listWith, verdict, type MetricId, type MetricSpec, type Verdict } from './duel'

/**
 * "Which ranking is better?": two ranked lists of the same query, drawn as strips of documents with the relevant ones
 * coloured, compared under one metric after another. Adapted from Jaime Arguello's INLS 509 lecture "Evaluation
 * Overview" (UNC Chapel Hill), whose slides show two such strips and ask the question for several metrics in turn.
 */

const N = 37
const MAX_GRADE = 3

export type PresetId = 'slides' | 'early' | 'order' | 'graded'

type Preset = {
  label: string
  a: number[]
  b: number[]
  k: number
  total: number
  graded: boolean
  metric: MetricId
}

const PRESETS: Record<PresetId, Preset> = {
  slides: {
    label: 'Arguello’s example',
    a: listWith(N, [5, 7, 11]),
    b: listWith(N, [11, 13, 17, 18, 19, 21, 24, 28]),
    k: 10,
    total: 10,
    graded: false,
    metric: 'first',
  },
  early: {
    label: 'A wins P@1, B wins R@30',
    a: listWith(N, [1, 36]),
    b: listWith(N, [3, 4, 5, 6, 8, 9, 12, 15]),
    k: 1,
    total: 8,
    graded: false,
    metric: 'p',
  },
  order: {
    label: 'Same P@10, different order',
    a: listWith(N, [1, 2, 3]),
    b: listWith(N, [8, 9, 10]),
    k: 10,
    total: 5,
    graded: false,
    metric: 'p',
  },
  graded: {
    label: 'One excellent vs several fair',
    a: listWith(N, { 1: 3 }),
    b: listWith(N, { 1: 1, 2: 1, 3: 2, 4: 1, 5: 1 }),
    k: 10,
    total: 6,
    graded: true,
    metric: 'err',
  },
}

const ALL_PRESETS = Object.keys(PRESETS) as PresetId[]
const SPEC = Object.fromEntries(METRICS.map((m) => [m.id, m])) as Record<MetricId, MetricSpec>
const KS = Array.from({ length: N }, (_, i) => i + 1)

// Strip geometry, in SVG units. Rows are ROW apart; each bar is BAR tall, leaving a gap as on the slides.
const TOP = 26
const ROW = 12
const BAR = 7
const VIEW_W = 350
const VIEW_H = TOP + N * ROW + 10
const STRIPS = [
  { x: 40, w: 96, arrow: 146 },
  { x: 190, w: 96, arrow: 296 },
] as const
const NAMES = ['A', 'B'] as const

type Gesture = { kind: 'cut' } | { kind: 'bar'; list: 0 | 1; from: number; startY: number; moved: boolean } | null

export type RankingDuelProps = {
  /** Preset loaded first. */
  preset?: PresetId
  /** Metric selected first; defaults to the preset's. */
  metric?: MetricId
  /** Presets offered as buttons, in this order. */
  presets?: PresetId[]
  title?: string
}

export function RankingDuel({
  preset = 'slides',
  metric: initialMetric,
  presets = ALL_PRESETS,
  title = 'Which ranking is better?',
}: RankingDuelProps) {
  const start = PRESETS[preset]
  const { resolved: mode } = useTheme()
  const [lists, setLists] = useState<[number[], number[]]>([start.a, start.b])
  const [graded, setGraded] = useState(start.graded)
  const [metric, setMetric] = useState<MetricId>(initialMetric ?? start.metric)
  const k = useParam(start.k, { min: 1, max: N, step: 1 })
  const totalParam = useParam(start.total, { min: 1, max: N, step: 1 })

  const found = lists.map((g) => g.filter(isRelevant).length)
  // R can never be below the number of relevant documents either list has already retrieved.
  const floor = Math.max(...found)
  const total = Math.max(totalParam.value, floor)
  const maxGrade = graded ? MAX_GRADE : 1
  const spec = SPEC[metric]

  const load = (id: PresetId) => {
    const p = PRESETS[id]
    setLists([p.a, p.b])
    setGraded(p.graded)
    setMetric(p.metric)
    k.set(p.k)
    totalParam.set(p.total)
  }

  const toggle = (list: 0 | 1, i: number) =>
    setLists((ls) => {
      const next: [number[], number[]] = [ls[0], ls[1]]
      next[list] = ls[list].map((g, j) => (j === i ? (graded ? (g + 1) % (MAX_GRADE + 1) : g > 0 ? 0 : 1) : g))
      return next
    })

  const moveDoc = (list: 0 | 1, from: number, to: number) =>
    setLists((ls) => {
      if (from === to) return ls
      const moved = [...ls[list]]
      const [doc] = moved.splice(from, 1)
      moved.splice(to, 0, doc)
      const next: [number[], number[]] = [ls[0], ls[1]]
      next[list] = moved
      return next
    })

  const setGradedMode = (on: boolean) => {
    setGraded(on)
    if (!on) setLists((ls) => [ls[0].map((g) => Math.min(g, 1)), ls[1].map((g) => Math.min(g, 1))])
  }

  // ---- The strips: pointer gestures on the SVG ----
  const svg = useRef<SVGSVGElement>(null)
  const gesture = useRef<Gesture>(null)
  const [dragging, setDragging] = useState<{ list: 0 | 1; from: number; to: number; y: number } | null>(null)

  const toSvgY = (e: ReactPointerEvent) => {
    const rect = svg.current!.getBoundingClientRect()
    return ((e.clientY - rect.top) * VIEW_H) / rect.height
  }
  const rowAt = (y: number) => Math.min(N - 1, Math.max(0, Math.floor((y - TOP) / ROW)))
  const cutFrom = (y: number) => Math.round((y - TOP) / ROW)

  const onDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return
    const target = e.target as SVGElement
    const y = toSvgY(e)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    const bar = target.dataset.list
    // The cut-off line wins within a few units of it; otherwise a bar starts a click or a drag; elsewhere the press
    // moves the cut-off, as a chart handle does.
    if (Math.abs(y - (TOP + k.value * ROW - (ROW - BAR) / 2)) < 5 || bar === undefined) {
      gesture.current = { kind: 'cut' }
      k.set(cutFrom(y))
      return
    }
    gesture.current = {
      kind: 'bar',
      list: Number(bar) as 0 | 1,
      from: Number(target.dataset.rank),
      startY: y,
      moved: false,
    }
  }
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const g = gesture.current
    if (!g) return
    const y = toSvgY(e)
    if (g.kind === 'cut') return k.set(cutFrom(y))
    if (!g.moved && Math.abs(y - g.startY) < ROW / 2) return
    g.moved = true
    setDragging({ list: g.list, from: g.from, to: rowAt(y), y })
  }
  const onUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    const g = gesture.current
    gesture.current = null
    setDragging(null)
    if (e.currentTarget.hasPointerCapture?.(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId)
    if (!g || g.kind !== 'bar') return
    if (g.moved) moveDoc(g.list, g.from, rowAt(toSvgY(e)))
    else toggle(g.list, g.from)
  }

  const cutY = TOP + k.value * ROW - (ROW - BAR) / 2
  const opacity = (g: number) => (graded ? [0, 0.4, 0.7, 1][g] : 1)

  const strips = (
    <svg
      ref={svg}
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="mx-auto block w-full max-w-[360px] touch-none select-none"
      role="img"
      aria-label={`Two ranked lists of ${N} documents, A and B, with relevant documents coloured and a cut-off after rank ${k.value}`}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
    >
      <rect x={0} y={0} width={VIEW_W} height={VIEW_H} fill="transparent" className="cursor-row-resize" />
      {[1, 5, 10, 15, 20, 25, 30, 35].map((r) => (
        <text
          key={r}
          x={30}
          y={TOP + (r - 1) * ROW + BAR - 0.5}
          textAnchor="end"
          className="pointer-events-none fill-muted-foreground font-sans text-[8px] tabular-nums"
        >
          {r}
        </text>
      ))}
      {STRIPS.map((s, list) => (
        <g key={list}>
          <text
            x={s.x + s.w / 2}
            y={14}
            textAnchor="middle"
            className="pointer-events-none fill-foreground font-sans text-[12px] font-medium"
          >
            {NAMES[list]}
          </text>
          {lists[list].map((g, i) => {
            const isDragged = dragging && dragging.list === list && dragging.from === i
            return (
              <rect
                key={i}
                data-list={list}
                data-rank={i}
                x={s.x}
                y={TOP + i * ROW}
                width={s.w}
                height={BAR}
                rx={1}
                className={cn('cursor-pointer', !isRelevant(g) && 'fill-muted-foreground/25')}
                style={isRelevant(g) ? { fill: seriesColor(mode, list), fillOpacity: opacity(g) } : undefined}
                opacity={isDragged ? 0.3 : 1}
              >
                <title>
                  {`${NAMES[list]}, rank ${i + 1}: ${isRelevant(g) ? (graded ? `relevant, grade ${g}` : 'relevant') : 'not relevant'}`}
                </title>
              </rect>
            )
          })}
          <line
            x1={s.arrow}
            x2={s.arrow}
            y1={TOP}
            y2={TOP + N * ROW - 4}
            className="pointer-events-none stroke-foreground"
            strokeWidth={1.5}
          />
          <path
            d={`M ${s.arrow - 4} ${TOP + N * ROW - 8} L ${s.arrow} ${TOP + N * ROW - 1} L ${s.arrow + 4} ${TOP + N * ROW - 8} Z`}
            className="pointer-events-none fill-foreground"
          />
        </g>
      ))}
      {dragging && (
        <g className="pointer-events-none">
          <rect
            x={STRIPS[dragging.list].x - 3}
            y={TOP + dragging.to * ROW - 2}
            width={STRIPS[dragging.list].w + 6}
            height={BAR + 4}
            rx={2}
            className="fill-none stroke-foreground"
            strokeDasharray="3 2"
          />
          <rect
            x={STRIPS[dragging.list].x}
            y={dragging.y - BAR / 2}
            width={STRIPS[dragging.list].w}
            height={BAR}
            rx={1}
            className={cn(!isRelevant(lists[dragging.list][dragging.from]) && 'fill-muted-foreground/40')}
            style={
              isRelevant(lists[dragging.list][dragging.from])
                ? { fill: seriesColor(mode, dragging.list), fillOpacity: 0.8 }
                : undefined
            }
          />
        </g>
      )}
      <line
        x1={STRIPS[0].x - 6}
        x2={STRIPS[1].arrow + 6}
        y1={cutY}
        y2={cutY}
        className="pointer-events-none stroke-foreground"
        strokeWidth={1.5}
        strokeDasharray="5 3"
      />
      <text
        x={VIEW_W - 2}
        y={cutY + 3}
        textAnchor="end"
        className="pointer-events-none fill-foreground font-sans text-[10px] tabular-nums"
      >
        {`k = ${k.value}`}
      </text>
    </svg>
  )

  // ---- Scores and verdicts ----
  const rows = METRICS.map((m) => {
    const a = m.value(lists[0], k.value, total, maxGrade)
    const b = m.value(lists[1], k.value, total, maxGrade)
    return { m, a, b, v: verdict(m, a, b) }
  })
  const current = rows.find((r) => r.m.id === metric)!

  // The curve panel plots the chosen metric over k; metrics without a cut-off fall back to P@k and R@k.
  const curveMetrics = curvesFor(spec)
  const curves = useMemo(
    () =>
      curvesFor(spec).flatMap((m, j) =>
        lists.map((g, list): XYSeries => ({
          name: `${m.label('k')} ${NAMES[list]}`,
          type: 'line',
          x: KS,
          y: KS.map((kk) => m.value(g, kk, total, maxGrade)),
          slot: list,
          dashed: j === 1,
        })),
      ),
    [lists, total, maxGrade, spec],
  )
  const handles = useMemo<Handle[]>(() => [{ kind: 'x', at: k.value, onDrag: k.set, label: `k = ${k.value}` }], [k])
  const runs = verdictRuns(curveMetrics[0], lists, total, maxGrade)

  const winnerText = (v: Verdict) => (v === 'tie' ? 'tie' : `${v} is better`)
  // Ranks are integers; scores get three decimals so that columns line up.
  const valueText = (x: number) => (!Number.isFinite(x) ? 'none' : Number.isInteger(x) ? String(x) : x.toFixed(3))

  return (
    <Interactive
      title={title}
      caption={
        <>
          Two systems, A and B, rank the same {N} documents for one query; coloured bars are relevant, grey bars are
          not. Click a bar to toggle its relevance (with graded relevance on, clicks cycle the grade 0–3 and darker is
          more relevant). Drag a bar up or down to move that document to another rank. Drag the dashed line, in the
          strips or on the curve chart, to set the cut-off k. R is the number of relevant documents in the whole
          collection; it cannot fall below what either list has found. Adapted from Jaime Arguello&rsquo;s INLS 509
          lecture &ldquo;Evaluation Overview&rdquo; (UNC Chapel Hill).
        </>
      }
      controls={
        <>
          <ParamSlider label="cut-off k" param={k} format={(v) => String(v)} withArrows />
          <ParamSlider
            label="relevant documents in the collection (R)"
            param={totalParam}
            format={(v) => String(Math.max(v, floor))}
            withArrows
          />
          <ParamSwitch label="graded relevance (0–3)" checked={graded} onChange={setGradedMode} />
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamChoice
              label="metric"
              value={metric}
              onChange={setMetric}
              options={METRICS.map((m) => ({ value: m.id, label: m.label(k.value) }))}
            />
          </div>
          <div className="flex flex-wrap items-end gap-2 sm:col-span-2 lg:col-span-3">
            <span className="w-full text-xs text-muted-foreground">presets</span>
            {presets.map((id) => (
              <ParamButton key={id} onClick={() => load(id)}>
                {PRESETS[id].label}
              </ParamButton>
            ))}
          </div>
        </>
      }
      readout={
        <>
          <Readout label={current.m.label(k.value)} value={`A ${valueText(current.a)} · B ${valueText(current.b)}`} />
          <Readout label="verdict" value={winnerText(current.v)} />
          <Readout label={`${curveMetrics[0].label('k')} over k`} value={runs} />
          <Readout label="R" value={total} />
        </>
      }
    >
      <div className="grid gap-6 md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        {strips}
        <div className="flex min-w-0 flex-col gap-3">
          <XYChart
            height={260}
            xLabel="cut-off k"
            yLabel={spec.usesK ? spec.label('k') : 'P@k (solid), R@k (dashed)'}
            xRange={[1, N]}
            yRange={[0, 1]}
            series={curves}
            handles={handles}
            ariaLabel="Scores of lists A and B as the cut-off k grows"
          />
          <table className="w-full font-sans text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="py-1 text-left font-normal">metric</th>
                <th className="py-1 pl-3 text-right font-normal">A</th>
                <th className="py-1 pl-3 text-right font-normal">B</th>
                <th className="py-1 pl-3 text-left font-normal">better</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ m, a, b, v }) => (
                <tr
                  key={m.id}
                  onClick={() => setMetric(m.id)}
                  className={cn('cursor-pointer border-b last:border-0', m.id === metric && 'bg-muted')}
                >
                  <td className="py-1 pl-1">
                    {m.label(k.value)}
                    {m.lowerIsBetter ? ' (lower is better)' : ''}
                  </td>
                  <td className="py-1 pl-3 text-right font-mono tabular-nums">{valueText(a)}</td>
                  <td className="py-1 pl-3 text-right font-mono tabular-nums">{valueText(b)}</td>
                  <td className="py-1 pl-3">
                    {v === 'tie' ? (
                      <span className="text-muted-foreground">tie</span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 font-medium">
                        <span
                          className="inline-block size-2 rounded-full"
                          style={{ background: seriesColor(mode, v === 'A' ? 0 : 1) }}
                        />
                        {v}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Interactive>
  )
}

const curvesFor = (spec: MetricSpec): MetricSpec[] => (spec.usesK ? [spec] : [SPEC.p, SPEC.r])

/** The verdict of `m` at every cut-off, as runs: "tie 1–4 · A 5–17 · B 18–37". */
function verdictRuns(m: MetricSpec, lists: number[][], total: number, maxGrade: number): string {
  const out: { v: Verdict; from: number; to: number }[] = []
  for (const kk of KS) {
    const v = verdict(m, m.value(lists[0], kk, total, maxGrade), m.value(lists[1], kk, total, maxGrade))
    const last = out[out.length - 1]
    if (last && last.v === v) last.to = kk
    else out.push({ v, from: kk, to: kk })
  }
  return out.map((r) => `${r.v} ${r.from === r.to ? r.from : `${r.from}–${r.to}`}`).join(' · ')
}
