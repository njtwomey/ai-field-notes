/**
 * Panels shared by the subgroup-discovery and exceptional-model pages: the search run and its history, the search tree
 * by depth and quality, quality against optimistic estimate, the progress curves, the ranked results, and the
 * refine-by-hand panel. Every number is read from aifn's search states (`refinementSearchSteps` via
 * `subgroupDiscoverySteps`, `searchHistory`) and its subgroup functions; the lab only lays them out.
 */
import type { PlantedPattern } from 'aifn-methods/data'
import {
  bitsetCount,
  bitsetJaccard,
  compatibleSelector,
  cutPoints,
  descriptionKey,
  formatDescription,
  selectorKey,
  type Description,
  type QualityMeasure,
  type Selector,
  type SelectorLanguage,
  type SelectorOp,
} from 'aifn-compute/learning/subgroups'
import type { SearchVisit } from 'aifn-compute/optim/search'
import { f3, plantedMatch, visibleAt, type Run } from './run'
import { Pin, PinOff, Plus, RotateCcw, X } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { Select } from 'aifn-render/controls'
import { cn } from 'aifn-render/lib/utils'
import { Button } from 'aifn-render/ui/button'
import { Annotation, Curve, Plot, Points, Readout, Segments, type AxisModel } from 'aifn-render/viz'

// ── The search tree: depth × quality ────────────────────────────────────────────────────────────────────────────

export function SearchTreePlot({
  run,
  k,
  pinned,
  onPin,
  x,
  y,
}: {
  run: Run
  k: number
  pinned: number | null
  onPin: (id: number | null) => void
  x: AxisModel
  y: AxisModel
}) {
  const state = run.steps[k]
  const vis = useMemo(() => visibleAt(run, k), [run, k])
  const v = run.history.visits
  const edges = useMemo(
    () =>
      vis.ids
        .filter((id) => v[id].parent >= 0 && Number.isFinite(v[v[id].parent].quality))
        .map((id) => ({
          from: [run.slot[v[id].parent], v[v[id].parent].quality] as const,
          to: [run.slot[id], v[id].quality] as const,
        })),
    [vis, v, run],
  )
  const layer = (f: number) => {
    const ids = vis.ids.filter((_, i) => vis.fate[i] === f)
    return { x: ids.map((id) => run.slot[id]), y: ids.map((id) => v[id].quality) }
  }
  const now = [...state.expanded, ...state.discarded].filter((n) => Number.isFinite(n.quality))
  const results = state.results
  const nearest = (p: [number, number]) => {
    const qs = vis.ids.map((id) => v[id].quality)
    const sy = Math.max(1e-12, Math.max(...qs) - Math.min(...qs))
    let best: number | null = null
    let d = Infinity
    for (const id of vis.ids) {
      const dd = ((run.slot[id] - p[0]) / 0.6) ** 2 + ((v[id].quality - p[1]) / sy) ** 2 * 25
      if (dd < d) [best, d] = [id, dd]
    }
    return d < 0.05 ? best : null
  }
  return (
    <Plot x={x} y={y} onPlotClick={(p) => onPin(nearest(p))} ariaLabel="The search tree by depth and quality">
      <Segments segments={edges} muted width={0.6} />
      <Points name="not refined" {...layer(3)} muted size={4} />
      <Points name="open" {...layer(0)} slot={0} size={5} />
      <Points name="expanded" {...layer(1)} slot={1} size={5} />
      <Points name="pruned" {...layer(2)} slot={2} shape={2} size={5} />
      <Points
        name="in the top k"
        x={results.map((r) => run.slot[r.id])}
        y={results.map((r) => r.quality)}
        emphasis
        shape={3}
        size={9}
      />
      {now.length > 0 && (
        <Points
          name="this step"
          x={now.map((n) => run.slot[n.id])}
          y={now.map((n) => n.quality)}
          emphasis
          size={11}
          shape={4}
        />
      )}
      {pinned !== null && pinned < v.length && Number.isFinite(v[pinned].quality) && (
        <Annotation at={[run.slot[pinned], v[pinned].quality]} text={formatDescription(v[pinned].node)} />
      )}
      {Number.isFinite(state.threshold) && <Annotation y={state.threshold} text="k-th best" dashed />}
    </Plot>
  )
}

// ── Quality against optimistic estimate ─────────────────────────────────────────────────────────────────────────

export function BoundPlot({ run, k, x, y }: { run: Run; k: number; x: AxisModel; y: AxisModel }) {
  const vis = useMemo(() => visibleAt(run, k), [run, k])
  const v = run.history.visits
  const finite = vis.ids.map((id, i) => [id, vis.fate[i]] as const).filter(([id]) => Number.isFinite(v[id].bound))
  if (!finite.length)
    return (
      <Empty>
        This measure has no optimistic estimate, so every node is refined until the depth limit or the beam drops it.
      </Empty>
    )
  const layer = (f: number) => {
    const ids = finite.filter(([, g]) => g === f).map(([id]) => id)
    return { x: ids.map((id) => v[id].quality), y: ids.map((id) => v[id].bound) }
  }
  const lo = Math.min(...finite.map(([id]) => v[id].quality))
  const hi = Math.max(...finite.map(([id]) => v[id].bound))
  const t = run.steps[k].threshold
  return (
    <Plot x={x} y={y} ariaLabel="Quality against optimistic estimate">
      <Curve name="estimate = quality" x={[lo, hi]} y={[lo, hi]} muted dashed silent />
      <Points name="not refined" {...layer(3)} muted size={4} />
      <Points name="open" {...layer(0)} slot={0} size={5} />
      <Points name="expanded" {...layer(1)} slot={1} size={5} />
      <Points name="pruned" {...layer(2)} slot={2} shape={2} size={5} />
      {Number.isFinite(t) && <Annotation y={t} text="prune below: k-th best" dashed />}
    </Plot>
  )
}

// ── Progress over steps ─────────────────────────────────────────────────────────────────────────────────────────

export function ProgressPlot({ run, k, x, y }: { run: Run; k: number; x: AxisModel; y: AxisModel }) {
  const series = useMemo(() => {
    const t = run.steps.map((s) => s.t)
    return {
      t,
      best: run.steps.map((s) => (Number.isFinite(s.best) ? s.best : NaN)),
      threshold: run.steps.map((s) => (Number.isFinite(s.threshold) ? s.threshold : NaN)),
    }
  }, [run])
  return (
    <Plot x={x} y={y} ariaLabel="Best and k-th best quality by step">
      <Curve name="best" x={series.t} y={series.best} emphasis />
      <Curve name="k-th best (threshold)" x={series.t} y={series.threshold} slot={2} dashed />
      <Annotation x={k} text={`step ${k}`} dashed />
    </Plot>
  )
}

// ── Ranked results ──────────────────────────────────────────────────────────────────────────────────────────────

export function ResultsTable({
  lang,
  results,
  planted,
  focus,
  pinned,
  onHover,
  onPin,
  quality = 'quality',
}: {
  lang: SelectorLanguage
  results: readonly SearchVisit<Description>[]
  planted: readonly PlantedPattern[]
  focus: number | null
  pinned: number | null
  onHover: (id: number | null) => void
  onPin: (id: number) => void
  quality?: string
}) {
  const th = 'px-2 py-1 text-left font-normal text-muted-foreground'
  if (!results.length) return <Empty>No results yet: play the search forward.</Empty>
  return (
    <div className="h-full overflow-auto">
      <table className="w-full border-collapse text-xs tabular-nums" onMouseLeave={() => onHover(null)}>
        <thead>
          <tr className="border-b">
            <th className={th}>pin</th>
            <th className={th}>#</th>
            <th className={th}>description</th>
            <th className={th}>n</th>
            <th className={th}>{quality}</th>
            {planted.length > 0 && <th className={th}>planted (Jaccard)</th>}
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => {
            const m = planted.length ? plantedMatch(lang, planted, r.node) : null
            return (
              <tr
                key={r.id}
                className={cn('cursor-pointer border-b border-border/50', r.id === focus && 'bg-muted')}
                onMouseEnter={() => onHover(r.id)}
                onClick={() => onPin(r.id)}
              >
                <td className="px-2 py-0.5">
                  <button
                    type="button"
                    aria-label={`pin ${formatDescription(r.node)}`}
                    aria-pressed={pinned === r.id}
                    className="rounded p-0.5 text-muted-foreground hover:text-foreground"
                    onClick={(e) => {
                      e.stopPropagation()
                      onPin(r.id)
                    }}
                  >
                    {pinned === r.id ? <PinOff className="size-3.5" /> : <Pin className="size-3.5" />}
                  </button>
                </td>
                <td className="px-2 py-0.5 font-mono">{i + 1}</td>
                <td className="px-2 py-0.5 font-mono">{formatDescription(r.node)}</td>
                <td className="px-2 py-0.5 font-mono">{bitsetCount(lang.cover(r.node))}</td>
                <td className="px-2 py-0.5 font-mono">{f3(r.quality)}</td>
                {m && (
                  <td className="px-2 py-0.5 font-mono">
                    {m.jaccard >= 0.5 ? `${m.index === 0 ? 'main' : `#${m.index + 1}`} (${f3(m.jaccard)})` : '—'}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Each planted pattern's own quality, and the rank of the result that matches it best (Jaccard ≥ 0.5). */
export function PlantedReadouts({
  lang,
  measure,
  planted,
  results,
}: {
  lang: SelectorLanguage
  measure: QualityMeasure
  planted: readonly PlantedPattern[]
  results: readonly SearchVisit<Description>[]
}) {
  return (
    <>
      {planted.map((p, i) => {
        const cover = lang.cover(p.description)
        let rank = -1
        let jaccard = 0
        results.forEach((r, j) => {
          const s = bitsetJaccard(cover, lang.cover(r.node))
          if (s >= 0.5 && s > jaccard) [rank, jaccard] = [j + 1, s]
        })
        return (
          <Readout
            key={i}
            label={formatDescription(p.description)}
            value={`q ${f3(measure.quality(cover))} · ${rank > 0 ? `rank ${rank} (J ${f3(jaccard)})` : 'not in the top k'}`}
          />
        )
      })}
    </>
  )
}

// ── Refining a description by hand ──────────────────────────────────────────────────────────────────────────────

const NUMERIC_OPS: SelectorOp[] = ['≥', '≤']
const NOMINAL_OPS: SelectorOp[] = ['=', '≠']

/** Cut points offered for a numeric attribute: the language's, or equal-frequency cuts of the whole column. */
function offeredCuts(lang: SelectorLanguage, column: readonly number[], attribute: string): number[] {
  const a = lang.attributes.find((x) => x.name === attribute)!
  return a.cuts.length ? [...a.cuts] : cutPoints(column, Math.max(lang.bins, 4), 'equal-frequency')
}

/**
 * The pinned description, editable: remove any selector, or add one (attribute, operator, value), and read the
 * quality, size and would-be rank of the result live. `onChange` reports the edited description (for the inspector).
 */
export function RefinePanel({
  lang,
  measure,
  base,
  table,
  results,
  edited,
  onChange,
}: {
  lang: SelectorLanguage
  measure: QualityMeasure
  base: Description
  table: Record<string, unknown>
  results: readonly SearchVisit<Description>[]
  edited: Description
  onChange: (d: Description) => void
}) {
  const names = lang.attributes.map((a) => a.name)
  const [attribute, setAttribute] = useState(names[0])
  const attr = lang.attributes.find((a) => a.name === attribute)!
  const ops = attr.kind === 'numeric' ? NUMERIC_OPS : NOMINAL_OPS
  const [op, setOp] = useState<SelectorOp>(ops[0])
  const opNow = ops.includes(op) ? op : ops[0]
  const values = useMemo(
    () =>
      attr.kind === 'numeric'
        ? offeredCuts(lang, table[attribute] as number[], attribute).map((c) => String(c))
        : attr.levels.map((l) => String(l)),
    [attr, lang, table, attribute],
  )
  const [value, setValue] = useState(values[0])
  const valueNow = values.includes(value) ? value : values[0]
  const candidate: Selector = {
    attribute,
    op: opNow,
    value: attr.kind === 'numeric' ? Number(valueNow) : (attr.levels.find((l) => String(l) === valueNow) ?? valueNow),
  }
  const can = valueNow !== undefined && compatibleSelector(edited, candidate)
  const cover = lang.cover(edited)
  const q = measure.quality(cover)
  const q0 = measure.quality(lang.cover(base))
  const rank = results.filter((r) => r.quality > q).length + 1
  const fmt = (s: Selector) =>
    `${s.attribute} ${s.op} ${typeof s.value === 'number' ? Number(s.value.toPrecision(3)) : s.value}`
  return (
    <div className="flex h-full flex-col gap-2 overflow-auto text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        {edited.length === 0 && <span className="text-muted-foreground">everything (no selectors)</span>}
        {edited.map((s) => (
          <span
            key={selectorKey(s)}
            className="inline-flex items-center gap-1 rounded-md border bg-muted/40 py-0.5 pr-0.5 pl-2 font-mono"
          >
            {fmt(s)}
            <button
              type="button"
              aria-label={`remove ${fmt(s)}`}
              className="rounded p-0.5 text-muted-foreground hover:text-foreground"
              onClick={() => onChange(edited.filter((t) => selectorKey(t) !== selectorKey(s)))}
            >
              <X className="size-3" />
            </button>
          </span>
        ))}
        <Button
          variant="ghost"
          size="xs"
          disabled={descriptionKey(edited) === descriptionKey(base)}
          onClick={() => onChange(base)}
        >
          <RotateCcw /> reset to pinned
        </Button>
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <Select label="attribute" value={attribute} onChange={setAttribute} options={names} className="min-w-28" />
        <Select
          label="test"
          value={opNow}
          onChange={(v) => setOp(v as SelectorOp)}
          options={ops}
          className="min-w-16"
        />
        <Select label="value" value={valueNow} onChange={setValue} options={values} className="min-w-24" />
        <Button
          variant="outline"
          size="sm"
          disabled={!can}
          onClick={() => onChange(lang.canonical([...edited, candidate]))}
          title={can ? undefined : 'the description already tests this attribute this way'}
        >
          <Plus /> add
        </Button>
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        <Readout label="quality" value={f3(q)} />
        <Readout label="vs pinned" value={`${q - q0 >= 0 ? '+' : ''}${f3(q - q0)}`} />
        <Readout label="rows" value={bitsetCount(cover)} />
        <Readout label="would rank" value={rank <= results.length ? `#${rank}` : `below the top ${results.length}`} />
      </div>
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center p-4 text-center text-sm text-muted-foreground">
      {children}
    </div>
  )
}
