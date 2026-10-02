/**
 * Subgroup discovery, steered by the reader: pick the table and target, the quality measure, the description language
 * and the search; play the search from step 0 (the frontier, quality against optimistic estimate, what was pruned);
 * pin a subgroup from the ranked list or the tree and see its cover and target distribution; then add or remove
 * selectors by hand and read the quality live. Everything is computed by `aifn/learning/subgroups` and
 * `aifn/optim/search`.
 */
import { bitsetHas, formatDescription, selectorLanguage, type Description } from 'aifn/learning/subgroups'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { MathText } from '@lab/diagram'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { choice, pinField, row, useFigureState, usePinned } from '@lab/state'
import { Bars, Histogram, Plot, Points, Readout, useAxis } from '@lab/viz'
import {
  BINARY_MEASURES,
  LANGUAGE_ROW,
  SEARCH_ROW,
  MEASURE_TEX,
  sdMeasure,
  sdProblem,
  SD_PROBLEMS,
  type MeasureKey,
  type SdProblem,
  type SdProblemKey,
} from './problems'
import { BoundPlot, Empty, PlantedReadouts, ProgressPlot, RefinePanel, ResultsTable, SearchTreePlot } from './panels'
import { f3, useRun } from './run'

const SCHEMA = {
  data: row('1 · table and quality', {
    problem: choice(SD_PROBLEMS, 'planted-outcome', { label: 'table · target' }),
    measure: choice(BINARY_MEASURES, 'wracc', { label: 'quality', when: (v) => v.problem !== 'planted-cost' }),
    direction: choice(['higher', 'lower', 'both'], 'higher', { label: 'target is' }),
  }),
  language: row('2 · description language', LANGUAGE_ROW),
  search: row('3 · search', SEARCH_ROW),
  pin: pinField(),
}

/** A column as plot coordinates: numeric as is, nominal as level index with a fixed jitter. */
function coordinates(values: readonly (string | number)[], levels: readonly (string | number)[] | null) {
  if (!levels) return values as number[]
  return values.map((v, i) => levels.indexOf(v) + 0.7 * (((i * 0.6180339887) % 1) - 0.5))
}

/** The pinned subgroup's rows on two attributes: inside drawn in the target's colours, outside faint. */
function CoverScatter({ problem, description }: { problem: SdProblem; description: Description }) {
  const lang = useMemo(() => selectorLanguage(problem.data.table, { exclude: problem.data.targets }), [problem])
  const numeric = lang.attributes.filter((a) => a.kind === 'numeric')
  const [ax, ay] = numeric.length >= 2 ? numeric : lang.attributes.slice(0, 2)
  const levels = (a: typeof ax) => (a.kind === 'nominal' ? a.levels : null)
  const xs = coordinates(problem.data.table[ax.name] as (string | number)[], levels(ax))
  const ys = coordinates(problem.data.table[ay.name] as (string | number)[], levels(ay))
  const cover = lang.cover(description)
  const fmt = (lv: readonly (string | number)[] | null) =>
    lv ? (v: number) => String(lv[Math.round(v)] ?? '') : undefined
  const x = useAxis({ label: ax.name, hold: 'initial', key: problem, format: fmt(levels(ax)) })
  const y = useAxis({ label: ay.name, hold: 'initial', key: problem, format: fmt(levels(ay)) })
  const split = (inside: boolean) => {
    const idx = xs.map((_, i) => i).filter((i) => bitsetHas(cover, i) === inside)
    return {
      x: idx.map((i) => xs[i]),
      y: idx.map((i) => ys[i]),
      group: problem.kind === 'binary' ? idx.map((i) => (problem.values[i] ? 1 : 0)) : null,
    }
  }
  const names = problem.kind === 'binary' ? [`${problem.target} = 0`, `${problem.target} = 1`] : undefined
  return (
    <Plot x={x} y={y} ariaLabel="The pinned subgroup's cover">
      <Points name="outside" {...split(false)} groupNames={names} thin dense />
      <Points name="inside" {...split(true)} groupNames={names} {...(problem.kind === 'numeric' ? { slot: 0 } : {})} />
    </Plot>
  )
}

/** The target inside the subgroup against the population (binary: rates; numeric: histograms). */
function TargetPanel({ problem, description }: { problem: SdProblem; description: Description }) {
  const lang = useMemo(() => selectorLanguage(problem.data.table, { exclude: problem.data.targets }), [problem])
  const cover = lang.cover(description)
  const inside = problem.values.filter((_, i) => bitsetHas(cover, i))
  const outside = problem.values.filter((_, i) => !bitsetHas(cover, i))
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
  const cats = ['inside', 'rest', 'all']
  const xc = useAxis({ label: '', categories: cats })
  const yr = useAxis({ label: `${problem.target} rate`, range: [0, 1] })
  const xn = useAxis({ label: problem.target, hold: 'initial', key: problem.target })
  const yn = useAxis({ label: 'density', range: [0, undefined] })
  if (problem.kind === 'binary')
    return (
      <Plot x={xc} y={yr} legend={false} ariaLabel="Target rate inside and outside">
        <Bars name="subgroup" x={[0]} y={[mean(inside)]} slot={1} />
        <Bars name="complement" x={[1]} y={[mean(outside)]} muted />
        <Bars name="population" x={[2]} y={[mean(problem.values)]} emphasis />
      </Plot>
    )
  return (
    <Plot x={xn} y={yn} ariaLabel="Target distribution inside and in the population">
      <Histogram name="population" values={problem.values} muted bins={30} />
      <Histogram name="subgroup" values={inside} slot={0} bins={30} />
    </Plot>
  )
}

export function SubgroupDiscoveryFigure() {
  const figure = useFigureState(SCHEMA)
  const { problem: pk, measure: mk, direction } = figure.data
  const { discretisation, bins, negations, minSupport } = figure.language
  const { strategy, beamWidth, depth, k, prune, redundancy } = figure.search
  const problem = useMemo(() => sdProblem(pk as SdProblemKey), [pk])
  const lang = useMemo(
    () =>
      selectorLanguage(problem.data.table, {
        exclude: problem.data.targets,
        discretisation: discretisation as 'equal-frequency' | 'equal-width' | 'on-the-fly',
        bins,
        negations,
      }),
    [problem, discretisation, bins, negations],
  )
  const measureKey: MeasureKey = problem.kind === 'numeric' ? 'meanShift' : (mk as MeasureKey)
  const measure = useMemo(
    () => sdMeasure(problem, measureKey, direction as 'higher' | 'lower' | 'both', minSupport),
    [problem, measureKey, direction, minSupport],
  )
  const run = useRun(lang, measure, {
    strategy: strategy as 'beam',
    beamWidth,
    maxDepth: depth,
    k,
    minSupport,
    prune,
    redundancy: redundancy === 'cover' ? { kind: 'cover', threshold: 0.5 } : { kind: redundancy as 'none' },
  })
  const [step, setStep] = useState(0)
  const last = run.steps.length - 1
  const t = Math.min(step, last)
  const state = run.steps[t]
  const v = run.history.visits
  const pins = usePinned(figure.pin, (p) => figure.set('pin', p), { valid: (id) => id < v.length })
  const focus = pins.focus
  const pinnedDescription = pins.pinned !== null ? v[pins.pinned].node : null
  const [edits, setEdits] = useState<{ base: string; d: Description } | null>(null)
  const baseKey = pinnedDescription ? JSON.stringify(pinnedDescription) : ''
  const edited = edits && edits.base === baseKey ? edits.d : pinnedDescription
  const shown = edited ?? (focus !== null ? v[focus].node : (state.results[0]?.node ?? null))
  const ax = {
    depth: useAxis({ label: 'selectors (depth)', range: [-0.5, depth + 0.5], integer: true, key: depth }),
    q: useAxis({ label: 'quality', range: run.qualityRange, key: run }),
    qx: useAxis({ label: 'quality', range: run.qualityRange, key: run }),
    bound: useAxis({ label: 'optimistic estimate', range: run.boundRange, key: run }),
    step: useAxis({ label: 'step', key: run }),
    best: useAxis({ label: 'quality', key: run }),
  }
  const expanded = state.expanded
  const band =
    t === 0
      ? `step 0: the root (every row) is open. Each step ${strategy === 'beam' ? 'refines the whole beam by one selector' : 'expands one open description'}; ${prune ? 'a description whose optimistic estimate is no better than the k-th best quality is pruned.' : 'nothing is pruned.'}`
      : state.discarded.length
        ? `step ${t}: ${formatDescription(state.discarded[0].node)} is discarded: its estimate ${f3(state.discarded[0].bound)} ≤ the k-th best ${f3(state.threshold)}.`
        : `step ${t}: refined ${expanded.length === 1 ? formatDescription(expanded[0].node) : `${expanded.length} descriptions`}; ${state.generated.length} refinements, ${state.generated.filter((g) => g.fate === 'pruned').length} pruned${strategy === 'beam' ? `, ${state.generated.filter((g) => g.fate === 'dropped').length} dropped from the beam` : ''}; best ${f3(state.best)}, k-th best ${f3(state.threshold)}.`
  return (
    <Figure
      title="Subgroup discovery: finding where the target is unusual"
      purpose="A refinement search over conjunctions of selectors finds the descriptions whose rows are large and unusual in the target; an optimistic estimate prunes branches that cannot enter the top k."
      state={figure}
      defaultSize="XL"
      controls={<Player value={t} onChange={setStep} count={run.steps.length} label="step" format={(p) => String(p)} />}
      equation={
        <div className="font-prose text-[15px] leading-snug">
          <MathText text={`$${MEASURE_TEX[measureKey]}$ · ${band}`} />
        </div>
      }
      readouts={{
        search: (
          <>
            <Readout label="evaluated" value={state.evaluated} />
            <Readout label="expanded" value={state.expansions} />
            <Readout label="pruned" value={state.pruned} />
            <Readout label="best" value={f3(state.best)} />
            <Readout label="k-th best" value={f3(state.threshold)} />
            <Readout label="steps" value={last} />
          </>
        ),
        planted: problem.planted.length ? (
          <PlantedReadouts lang={lang} measure={measure} planted={problem.planted} results={state.results} />
        ) : (
          <Readout label="planted" value="none (real data)" />
        ),
      }}
      caption={
        <>
          Play the search from step 0. Left: every description evaluated so far, by depth and quality (open, expanded,
          pruned; faint ones were dropped from the beam or are at the depth limit); diamonds are the current top k and
          the dashed line is the k-th best quality. Middle: quality against optimistic estimate; a description below the
          dashed line cannot have a refinement better than the k-th best, so branch and bound prunes it. Click a row of
          the ranked list (or a node of the tree) to pin it, again or Escape to unpin; the scatter shows its rows in the
          target&apos;s colours (others faint) and the bars its target against the population. Then add or remove
          selectors below and read the quality and rank change live. The planted readout gives each planted
          subgroup&apos;s own quality and the rank of the result that covers it.
        </>
      }
    >
      <Dashboard>
        <DashboardRow minHeight={250} ratio={1.2}>
          <DashboardCell ratio={1.3}>
            <SearchTreePlot
              run={run}
              k={t}
              pinned={pins.pinned}
              onPin={(id) => (id === null ? pins.clear() : pins.toggle(id))}
              x={ax.depth}
              y={ax.q}
            />
          </DashboardCell>
          <DashboardCell>
            <BoundPlot run={run} k={t} x={ax.qx} y={ax.bound} />
          </DashboardCell>
          <DashboardCell ratio={0.8}>
            <ProgressPlot run={run} k={t} x={ax.step} y={ax.best} />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={230} ratio={1.1}>
          <DashboardCell ratio={1.4}>
            <ResultsTable
              lang={lang}
              results={state.results}
              planted={problem.planted}
              focus={focus}
              pinned={pins.pinned}
              onHover={pins.hover}
              onPin={pins.toggle}
              quality={measure.name}
            />
          </DashboardCell>
          <DashboardCell>
            {shown ? <CoverScatter problem={problem} description={shown} /> : <Empty>Pin a subgroup.</Empty>}
          </DashboardCell>
          <DashboardCell ratio={0.7}>
            {shown ? <TargetPanel problem={problem} description={shown} /> : <Empty>Pin a subgroup.</Empty>}
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={150} ratio={0.45}>
          <DashboardCell>
            {pinnedDescription && edited ? (
              <RefinePanel
                lang={lang}
                measure={measure}
                base={pinnedDescription}
                table={problem.data.table}
                results={run.steps[last].results}
                edited={edited}
                onChange={(d) => setEdits({ base: baseKey, d })}
              />
            ) : (
              <Empty>Pin a subgroup from the ranked list to refine it by hand.</Empty>
            )}
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
