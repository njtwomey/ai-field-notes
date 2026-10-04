/**
 * Exceptional model mining: the same refinement search, but a subgroup scores by how much a model fitted on its rows
 * differs from the model fitted on the rest. The reader picks the model class (correlation, regression slope, logistic
 * classifier, association) and its measure, plays the search, pins a subgroup, sees its fitted model against the
 * complement's, and refines it by hand. The fits come from `aifn/learning/subgroups`' model classes.
 */
import type { AssociationFit, BivariateFit, Description, LogisticFit, ModelFits } from 'aifn/learning/subgroups'
import { bitsetHas, formatDescription, selectorLanguage } from 'aifn/learning/subgroups'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { MathText } from '@lab/diagram'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { choice, pinField, row, useFigureState, usePinned } from '@lab/state'
import { Bars, Curve, Plot, Points, Readout, useAxis } from '@lab/viz'
import { Empty, PlantedReadouts, ProgressPlot, RefinePanel, ResultsTable, SearchTreePlot } from './panels'
import { f3, useRun } from './run'
import {
  emmData,
  emmMeasure,
  LANGUAGE_ROW,
  SEARCH_ROW,
  EMM_MEASURES,
  EMM_MODELS,
  EMM_TARGETS,
  type AnyModel,
  type EmmModelKey,
} from './problems'

const SCHEMA = {
  model: row('1 · model class and measure', {
    model: choice(EMM_MODELS, 'correlation', { label: 'model class' }),
    correlation: choice(EMM_MEASURES.correlation, 'fisher-z', {
      label: 'measure',
      when: (v) => v.model === 'correlation',
    }),
    regression: choice(EMM_MEASURES.regression, 'slope-difference', {
      label: 'measure',
      when: (v) => v.model === 'regression',
    }),
    classification: choice(EMM_MEASURES.classification, 'wald', {
      label: 'measure',
      when: (v) => v.model === 'classification',
    }),
    association: choice(EMM_MEASURES.association, 'entropy', {
      label: 'measure',
      when: (v) => v.model === 'association',
    }),
  }),
  language: row('2 · description language', { ...LANGUAGE_ROW, bins: { ...LANGUAGE_ROW.bins, initial: 2 } }),
  search: row('3 · search', {
    ...SEARCH_ROW,
    strategy: { ...SEARCH_ROW.strategy, initial: 'beam' as const },
    beamWidth: { ...SEARCH_ROW.beamWidth, initial: 6 },
    depth: { ...SEARCH_ROW.depth, initial: 2 },
  }),
  pin: pinField(),
}

const TEX: Record<EmmModelKey, string> = {
  correlation: String.raw`\varphi = \frac{|\operatorname{atanh}\rho_G - \operatorname{atanh}\rho_{\bar G}|}{\sqrt{1/(n-3) + 1/(N-n-3)}}`,
  regression: String.raw`\varphi = \frac{|b_G - b_{\bar G}|}{\sqrt{s_G^2/S_{xx,G} + s_{\bar G}^2/S_{xx,\bar G}}}`,
  classification: String.raw`\varphi = \frac{|b_G - b_{\bar G}|}{\sqrt{\operatorname{var} b_G + \operatorname{var} b_{\bar G}}}`,
  association: String.raw`\varphi = H(n/N)\,|Q_G - Q_{\bar G}|,\ \ Q = \frac{n_{11}n_{00} - n_{10}n_{01}}{n_{11}n_{00} + n_{10}n_{01}}`,
}

/** Points of the 2-sd ellipse of a bivariate fit. */
function ellipse(f: BivariateFit) {
  const a = f.varX
  const b = f.cov
  const c = f.varY
  const tr = (a + c) / 2
  const det = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b))
  const l1 = tr + det
  const l2 = Math.max(0, tr - det)
  const theta = Math.atan2(l1 - a, b || 1e-12)
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i <= 64; i++) {
    const s = (2 * Math.PI * i) / 64
    const u = 2 * Math.sqrt(l1) * Math.cos(s)
    const w = 2 * Math.sqrt(l2) * Math.sin(s)
    xs.push(f.meanX + u * Math.cos(theta) - w * Math.sin(theta))
    ys.push(f.meanY + u * Math.sin(theta) + w * Math.cos(theta))
  }
  return { x: xs, y: ys }
}

function ModelPanel({
  model,
  measure,
  description,
}: {
  model: EmmModelKey
  measure: AnyModel
  description: Description
}) {
  const data = emmData()
  const lang = useMemo(() => selectorLanguage(data.table, { exclude: data.targets }), [data])
  const cover = lang.cover(description)
  const [tx, ty] = EMM_TARGETS[model]
  const xs = data.table[tx] as number[]
  const ys = data.table[ty] as number[]
  const fits = measure.fit(cover) as ModelFits<unknown>
  const x = useAxis({ label: tx, hold: 'initial', key: model })
  const y = useAxis({ label: ty, hold: 'initial', key: model })
  const cats = useAxis({ label: '', categories: ['inside', 'rest', 'all'] })
  const qAxis = useAxis({ label: 'Yule’s Q', range: [-1, 1] })
  const split = (inside: boolean, jitter = 0) => {
    const idx = xs.map((_, i) => i).filter((i) => bitsetHas(cover, i) === inside)
    return {
      x: idx.map((i) => xs[i] + (jitter ? jitter * ((((i * 0.754877666) % 1) - 0.5) * 2) : 0)),
      y: idx.map((i) => ys[i] + (jitter ? jitter * ((((i * 0.569840291) % 1) - 0.5) * 2) : 0)),
    }
  }
  if (model === 'association') {
    const f = fits as ModelFits<AssociationFit>
    return (
      <Plot x={cats} y={qAxis} legend={false} ariaLabel="Association inside and outside">
        <Bars name="subgroup" x={[0]} y={[f.inside.q]} slot={0} />
        <Bars name="complement" x={[1]} y={[f.outside.q]} muted />
        <Bars name="all" x={[2]} y={[f.all.q]} emphasis />
      </Plot>
    )
  }
  const lo = Math.min(...xs)
  const hi = Math.max(...xs)
  const grid = Array.from({ length: 61 }, (_, i) => lo + ((hi - lo) * i) / 60)
  if (model === 'classification') {
    const f = fits as ModelFits<LogisticFit>
    const curve = (g: LogisticFit) => grid.map((v) => 1 / (1 + Math.exp(-(g.intercept + g.slope * v))))
    return (
      <Plot x={x} y={y} ariaLabel="Logistic fits inside and outside">
        <Points name="complement" {...split(false, 0.04)} muted thin dense />
        <Points name="subgroup" {...split(true, 0.04)} slot={0} />
        <Curve name="P(label = 1 | x), subgroup" x={grid} y={curve(f.inside)} slot={0} />
        <Curve name="P(label = 1 | x), complement" x={grid} y={curve(f.outside)} emphasis dashed />
      </Plot>
    )
  }
  const f = fits as ModelFits<BivariateFit>
  const line = (g: BivariateFit) => grid.map((v) => g.intercept + g.slope * v)
  return (
    <Plot x={x} y={y} ariaLabel="Fits inside and outside">
      <Points name="complement" {...split(false)} muted thin dense />
      <Points name="subgroup" {...split(true)} slot={0} />
      {model === 'correlation' ? (
        <>
          <Curve name="2-sd ellipse, subgroup" {...ellipse(f.inside)} slot={0} />
          <Curve name="2-sd ellipse, complement" {...ellipse(f.outside)} emphasis dashed />
        </>
      ) : (
        <>
          <Curve name="fit, subgroup" x={grid} y={line(f.inside)} slot={0} />
          <Curve name="fit, complement" x={grid} y={line(f.outside)} emphasis dashed />
          <Curve name="fit, all rows" x={grid} y={line(f.all)} muted />
        </>
      )}
    </Plot>
  )
}

function ModelReadouts({
  model,
  measure,
  description,
}: {
  model: EmmModelKey
  measure: AnyModel
  description: Description
}) {
  const data = emmData()
  const lang = useMemo(() => selectorLanguage(data.table, { exclude: data.targets }), [data])
  const fits = measure.fit(lang.cover(description))
  if (model === 'association') {
    const f = fits as ModelFits<AssociationFit>
    return (
      <>
        <Readout label="Q subgroup" value={f3(f.inside.q)} />
        <Readout label="Q complement" value={f3(f.outside.q)} />
      </>
    )
  }
  if (model === 'classification') {
    const f = fits as ModelFits<LogisticFit>
    return (
      <>
        <Readout label="slope subgroup" value={f3(f.inside.slope)} />
        <Readout label="slope complement" value={f3(f.outside.slope)} />
      </>
    )
  }
  const f = fits as ModelFits<BivariateFit>
  return (
    <>
      <Readout label="ρ subgroup" value={f3(f.inside.rho)} />
      <Readout label="ρ complement" value={f3(f.outside.rho)} />
      <Readout label="slope subgroup" value={f3(f.inside.slope)} />
      <Readout label="slope complement" value={f3(f.outside.slope)} />
    </>
  )
}

export function ExceptionalModelFigure() {
  const figure = useFigureState(SCHEMA)
  const model = figure.model.model as EmmModelKey
  const measureKey = figure.model[model] as string
  const { discretisation, bins, negations, minSupport } = figure.language
  const { strategy, beamWidth, depth, k, prune, redundancy } = figure.search
  const data = emmData()
  const planted = data.planted.filter((p) => EMM_TARGETS[model].every((t) => p.targets.includes(t)))
  const lang = useMemo(
    () =>
      selectorLanguage(data.table, {
        exclude: data.targets,
        discretisation: discretisation as 'equal-frequency' | 'equal-width' | 'on-the-fly',
        bins,
        negations,
      }),
    [data, discretisation, bins, negations],
  )
  const measure = useMemo(() => emmMeasure(model, measureKey), [model, measureKey])
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
  const pinnedDescription = pins.pinned !== null ? v[pins.pinned].node : null
  const [edits, setEdits] = useState<{ base: string; d: Description } | null>(null)
  const baseKey = pinnedDescription ? JSON.stringify(pinnedDescription) : ''
  const edited = edits && edits.base === baseKey ? edits.d : pinnedDescription
  const shown =
    edited ?? (pins.focus !== null ? v[pins.focus].node : (state.results[0]?.node ?? planted[0]?.description ?? null))
  const ax = {
    depth: useAxis({ label: 'selectors (depth)', range: [-0.5, depth + 0.5], integer: true, key: depth }),
    q: useAxis({ label: 'quality', range: run.qualityRange, key: run }),
    step: useAxis({ label: 'step', key: run }),
    best: useAxis({ label: 'quality', key: run }),
  }
  const band =
    t === 0
      ? 'step 0: every row is one group; no model class here has an optimistic estimate, so the beam (or the depth limit) bounds the search.'
      : `step ${t}: refined ${state.expanded.length === 1 ? formatDescription(state.expanded[0].node) : `${state.expanded.length} descriptions`}; best ${f3(state.best)}.`
  return (
    <Figure
      title="Exceptional model mining"
      purpose="The same search, but a subgroup is interesting when a model fitted on its rows differs from the model fitted on the rest: a flipped correlation, a different slope, a reversed classifier."
      state={figure}
      defaultSize="XL"
      controls={<Player value={t} onChange={setStep} count={run.steps.length} label="step" format={(p) => String(p)} />}
      equation={
        <div className="font-prose text-[15px] leading-snug">
          <MathText text={`$${TEX[model]}$ · ${band}`} />
        </div>
      }
      readouts={{
        search: (
          <>
            <Readout label="evaluated" value={state.evaluated} />
            <Readout label="best" value={f3(state.best)} />
            <Readout label="k-th best" value={f3(state.threshold)} />
          </>
        ),
        [pinnedDescription ? 'pinned' : 'shown']: shown ? (
          <ModelReadouts model={model} measure={measure} description={shown} />
        ) : null,
        planted: <PlantedReadouts lang={lang} measure={measure} planted={planted} results={state.results} />,
      }}
      caption={
        <>
          The table has four attributes and planted flips: x and y correlate at +0.7 except inside group = b ∧ level ≥ 5
          (−0.7), and the label&apos;s dependence on x and the association of a and b reverse inside flag = yes ∧ region
          = east. Play the search from step 0; the right-hand panel shows the shown subgroup&apos;s fitted model
          (coloured) against its complement&apos;s (dashed ink), with the subgroup&apos;s rows coloured and the rest
          faint. Click a ranked row to pin it (again or Escape to unpin), then add or remove selectors and watch the two
          fits and the quality move. Cook&apos;s distance grows with the rows deleted, so it favours large subgroups.
        </>
      }
    >
      <Dashboard>
        <DashboardRow minHeight={240} ratio={1.3}>
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
          <DashboardCell ratio={0.8}>
            <ProgressPlot run={run} k={t} x={ax.step} y={ax.best} />
          </DashboardCell>
          <DashboardCell ratio={1.2}>
            {shown ? (
              <ModelPanel model={model} measure={measure} description={shown} />
            ) : (
              <Empty>Pin a subgroup.</Empty>
            )}
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={220}>
          <DashboardCell ratio={1.3}>
            <ResultsTable
              lang={lang}
              results={state.results}
              planted={planted}
              focus={pins.focus}
              pinned={pins.pinned}
              onHover={pins.hover}
              onPin={pins.toggle}
              quality={measure.name}
            />
          </DashboardCell>
          <DashboardCell>
            {pinnedDescription && edited ? (
              <RefinePanel
                lang={lang}
                measure={measure}
                base={pinnedDescription}
                table={data.table}
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
