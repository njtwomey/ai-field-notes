/**
 * Global effects (`aifn/learning/explain`): partial dependence against accumulated local effects for a boosted model
 * fitted to two strongly correlated features, and, for an additive model with an interaction, Friedman's H-statistics
 * and the functional ANOVA decomposition, which separates a GAM's main effects from its interaction surface.
 */
import { useMemo, useState } from 'react'
import { correlatedEffects, correlatedEffectsTerm, interactionTask } from 'aifn-methods/data/synthetic'
import { gam, s as smooth, te } from 'aifn-methods/learning/generalised/gam'
import { gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, mean, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { accumulatedLocalEffects, functionalAnova, hStatistic, partialDependence } from 'aifn-compute/learning/explain'
import { StatusText } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, int, row, slider, toggle, useComputed, useFigureState } from 'aifn-render/state'
import { Button } from 'aifn-render/ui/button'
import { Bars, Curve, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { fmt } from './mlp'

const centred = (v: ArrayLike<number>) => {
  const m = Number(mean(fromData(Float64Array.from(v))))
  return Array.from(v, (u) => u - m)
}

function FitRow({
  label,
  fitted,
  stale,
  onFit,
  status,
}: {
  label: string
  fitted: boolean
  stale: boolean
  onFit: () => void
  status: string
}) {
  return (
    <ControlRow label={label}>
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" variant={!fitted || stale ? 'default' : 'outline'} aria-label="Train" onClick={onFit}>
          {fitted ? 'Refit' : 'Train'}
        </Button>
        <StatusText tone={!fitted || stale ? 'attention' : 'muted'}>
          {!fitted ? 'Not fitted yet: press Train.' : stale ? 'Settings changed since the fit: press Refit.' : status}
        </StatusText>
      </div>
    </ControlRow>
  )
}

type PdSetup = { n: number; correlation: number; noise: number; seed: number; stages: number; depth: number }

export function EffectsShowcase() {
  const state = useFigureState({
    data: row('1 · data and model', {
      correlation: slider(0, 0.99, 0.98, { step: 0.01, label: 'correlation ρ of x₁ and x₂' }),
      n: int(500, { ge: 50, le: 5000, suggestions: [300, 500, 2000], label: 'rows n' }),
      noise: slider(0, 0.5, 0.1, { step: 0.01, label: 'noise sd' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
      stages: int(200, { ge: 1, le: 2000, suggestions: [50, 200, 500], label: 'boosting stages' }),
      depth: int(3, { ge: 1, le: 8, suggestions: [2, 3, 5], label: 'tree depth' }),
    }),
    view: row('2 · effect', {
      feature: choice(
        [
          { value: 0, label: 'x₁ (true effect x₁)' },
          { value: 1, label: 'x₂ (true effect x₂²)' },
        ],
        1,
        { label: 'feature' },
      ),
      bins: int(12, { ge: 2, le: 60, suggestions: [6, 12, 30], label: 'ALE bins' }),
      where: toggle(true, 'where each method queries the model'),
    }),
  })
  const current: PdSetup = { ...state.data }
  const [fitted, setFitted] = useState<PdSetup | null>(null)
  const stale = fitted !== null && JSON.stringify(fitted) !== JSON.stringify(current)
  const shown = fitted ?? current
  const data = useMemo(
    () =>
      correlatedEffects(stream(`ale-${shown.seed}`), {
        n: shown.n,
        correlation: shown.correlation,
        noise: shown.noise,
      }),
    [shown.seed, shown.n, shown.correlation, shown.noise],
  )
  const model = useMemo(() => {
    if (!fitted) return null
    return gradientBoosting({ stages: fitted.stages, tree: { maxDepth: fitted.depth }, learningRate: 0.1 }).fit(
      dataset(data.x, data.y!),
    )
  }, [fitted, data])
  const feature = Number(state.view.feature)
  const X = useMemo(() => Float64Array.from(toFlat(data.x)), [data])
  const n = X.length / 2
  const grid = useMemo(() => {
    const col = Array.from({ length: n }, (_, i) => X[2 * i + feature])
    return Array.from(toFlat(linspace(Math.min(...col), Math.max(...col), 41)))
  }, [X, n, feature])
  const effects = useComputed(
    () => {
      if (!model) return null
      const f = (Z: Tensor) => toFlat(model.forward(Z) as Tensor)
      const pd = partialDependence(f, data.x, feature, { grid })
      const ale = accumulatedLocalEffects(f, data.x, feature, { bins: state.view.bins })
      return { pd: centred(pd.average), ale: { x: Array.from(ale.edges), y: centred(ale.effect) } }
    },
    [model, data, feature, grid, state.view.bins],
    { mode: 'release' },
  )
  const truth = useMemo(() => centred(grid.map((v) => correlatedEffectsTerm(feature, v))), [grid, feature])
  const e = effects.value
  // Where the methods evaluate the model: PD pairs every grid value with every row; ALE moves rows to their bin's edges.
  const queries = useMemo(() => {
    const rows = Array.from({ length: Math.min(n, 80) }, (_, k) => Math.floor((k * n) / Math.min(n, 80)))
    const other = 1 - feature
    const pd = grid
      .filter((_, k) => k % 4 === 0)
      .flatMap((g) => rows.map((i) => (feature === 0 ? [g, X[2 * i + other]] : [X[2 * i + other], g])))
    return { pd }
  }, [grid, X, n, feature])
  const ax = useAxis({ label: 'x₁', range: [-1.1, 1.1], nice: false })
  const ay = useAxis({ label: 'x₂', range: [-1.1, 1.1], nice: false, equal: ax })
  const gAxis = useAxis({ label: feature === 0 ? 'x₁' : 'x₂', range: [-1.1, 1.1], nice: false })
  const vAxis = useAxis({
    label: 'centred effect on the prediction',
    hold: 'union',
    key: `${feature}-${JSON.stringify(fitted)}`,
  })
  const err = (curve: number[], xs: number[]) => {
    let s = 0
    curve.forEach((v, k) => (s += (v - centred(xs.map((u) => correlatedEffectsTerm(feature, u)))[k]) ** 2))
    return Math.sqrt(s / curve.length)
  }
  return (
    <>
      <Figure
        title="Partial dependence extrapolates; ALE stays on the data"
        purpose="Partial dependence averages the model over every row with the feature set to each value, so with correlated features it queries combinations the model never saw; accumulated local effects average only small moves within each bin of the data, so they recover the true effect where partial dependence is pulled off it."
        state={state}
        defaultSize="XL"
        controls={
          <FitRow
            label="3 · fit"
            fitted={fitted !== null}
            stale={stale}
            onFit={() => setFitted({ ...current })}
            status={`gradient boosting: ${shown.stages} trees of depth ${shown.depth}`}
          />
        }
        readouts={{
          'root mean square distance from the true effect': (
            <>
              <Readout label="partial dependence" value={e ? fmt(err(e.pd, grid)) : '—'} />
              <Readout label="ALE" value={e ? fmt(err(e.ale.y, e.ale.x)) : '—'} />
            </>
          ),
        }}
        caption={`Data: aifn correlatedEffects (seeded): ${n} rows with x₁ ~ U(−1, 1) and x₂ = ρx₁ + √(1 − ρ²) z, z ~ U(−1, 1), and y = x₁ + x₂² + noise, so the true effect of x₁ is x₁ and of x₂ is x₂². Press Train to fit gradient-boosted trees by squared loss. Left: the rows (colour) and, when shown, the queries partial dependence makes (grey): every grid value paired with every row's other feature, far off the diagonal band the data occupy when ρ is near 1; ALE's queries stay at the rows, moved only to their bin's edges. Right: the true effect (ink), partial dependence and ALE (${state.view.bins} quantile bins), each centred to mean zero over its points. Drag ρ down to 0 and the two agree; raise it and partial dependence bends where the trees extrapolate.`}
      >
        <Plots cols={2} widths={[42, 58]}>
          <Plot x={ax} y={ay}>
            {state.view.where && (
              <Points
                name="PD queries"
                x={queries.pd.map((p) => p[0])}
                y={queries.pd.map((p) => p[1])}
                muted
                thin
                size={3}
              />
            )}
            <Points
              name="rows"
              x={Array.from({ length: n }, (_, i) => X[2 * i])}
              y={Array.from({ length: n }, (_, i) => X[2 * i + 1])}
              slot={0}
            />
          </Plot>
          <Plot x={gAxis} y={vAxis}>
            <Curve name="true effect" x={grid} y={truth} emphasis width={2.5} />
            {e && <Curve name="partial dependence" x={grid} y={e.pd} slot={2} stale={effects.stale} />}
            {e && <Curve name="ALE" x={e.ale.x} y={e.ale.y} slot={3} showPoints stale={effects.stale} />}
          </Plot>
        </Plots>
      </Figure>
      <InteractionFigure />
    </>
  )
}

// ── Interactions and functional ANOVA ────────────────────────────────────────────────────────────────────────────────

type AnovaSetup = { interaction: number; n: number; seed: number; model: string }
const MID = Array.from({ length: 15 }, (_, k) => -1 + (2 * (k + 0.5)) / 15)
const PAIRS = ['x₁x₂', 'x₁x₃', 'x₂x₃']

function InteractionFigure() {
  const state = useFigureState({
    data: row('1 · data and model', {
      interaction: slider(-2, 2, 1, { step: 0.1, label: 'interaction β (x₁x₂)' }),
      n: int(500, { ge: 50, le: 5000, suggestions: [300, 500, 2000], label: 'rows n' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
      model: choice(
        [
          { value: 'gam-te', label: 'GAM: s(x₁) + s(x₂) + s(x₃) + te(x₁, x₂)' },
          { value: 'gam', label: 'GAM: s(x₁) + s(x₂) + s(x₃)' },
          { value: 'boosting', label: 'gradient boosting (depth 3)' },
        ],
        'gam-te',
        { label: 'model' },
      ),
    }),
    view: row('2 · main effect', {
      feature: choice(
        [
          { value: 0, label: 'x₁ (true sin πx₁)' },
          { value: 1, label: 'x₂ (true x₂²)' },
          { value: 2, label: 'x₃ (true ½x₃)' },
        ],
        0,
        { label: 'feature' },
      ),
    }),
  })
  const current: AnovaSetup = { ...state.data, model: state.data.model as string }
  const [fitted, setFitted] = useState<AnovaSetup | null>(null)
  const stale = fitted !== null && JSON.stringify(fitted) !== JSON.stringify(current)
  const shown = fitted ?? current
  const data = useMemo(
    () => interactionTask(stream(`anova-${shown.seed}`), { n: shown.n, interaction: shown.interaction }),
    [shown.seed, shown.n, shown.interaction],
  )
  const fit = useMemo(() => {
    if (!fitted) return null
    const d = dataset(data.x, data.y!)
    if (fitted.model === 'boosting') {
      const m = gradientBoosting({ stages: 300, tree: { maxDepth: 3 } }).fit(d)
      return { f: (Z: Tensor) => toFlat(m.forward(Z) as Tensor), gam: null }
    }
    const terms = [smooth(0), smooth(1), smooth(2), ...(fitted.model === 'gam-te' ? [te(0, 1)] : [])]
    const m = gam({ terms }).fit(d)
    return { f: (Z: Tensor) => toFlat(m.forward(Z) as Tensor), gam: m }
  }, [fitted, data])
  const result = useComputed(
    () => {
      if (!fit) return null
      const anova = functionalAnova(fit.f, [MID, MID, MID])
      const sample = fromData(Float64Array.from(toFlat(data.x).slice(0, 3 * 60)), [60, 3])
      const h = hStatistic(fit.f, sample)
      return { anova, h }
    },
    [fit, data],
    { mode: 'release' },
  )
  const r = result.value
  const feature = Number(state.view.feature)
  const truth = centred(MID.map((v) => (feature === 0 ? Math.sin(Math.PI * v) : feature === 1 ? v * v : 0.5 * v)))
  const termCurve = useMemo(() => {
    if (!fit?.gam) return null
    const p = fit.gam.partial(feature, fromData(Float64Array.from(MID)))
    return centred(toFlat(p.fit))
  }, [fit, feature])
  const total = r ? r.anova.variance : 1
  const shares = r
    ? [
        ...Array.from(r.anova.mainVariance),
        ...Array.from(r.anova.pairVariance),
        Math.max(0, r.anova.higherVariance),
      ].map((v) => v / total)
    : null
  const shareNames = ['x₁', 'x₂', 'x₃', ...PAIRS, 'higher']
  const gAxis = useAxis({ label: ['x₁', 'x₂', 'x₃'][feature], range: [-1, 1], nice: false })
  const vAxis = useAxis({ label: 'centred main effect', hold: 'union', key: `${feature}-${JSON.stringify(fitted)}` })
  const sAxis = useAxis({ label: 'term', range: [0.4, 7.6], integer: true, format: (v) => shareNames[v - 1] ?? '' })
  const shAxis = useAxis({ label: 'share of the variance of f', range: [0, 1] })
  const hAxis = useAxis({
    label: 'H-statistic',
    range: [0.4, 6.6],
    integer: true,
    format: (v) => ['H x₁', 'H x₂', 'H x₃', ...PAIRS][v - 1] ?? '',
  })
  const hvAxis = useAxis({ label: 'H', range: [0, 1] })
  const hValues = r ? [...Array.from(r.h.overall), r.h.pairwise[1], r.h.pairwise[2], r.h.pairwise[5]] : null
  return (
    <Figure
      title="Interactions: H-statistics and the functional ANOVA"
      purpose="The functional ANOVA writes the model as a constant, main effects and pairwise surfaces, each averaging to zero over each of its features, so their variances divide the model's variance without overlap; Friedman's H measures the share of a feature's or a pair's effect that is interaction. Together they show what a GAM's interaction term carries, and what an additive GAM cannot."
      state={state}
      defaultSize="XL"
      controls={
        <FitRow
          label="3 · fit"
          fitted={fitted !== null}
          stale={stale}
          onFit={() => setFitted({ ...current })}
          status={`fitted ${shown.model === 'boosting' ? 'gradient boosting' : 'a GAM by penalised least squares'}`}
        />
      }
      readouts={{
        'variance shares': (
          <>
            {shareNames.map((name, k) => (
              <Readout key={name} label={name} value={shares ? fmt(shares[k], 2) : '—'} />
            ))}
          </>
        ),
      }}
      caption={`Data: aifn interactionTask (seeded): ${data.x.shape[0]} rows of three independent U(−1, 1) features, y = sin(πx₁) + x₂² + ½x₃ + βx₁x₂ + noise. Press Train to fit the chosen model. Left: the functional ANOVA main effect of the chosen feature (computed over the 15³ grid of midpoints on [−1, 1], the product measure of the features), the true effect (ink) and, for a GAM, its own smooth term s(·), each centred. Middle: each term's share of the model's variance (Sobol indices); with β ≠ 0 the x₁x₂ share appears only for models that can represent it. Right: Friedman's H on 60 rows, overall per feature and for each pair. Set β to 0 and every interaction measure falls to about zero.`}
    >
      <Plots cols={3} widths={[36, 32, 32]}>
        <Plot x={gAxis} y={vAxis}>
          <Curve name="true effect" x={MID} y={truth} emphasis width={2.5} />
          {r && (
            <Curve
              name="functional ANOVA"
              x={MID}
              y={centred(r.anova.main[feature])}
              slot={2}
              showPoints
              stale={result.stale}
            />
          )}
          {termCurve && <Curve name="GAM term s(·)" x={MID} y={termCurve} slot={3} dashed />}
        </Plot>
        <Plot x={sAxis} y={shAxis}>
          {shares && (
            <Bars name="variance share" x={shares.map((_, k) => k + 1)} y={shares} slot={2} stale={result.stale} />
          )}
        </Plot>
        <Plot x={hAxis} y={hvAxis}>
          {hValues && <Bars name="H" x={hValues.map((_, k) => k + 1)} y={hValues} slot={4} stale={result.stale} />}
        </Plot>
      </Plots>
    </Figure>
  )
}
