import { useMemo, useState } from 'react'
import { additiveData } from 'aifn-methods/data/synthetic'
import type { AdditiveTruth } from 'aifn-methods/data'
import {
  explainableBoostingMachine,
  gam,
  gamBackfitting,
  gamModel,
  gamProblem,
  s,
  type SmoothingMethod,
} from 'aifn-methods/learning/generalised/gam'
import { dataset } from 'aifn-compute/learning/estimators'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { Player } from 'aifn-render/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, useFigureState } from 'aifn-render/state'
import { Area, Curve, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const N = 300
const GRID = linspace(0, 1, 101)
const GRID_X = toFlat(GRID)

/** Three features uniform on [0, 1] from aifn's additive generator: a periodic, a parabolic and a linear effect. */
function additive(seed: string, noise = 0.4) {
  const d = additiveData(stream(seed), { n: N, noise, shapes: ['periodic', 'smooth', 'linear'] })
  const rows = toRows(d.x as Tensor) as number[][]
  const truth = d.meta!.truth as AdditiveTruth
  const columns = [0, 1, 2].map((j) =>
    fromData(
      Float64Array.from(rows, (r) => r[j]),
      [N],
    ),
  )
  return {
    rows,
    x: d.x as Tensor,
    y: d.y as Tensor,
    // Each true effect centred on the data, as the fitted smooths are.
    centred: [0, 1, 2].map((j) => toFlat(truth.partial(j, GRID, columns[j]))),
  }
}

/** A Gaussian GAM: each partial effect with its Bayesian band, partial residuals and posterior draws. */
export function PartialEffects() {
  const state = useFigureState({
    model: row('1 · model', {
      method: choice(['reml', 'gcv', 'fixed'] as SmoothingMethod[], 'reml', { label: 'smoothing parameters by' }),
      k: slider(5, 25, 12, { label: 'basis size k', step: 1 }),
    }),
    reveal: row('2 · reveal', {
      residuals: toggle(true, 'partial residuals'),
      draws: toggle(false, 'posterior draws'),
      count: choice([5, 20, 50], 20, { label: 'draws', when: (v) => v.draws === true }),
    }),
  })
  const method = state.model.method as SmoothingMethod
  const { k } = state.model
  const { residuals: showResiduals, draws: showDraws, count } = state.reveal
  const data = useMemo(() => additive('gam-partial'), [])
  const model = useMemo(
    () => gam({ terms: [s(0, { k }), s(1, { k }), s(2, { k })], method }).fit(dataset(data.x, data.y)),
    [data, k, method],
  )
  const panels = useMemo(
    () =>
      [0, 1, 2].map((j) => {
        const p = model.partial(j, GRID)
        const f = toFlat(p.fit)
        const se = toFlat(p.se)
        return {
          f,
          upper: f.map((v, i) => v + 2 * se[i]),
          lower: f.map((v, i) => v - 2 * se[i]),
          xs: data.rows.map((r) => r[j]),
          residuals: toFlat(model.partialResiduals(j)),
        }
      }),
    [model, data],
  )
  const draws = useMemo(
    () =>
      showDraws
        ? [0, 1, 2].map((j) => toRows(model.partialDraws(stream(`gam-draws-${j}`), j, GRID, count)) as number[][])
        : null,
    [model, showDraws, count],
  )
  const y = useAxis({ label: 'partial effect', hold: 'union' })
  const x1 = useAxis({ label: 'x1', range: [0, 1] })
  const x2 = useAxis({ label: 'x2', range: [0, 1] })
  const x3 = useAxis({ label: 'x3', range: [0, 1] })
  const xs = [x1, x2, x3]
  return (
    <Figure
      title="Partial effects with posterior bands"
      purpose="An additive model splits the fit into one curve per feature, each with a Bayesian ± 2 se band; REML or GCV picks a separate smoothness for each."
      state={state}
      readouts={{
        'per term': (
          <>
            {model.labels.map((l, j) => (
              <Readout
                key={l}
                label={`${l}: EDF, λ`}
                value={`${formatValue(model.termEdf[j])}, ${formatValue(model.lambdas[j])}`}
              />
            ))}
          </>
        ),
        model: (
          <>
            <Readout label="σ̂" value={formatValue(Math.sqrt(model.dispersion))} />
            <Readout label="criterion evaluations" value={model.smoothingScore.evaluations} />
          </>
        ),
      }}
      caption="Data from aifn's additive generator: a periodic effect of x₁ (sin 2πx + ½ cos 4πx), a parabola in x₂ and a line in x₃, plus Gaussian noise with sd 0.4. REML and GCV give each smooth its own λ: the line gets a huge λ and about one EDF, the periodic term several. With 'fixed' every λ is 1. Blue: the fit and its ± 2 se band (V_β = (XᵀX + S_λ)⁻¹σ̂²); ink dashed: the true effect, centred on the data as the fit is; thin orange: posterior draws. The draws show the band's meaning: plausible curves, not a pointwise envelope."
    >
      <Plots cols={3}>
        {panels.map((p, j) => (
          <Plot key={j} x={xs[j]} y={y} legend={false}>
            {showResiduals && <Points name="partial residuals" x={p.xs} y={p.residuals} muted thin />}
            <Area name="± 2 se" x={GRID_X} y={p.upper} base={p.lower} slot={0} opacity={0.18} line={false} />
            {draws?.[j].map((d, r) => (
              <Curve key={r} name="draws" x={GRID_X} y={d} thin slot={1} />
            ))}
            <Curve name="true effect (centred)" x={GRID_X} y={data.centred[j]} emphasis dashed />
            <Curve name={`f${j + 1}`} x={GRID_X} y={p.f} slot={0} />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}

/** Backfitting, sweep by sweep. */
export function BackfittingSweeps() {
  const [sweep, setSweep] = useState(0)
  const data = useMemo(() => additive('gam-backfit', 0.3), [])
  const problem = useMemo(
    () => gamProblem({ terms: [s(0), s(1), s(2)], method: 'fixed', lambda: 1 }, { x: data.x, y: data.y }),
    [data],
  )
  const run = useMemo(() => trace(gamBackfitting(problem), undefined, 30), [problem])
  const k = Math.min(sweep, run.steps.length - 1)
  const state = run.steps[k]
  // Each term after this sweep: the model at the sweep's coefficients, on a grid.
  const curves = useMemo(() => {
    const model = gamModel(problem, toFlat(state.coefficients))
    return [0, 1, 2].map((j) => toFlat(model.partial(j, GRID).fit))
  }, [problem, state])
  const best = problem.evaluate(problem.optimum.beta).penalisedDeviance
  const gap = useMemo(
    () => ({ x: Array.from(run.index), y: run.steps.map((st) => Math.max(st.penalisedDeviance - best, 1e-14)) }),
    [run, best],
  )
  const now = useMemo(() => ({ x: [k], y: [gap.y[k]] }), [k, gap])
  const y = useAxis({ label: 'partial effect', range: [-2, 1.5] })
  const x1 = useAxis({ label: 'x1', range: [0, 1] })
  const x2 = useAxis({ label: 'x2', range: [0, 1] })
  const x3 = useAxis({ label: 'x3', range: [0, 1] })
  const xs = [x1, x2, x3]
  const sweepAxis = useAxis({ label: 'sweep', hold: 'initial' })
  const gapAxis = useAxis({ label: 'gap to optimum', log: true, hold: 'initial' })
  return (
    <Figure
      title="Backfitting sweep by sweep"
      defaultSize="L"
      purpose="Backfitting smooths each feature's partial residual in turn, holding the other curves fixed; the sweeps converge to the penalised least-squares fit."
      controls={
        <ControlRow label="sweeps">
          <Player value={k} onChange={setSweep} count={run.steps.length} format={(i) => `sweep ${i}`} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="intercept" value={formatValue(state.inner.intercept)} />
          <Readout
            label="largest change in this sweep"
            value={Number.isFinite(state.inner.change) ? formatValue(state.inner.change) : '—'}
          />
          <Readout label="penalised RSS" value={formatValue(state.penalisedDeviance)} />
          <Readout label="converged" value={String(state.converged)} />
        </>
      }
      caption="Blue: each curve after the chosen sweep; ink dashed: the true effect, centred on the data. Sweep 0 starts every curve at zero. The first sweep already recovers most of each shape because the features are independent; later sweeps only trade small amounts between terms, and the penalised RSS falls to the P-IRLS optimum geometrically (bottom, log scale). λ = 1 for every smooth."
    >
      <Dashboard>
        <DashboardRow ratio={2} minHeight={220}>
          <DashboardCell>
            <Plots cols={3}>
              {curves.map((c, j) => (
                <Plot key={j} x={xs[j]} y={y} legend={false}>
                  <Curve name="true effect (centred)" x={GRID_X} y={data.centred[j]} emphasis dashed />
                  <Curve name={`f${j + 1} after sweep ${k}`} x={GRID_X} y={c} slot={0} />
                </Plot>
              ))}
            </Plots>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={150}>
          <DashboardCell>
            <Plot x={sweepAxis} y={gapAxis} legend={false}>
              <Curve name="penalised RSS − optimum" x={gap.x} y={gap.y} slot={0} showPoints />
              <Points name="current" x={now.x} y={now.y} emphasis />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** An explainable boosting machine's step-function shapes as rounds accumulate. */
export function EbmShapes() {
  const figure = useFigureState({
    boosting: row('boosting', { logRounds: slider(0, 3.3, 1, { label: 'log₁₀ rounds' }) }),
  })
  const { logRounds } = figure.boosting
  const data = useMemo(() => additive('gam-ebm', 0.4), [])
  const rounds = Math.round(10 ** logRounds)
  const model = useMemo(
    () => explainableBoostingMachine({ rounds, learningRate: 0.02 }).fit(dataset(data.x, data.y)),
    [data, rounds],
  )
  const shapes = useMemo(() => {
    const edges = toRows(model.edges)
    const sh = toRows(model.shapes)
    return [0, 1, 2].map((j) => ({
      x: sh[j].flatMap((_, b) => [edges[j][b], edges[j][b + 1]]),
      y: sh[j].flatMap((v) => [v, v]),
    }))
  }, [model])
  const loss = toFlat(model.training.series.loss)
  const y = useAxis({ label: 'shape', range: [-1.5, 1.5] })
  const x1 = useAxis({ label: 'x1', range: [0, 1] })
  const x2 = useAxis({ label: 'x2', range: [0, 1] })
  const x3 = useAxis({ label: 'x3', range: [0, 1] })
  const xs = [x1, x2, x3]
  return (
    <Figure
      title="Explainable boosting: shapes from many small trees"
      purpose="An EBM adds one-split trees on binned features, one feature at a time with a small learning rate; the sum per feature is a step-function shape that converges on the true effect."
      state={figure}
      readouts={
        <>
          <Readout label="rounds" value={rounds} />
          <Readout label="training MSE" value={formatValue(loss.at(-1)!)} />
          <Readout label="intercept" value={formatValue(model.intercept)} />
        </>
      }
      caption="Blue: the learned shape; ink dashed: the true effect, centred on the data. After ten rounds each shape is a coarse, shrunken step; after hundreds it traces the true effect with 32 bins per feature. The cyclic order keeps one feature from absorbing effects of another."
    >
      <Plots cols={3}>
        {shapes.map((c, j) => (
          <Plot key={j} x={xs[j]} y={y} legend={false}>
            <Curve name="true effect (centred)" x={GRID_X} y={data.centred[j]} emphasis dashed />
            <Curve name={`shape ${j + 1}`} x={c.x} y={c.y} slot={0} />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}
