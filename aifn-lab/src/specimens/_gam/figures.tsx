import { useMemo, useState } from 'react'
import { backfitting, explainableBoostingMachine, gam, s, type SmoothingMethod } from 'aifn/gam'
import { normals, stream, uniform } from 'aifn/random'
import { linspace, tensor, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import { Player, Select, Slider, Switch } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

const TRUTH = [
  (x: number) => Math.sin(2 * Math.PI * x),
  (x: number) => 4 * (x - 0.5) ** 2 - 1 / 3,
  (x: number) => 0.8 * x - 0.4,
]
const N = 300
const GRID = linspace(0, 1, 101)
const GRID_X = toFlat(GRID)

/** Three features uniform on [0, 1]; y = Σ fⱼ(xⱼ) + noise. */
function additiveData(seed: string, sd = 0.4) {
  const st = stream(seed)
  const rows = toRows(uniform(st.child('x'), 0, 1, { shape: [N, 3] }) as Tensor)
  const e = toFlat(normals(st.child('e'), N, 0, sd))
  const y = rows.map((r, i) => r.reduce((acc, v, j) => acc + TRUTH[j](v), 0) + e[i])
  return { rows, x: tensor(rows), y: tensor(y), yv: y }
}

/** A Gaussian GAM: each partial effect with its Bayesian band, partial residuals and posterior draws. */
export function PartialEffects() {
  const [method, setMethod] = useState<SmoothingMethod>('reml')
  const [k, setK] = useState(12)
  const [showDraws, setShowDraws] = useState(false)
  const [showResiduals, setShowResiduals] = useState(true)
  const data = useMemo(() => additiveData('gam-partial'), [])
  const model = useMemo(
    () => gam({ terms: [s(0, { k }), s(1, { k }), s(2, { k })], method }).fit({ x: data.x, y: data.y }),
    [data, k, method],
  )
  const panels = [0, 1, 2].map((j) => {
    const p = model.partial(j, GRID)
    const f = toFlat(p.fit)
    const se = toFlat(p.se)
    const residuals = toFlat(model.partialResiduals(j))
    const mean = GRID_X.reduce((a, v) => a + TRUTH[j](v), 0) / GRID_X.length
    const draws = showDraws ? toRows(model.partialDraws(stream(`gam-draws-${j}`), j, GRID, 20)) : []
    const series: XYSeries[] = [
      ...(showResiduals
        ? [
            {
              name: 'partial residuals',
              type: 'scatter' as const,
              x: data.rows.map((r) => r[j]),
              y: residuals,
              muted: true,
            },
          ]
        : []),
      ...draws.map((d, r) => ({ name: `draw ${r + 1}`, type: 'line' as const, x: GRID_X, y: d, thin: true, slot: 1 })),
      {
        name: 'true effect (centred)',
        type: 'line',
        x: GRID_X,
        y: GRID_X.map((v) => TRUTH[j](v) - mean),
        slot: 2,
        dashed: true,
      },
      { name: `f${j + 1}`, type: 'line', x: GRID_X, y: f, slot: 0 },
      { name: '± 2 se', type: 'line', x: GRID_X, y: f.map((v, i) => v + 2 * se[i]), slot: 0, dashed: true },
      { name: '± 2 se ', type: 'line', x: GRID_X, y: f.map((v, i) => v - 2 * se[i]), slot: 0, dashed: true },
    ]
    return series
  })
  return (
    <Figure
      title="Partial effects with posterior bands"
      description="An additive model splits the fit into one curve per feature; each comes with a Bayesian ± 2 se band from V_β = (XᵀX + S_λ)⁻¹σ̂²."
      controls={
        <>
          <ControlRow label="Model">
            <Select
              label="smoothing parameters by"
              value={method}
              onChange={setMethod}
              options={['reml', 'gcv', 'fixed']}
            />
            <Slider label="basis size k" value={k} onChange={setK} min={5} max={25} step={1} />
          </ControlRow>
          <ControlRow label="Reveal">
            <Switch label="partial residuals" checked={showResiduals} onChange={setShowResiduals} />
            <Switch label="20 posterior draws" checked={showDraws} onChange={setShowDraws} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          {model.labels.map((l, j) => (
            <Readout
              key={l}
              label={`${l}: EDF, λ`}
              value={`${formatValue(model.termEdf[j])}, ${formatValue(model.lambdas[j])}`}
            />
          ))}
          <Readout label="σ̂" value={formatValue(Math.sqrt(model.dispersion))} />
          <Readout label="criterion evaluations" value={model.score.evaluations} />
        </>
      }
      caption="The truth is sin(2πx₁) + a parabola in x₂ + a line in x₃, each shown centred. REML and GCV give each smooth its own λ: the line gets a huge λ and about one EDF, the sine several. With 'fixed' every λ is 1. Blue: the fit and its ± 2 se band; green dashed: the truth; orange: draws. The draws show the band's meaning: plausible curves, not a pointwise envelope."
    >
      <Subplots rows={1} cols={3} sharey>
        {panels.map((series, j) => (
          <Panel key={j}>
            <XYChart
              series={series}
              xLabel={`x${j + 1}`}
              yLabel={j === 0 ? 'partial effect' : undefined}
              legend={false}
            />
          </Panel>
        ))}
      </Subplots>
    </Figure>
  )
}

/** Backfitting, sweep by sweep. */
export function BackfittingSweeps() {
  const [sweep, setSweep] = useState(1)
  const data = useMemo(() => additiveData('gam-backfit', 0.3), [])
  const run = useMemo(
    () =>
      trace(
        backfitting({ terms: [s(0, { lambda: 1 }), s(1, { lambda: 1 }), s(2, { lambda: 1 })], x: data.x, y: data.y }),
        {},
        30,
      ),
    [data],
  )
  const k = Math.min(sweep, run.steps.length - 1)
  const state = run.steps[k]
  const contributions = toRows(state.contributions)
  const panels = [0, 1, 2].map((j) => {
    const order = data.rows.map((r, i) => [r[j], contributions[j][i]] as const).sort((a, b) => a[0] - b[0])
    const mean = GRID_X.reduce((a, v) => a + TRUTH[j](v), 0) / GRID_X.length
    const series: XYSeries[] = [
      {
        name: 'true effect (centred)',
        type: 'line',
        x: GRID_X,
        y: GRID_X.map((v) => TRUTH[j](v) - mean),
        slot: 2,
        dashed: true,
      },
      {
        name: `f${j + 1} after sweep ${k}`,
        type: 'line',
        x: order.map((o) => o[0]),
        y: order.map((o) => o[1]),
        slot: 0,
      },
    ]
    return series
  })
  const deviance: XYSeries[] = [
    {
      name: 'residual sum of squares',
      type: 'line',
      x: run.index,
      y: run.steps.map((s) => s.deviance),
      slot: 0,
      showPoints: true,
    },
    { name: 'current', type: 'scatter', x: [k], y: [state.deviance], emphasis: true },
  ]
  return (
    <Figure
      title="Backfitting sweep by sweep"
      defaultSize="L"
      description="Backfitting smooths each feature's partial residual in turn, holding the other curves fixed; the sweeps converge to the penalised least-squares fit."
      controls={
        <ControlRow label="Sweeps">
          <Player value={k} onChange={setSweep} count={run.steps.length} format={(i) => `sweep ${i}`} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="intercept" value={formatValue(state.intercept)} />
          <Readout label="largest change in this sweep" value={formatValue(state.change)} />
          <Readout label="converged" value={String(state.converged)} />
        </>
      }
      caption="Blue: each curve after the chosen sweep; green dashed: the centred truth. Sweep 0 starts every curve at zero. The first sweep already recovers most of each shape because the features are independent; later sweeps only trade small amounts between terms. λ = 1 for every smooth."
    >
      <Subplots rows={2} cols={3} heightRatios={[2, 1]}>
        {panels.map((series, j) => (
          <Panel key={j}>
            <XYChart series={series} xLabel={`x${j + 1}`} yRange={[-1.5, 1.5]} legend={false} />
          </Panel>
        ))}
        <Panel>
          <XYChart series={deviance} xLabel="sweep" yLabel="RSS" integerX />
        </Panel>
      </Subplots>
    </Figure>
  )
}

/** An explainable boosting machine's step-function shapes as rounds accumulate. */
export function EbmShapes() {
  const [logRounds, setLogRounds] = useState(2.5)
  const data = useMemo(() => additiveData('gam-ebm', 0.4), [])
  const rounds = Math.round(10 ** logRounds)
  const model = useMemo(
    () => explainableBoostingMachine({ rounds, learningRate: 0.02 }).fit({ x: data.x, y: data.y }),
    [data, rounds],
  )
  const edges = toRows(model.edges)
  const shapes = toRows(model.shapes)
  const panels = [0, 1, 2].map((j) => {
    const xs = shapes[j].flatMap((_, b) => [edges[j][b], edges[j][b + 1]])
    const ys = shapes[j].flatMap((v) => [v, v])
    const mean = GRID_X.reduce((a, v) => a + TRUTH[j](v), 0) / GRID_X.length
    const series: XYSeries[] = [
      {
        name: 'true effect (centred)',
        type: 'line',
        x: GRID_X,
        y: GRID_X.map((v) => TRUTH[j](v) - mean),
        slot: 2,
        dashed: true,
      },
      { name: `shape ${j + 1}`, type: 'line', x: xs, y: ys, slot: 0 },
    ]
    return series
  })
  const loss = toFlat(model.training.series.loss)
  return (
    <Figure
      title="Explainable boosting: shapes from many small trees"
      description="An EBM adds one-split trees on binned features, one feature at a time with a small learning rate; the sum per feature is a step-function shape."
      controls={
        <ControlRow label="Boosting">
          <Slider label="log₁₀ rounds" value={logRounds} onChange={setLogRounds} min={0} max={3.3} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="rounds" value={rounds} />
          <Readout label="training MSE" value={formatValue(loss.at(-1)!)} />
          <Readout label="intercept" value={formatValue(model.intercept)} />
        </>
      }
      caption="Blue: the learned shape; green dashed: the centred truth. After a few rounds each shape is a coarse step; after hundreds it traces the true effect with 32 bins per feature. The cyclic order keeps one feature from absorbing effects of another."
    >
      <Subplots rows={1} cols={3} sharey>
        {panels.map((series, j) => (
          <Panel key={j}>
            <XYChart series={series} xLabel={`x${j + 1}`} yRange={[-1.5, 1.5]} legend={false} />
          </Panel>
        ))}
      </Subplots>
    </Figure>
  )
}
