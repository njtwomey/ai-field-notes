import { useMemo, useState } from 'react'
import { dataset as toDataset } from 'aifn-compute/learning/estimators'
import { adaBoost, gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { grid2d } from 'aifn-compute/numerics/geometry'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, row, useFigureState } from 'aifn-render/state'
import { Contours, Plot, Points, Raster, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'
import { DATASET_OPTIONS, DATASETS, type DatasetName } from './data'

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x0: rows.map((r) => r[0]), x1: rows.map((r) => r[1]) }
}

/** z[i][j] from a flat [ny · nx] vector. */
const reshape = (v: number[], ny: number, nx: number) =>
  Array.from({ length: ny }, (_, i) => v.slice(i * nx, (i + 1) * nx))

// ── Boosting round by round ──────────────────────────────────────────────────────────────────────────────────────

export function BoostingSpecimen() {
  const figure = useFigureState({
    setup: row('1 · setup', {
      method: choice(
        [
          { value: 'adaboost', label: 'AdaBoost (SAMME)' },
          { value: 'gradient', label: 'gradient boosting (logistic)' },
        ],
        'adaboost',
        { label: 'method' },
      ),
      dataset: choice(
        DATASET_OPTIONS.filter((o) => o.value !== 'blobs'),
        'circles',
        { label: 'dataset' },
      ),
    }),
  })
  const method = figure.setup.method as 'adaboost' | 'gradient'
  const dataset = figure.setup.dataset as DatasetName
  const [round, setRound] = useState(0)
  const data = useMemo(() => DATASETS[dataset].make(), [dataset])
  const rounds = 80
  const ada = useMemo(() => adaBoost({ rounds }).fit(toDataset(data.x, data.y!)), [data])
  const gb = useMemo(
    () =>
      gradientBoosting({ loss: 'logistic', stages: rounds, learningRate: 0.3, tree: { maxDepth: 1 } }).fit(
        toDataset(data.x, data.y!),
      ),
    [data],
  )
  const g = useMemo(() => {
    const c = columns(data.x)
    const pad = (v: number[]) => {
      const lo = Math.min(...v)
      const hi = Math.max(...v)
      return [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)] as [number, number]
    }
    return grid2d(pad(c.x0), pad(c.x1), 70)
  }, [data])
  const K = ada.classes
  const r = Math.max(1, Math.min(round, method === 'adaboost' ? ada.learners.length : rounds))
  const z = useMemo(() => {
    const [ny, nx] = g.shape
    if (method === 'adaboost') {
      // The ensemble's vote share for class 1 (binary) or the winning class's share.
      const v = toRows(ada.votesUpTo(g.points, r))
      return reshape(
        v.map((row) => (K === 2 ? row[1] - row[0] : Math.max(...row))),
        ny,
        nx,
      )
    }
    const f = toFlat(gb.rawUpTo(g.points, r))
    return reshape(K === 2 ? f : f, ny, nx)
  }, [method, ada, gb, g, r, K])
  const state = method === 'adaboost' ? ada.training.steps[Math.min(r, ada.training.steps.length - 1)] : null
  const cols = useMemo(() => columns(data.x), [data])
  const labels = useMemo(() => toFlat(data.y!), [data])
  const heavy = useMemo(() => {
    if (!state) return null
    const idx = toFlat(state.sampleWeights)
      .map((w, i) => [w, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, 15)
      .map((p) => p[1])
    return { x: idx.map((i) => cols.x0[i]), y: idx.map((i) => cols.x1[i]) }
  }, [state, cols])
  const field = useMemo(() => ({ x: toFlat(g.x), y: toFlat(g.y) }), [g])
  const ax0 = useAxis({ label: 'x₀', hold: 'initial', key: dataset })
  const ax1 = useAxis({ label: 'x₁', hold: 'initial', key: dataset, equal: ax0 })
  const loss = method === 'adaboost' ? toFlat(ada.training.series.trainingError) : toFlat(gb.training.series.loss)
  return (
    <Figure
      title="Boosting, round by round"
      purpose="Each round adds one stump fitted to what the ensemble so far gets wrong: reweighted rows for AdaBoost, pseudo-residuals for gradient boosting."
      state={figure}
      defaultSize="L"
      controls={
        <ControlRow label="2 · rounds">
          <Player value={round} onChange={setRound} count={rounds + 1} label="round" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="round" value={r} />
          <Readout
            label={method === 'adaboost' ? 'training error' : 'training log loss'}
            value={formatValue(loss[Math.min(r, loss.length - 1)])}
          />
          {state && <Readout label="this stump's α" value={formatValue(state.alphas[state.alphas.length - 1])} />}
        </>
      }
      caption="The colour is the ensemble's score after the chosen round (vote margin for AdaBoost, log-odds for gradient boosting), with its zero contour (ink) as the boundary. Round 0 shows the first stump. For AdaBoost, the ringed rows carry the most weight going into the next round: they sit where the current boundary is wrong. One stump draws one axis-aligned cut; eighty of them trace the circle."
    >
      <Plot x={ax0} y={ax1}>
        <Raster
          x={field.x}
          y={field.y}
          z={z}
          scale="diverging"
          range={method === 'adaboost' ? [-1, 1] : [-6, 6]}
          valueLabel="score"
        />
        <Contours x={field.x} y={field.y} z={z} levels={[0]} />
        <Points name="rows" x={cols.x0} y={cols.x1} group={labels} groupNames={data.meta.labelNames} />
        {heavy && <Points name="heaviest weights" x={heavy.x} y={heavy.y} emphasis />}
      </Plot>
    </Figure>
  )
}
