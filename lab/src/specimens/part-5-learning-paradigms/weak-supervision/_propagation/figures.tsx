/**
 * Label propagation on a k-nearest-neighbour graph of two moons: a few labelled points, the harmonic iteration
 * (labelled rows clamped) or label spreading (α), stepped from the labels alone. Graph, affinities, steps and
 * closed-form limits are `aifn/graph`'s; the accuracy is `aifn/learning/metrics`'.
 */
import { useMemo, useState } from 'react'
import { permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { kNearestNeighbourGraph } from 'aifn-compute/graph'
import {
  graphAffinity,
  harmonicLabels,
  labelPropagationSteps,
  labelSpreading,
  labelSpreadingSteps,
  normaliseScores,
} from 'aifn-compute/graph/propagation'
import { accuracy } from 'aifn-compute/learning/metrics'
import { moons } from 'aifn-methods/data/synthetic'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, float, int, row, useFigureState, when } from 'aifn-render/state'
import { formatValue } from '@lab/views'
import { Curve, Plot, Plots, Points, Readout, Segments, useAxis } from 'aifn-render/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const CLASSES = ['upper moon', 'lower moon']

export function LabelPropagationFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      n: int(200, { ge: 20, le: 600, suggestions: [100, 200, 400], label: 'points' }),
      noise: float(0.08, { ge: 0, le: 0.5, suggestions: [0.05, 0.08, 0.15, 0.25], label: 'noise' }),
      perClass: int(1, { ge: 1, le: 50, suggestions: [1, 2, 5], label: 'labels per class' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    graph: row('2 · graph', {
      k: int(7, { ge: 1, le: 50, suggestions: [3, 7, 15], label: 'neighbours k' }),
      sigma: float(0.3, { gt: 0, le: 5, scale: 'log10', suggestions: [0.1, 0.3, 1], label: 'heat-kernel σ' }),
    }),
    method: row('3 · method', {
      method: choice(
        [
          { value: 'propagation', label: 'harmonic propagation (clamped)' },
          { value: 'spreading', label: 'label spreading' },
        ],
        'propagation',
        { label: 'method' },
      ),
      alpha: float(0.9, {
        gt: 0,
        lt: 1,
        suggestions: [0.2, 0.5, 0.9, 0.99],
        label: 'α',
        when: when('method', 'spreading'),
      }),
      steps: int(150, { ge: 1, le: 5000, suggestions: [50, 150, 500], label: 'steps' }),
    }),
  })
  const { n, noise, perClass, seed } = state.data
  const { k, sigma } = state.graph
  const { method, alpha, steps } = state.method
  const data = useMemo(() => moons(stream(`lab/label-propagation/${seed}`), { n, noise }), [n, noise, seed])
  const truth = useMemo(() => Array.from(toFlat(data.y as Tensor)), [data])
  const rows = useMemo(() => toRows(data.x as Tensor) as number[][], [data])
  // The labelled points: the first `perClass` of each class in a seeded random order.
  const given = useMemo(() => {
    const order = Array.from(toFlat(permutation(stream(`lab/label-propagation/labels/${seed}`), truth.length)))
    const picked = [0, 1].flatMap((c) => order.filter((i) => truth[i] === c).slice(0, perClass))
    const labels = truth.map(() => -1)
    for (const i of picked) labels[i] = truth[i]
    return { labels, picked }
  }, [truth, perClass, seed])
  const kk = Math.min(k, n - 1)
  const graph = useMemo(() => kNearestNeighbourGraph(data.x as Tensor, kk), [data, kk])
  const W = useMemo(() => graphAffinity(graph, { kind: 'heat', sigma }), [graph, sigma])
  const unlabelled = useMemo(() => truth.flatMap((_, i) => (given.labels[i] === -1 ? [i] : [])), [truth, given])
  const accuracyOf = (pred: ArrayLike<number>) =>
    accuracy(
      unlabelled.map((i) => truth[i]),
      unlabelled.map((i) => pred[i]),
    )
  const run = useMemo(() => {
    const alg =
      method === 'spreading'
        ? labelSpreadingSteps(W, given.labels, { alpha, classes: 2, tolerance: 0 })
        : labelPropagationSteps(W, given.labels, { classes: 2, tolerance: 0 })
    return trace(alg, undefined, steps, {
      keep: 'all',
      record: {
        change: (s) => (Number.isFinite(s.change) ? s.change : NaN),
        accuracy: (s) => accuracyOf(toFlat(s.labels)),
        reached: (s) => toFlat(s.labels).filter((l) => l >= 0).length,
      },
    })
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- accuracyOf reads truth and unlabelled, both inputs
  }, [W, given, method, alpha, steps, truth, unlabelled])
  const limit = useMemo(() => {
    try {
      return method === 'spreading'
        ? labelSpreading(W, given.labels, { alpha, classes: 2 })
        : harmonicLabels(W, given.labels, { classes: 2 })
    } catch {
      return null
    }
  }, [W, given, method, alpha])
  const [step, setStep] = useState(0)
  const at = Math.min(step, run.steps.length - 1)
  const s = run.steps[at]
  const pred = useMemo(() => Array.from(toFlat(normaliseScores(s.scores).labels)), [s])
  const edges = useMemo(
    () =>
      graph.edges.map((e) => ({
        from: rows[e.from] as unknown as readonly [number, number],
        to: rows[e.to] as unknown as readonly [number, number],
      })),
    [graph, rows],
  )
  const reached = pred.flatMap((l, i) => (l >= 0 ? [i] : []))
  const unreached = pred.flatMap((l, i) => (l < 0 ? [i] : []))
  const series = {
    step: Array.from(run.index),
    accuracy: Array.from(toFlat(run.series.accuracy)),
    change: Array.from(toFlat(run.series.change)),
  }
  const ax = useAxis({ label: 'x₀', range: [-1.6, 2.6] })
  const ay = useAxis({ label: 'x₁', range: [-1.1, 1.6], equal: ax })
  const sx = useAxis({ label: 'step', range: [0, steps], key: steps })
  const sy = useAxis({ label: 'accuracy (unlabelled)', range: [0, 1.02] })
  const limitAccuracy = limit ? accuracyOf(toFlat(limit.labels)) : NaN
  return (
    <Figure
      title="Label propagation on a k-nearest-neighbour graph"
      purpose="A label spreads along short edges: each unlabelled point takes the weighted average of its neighbours' class scores, so one label per moon is enough when the graph keeps the moons apart."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="4 · steps">
          <Player value={at} onChange={setStep} count={run.steps.length} label="step" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="step" value={`${s.t} of ${run.steps.length - 1}`} />
          <Readout label="points reached" value={`${reached.length} of ${n}`} />
          <Readout label="accuracy (unlabelled)" value={f3(series.accuracy[at])} />
          <Readout label="change Σ|ΔF|" value={f3(s.change)} />
          <Readout
            label="closed-form limit accuracy"
            value={limit ? f3(limitAccuracy) : 'singular (a component has no label)'}
          />
          <Readout label="edges" value={graph.edges.length} />
        </>
      }
      caption={
        <>
          aifn <code>moons</code> ({n} points), <code>kNearestNeighbourGraph</code> (k = {kk}, symmetric) with
          heat-kernel affinities exp(−d²/2σ²) from <code>graphAffinity</code>, and{' '}
          <code>{method === 'spreading' ? 'labelSpreadingSteps' : 'labelPropagationSteps'}</code>. The large marks are
          the {2 * perClass} given labels; every other point takes its class colour once a score reaches it (grey: no
          score yet). Step 0 holds only the given labels; play to watch them flow along the graph.{' '}
          {method === 'spreading'
            ? `Spreading iterates F ← αSF + (1 − α)Y with α = ${alpha}; the given labels can be revised.`
            : 'Propagation iterates F ← D⁻¹WF and resets the given labels each step; its limit is the harmonic solution.'}{' '}
          Right: accuracy on the unlabelled points by step. Raise the noise or k until an edge bridges the moons and the
          labels leak across.
        </>
      }
    >
      <Plots cols={2} widths={[3, 2]}>
        <Plot x={ax} y={ay}>
          <Segments name="edges" segments={edges} />
          <Points name="no score yet" x={unreached.map((i) => rows[i][0])} y={unreached.map((i) => rows[i][1])} muted />
          <Points
            name="predicted"
            x={reached.map((i) => rows[i][0])}
            y={reached.map((i) => rows[i][1])}
            group={reached.map((i) => pred[i])}
            groupNames={CLASSES}
          />
          <Points
            name="given labels"
            x={given.picked.map((i) => rows[i][0])}
            y={given.picked.map((i) => rows[i][1])}
            group={given.picked.map((i) => truth[i])}
            groupNames={CLASSES}
            size={14}
          />
        </Plot>
        <Plot x={sx} y={sy}>
          <Curve name="accuracy" x={series.step} y={series.accuracy} slot={2} />
          {limit && (
            <Curve
              name="closed-form limit"
              x={[0, steps]}
              y={[limitAccuracy, limitAccuracy]}
              emphasis
              dashed
              width={1}
            />
          )}
          <Points name="now" x={[s.t]} y={[series.accuracy[at]]} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
