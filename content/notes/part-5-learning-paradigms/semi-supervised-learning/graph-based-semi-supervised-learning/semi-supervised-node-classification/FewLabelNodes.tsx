import { useMemo } from 'react'
import {
  call,
  choice,
  Figure,
  float,
  forceLayout,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
  useStreamed,
} from 'aifn-render'
import { karateClub } from 'aifn-methods/data/real'
import type { NodeSnapshot } from 'aifn-methods/neural/graph'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { harmonicLabels } from 'aifn-compute/graph/propagation'
import { child, stream } from 'aifn-compute/foundation/random'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { contextualBlockModel, semiSupervisedSplit } from 'aifn-methods/data/synthetic'

const BLOCK = 40
const HIDDEN = 8

/** The share of `rows` whose prediction matches the truth. */
const accuracyOf = (pred: ArrayLike<number>, y: ArrayLike<number>, rows: readonly number[]) =>
  rows.length ? rows.filter((i) => pred[i] === y[i]).length / rows.length : 0

/**
 * Semi-supervised node classification on one graph: harmonic label propagation (graph only), logistic regression on
 * the node features (features only) and a two-layer graph convolutional network trained on the labelled nodes (both),
 * on a two-block stochastic block model with noisy features or on Zachary's karate club.
 */
export function FewLabelNodes() {
  const fig = useFigureState({
    graph: choice(
      [
        { value: 'block', label: 'two-block model with features' },
        { value: 'karate', label: "Zachary's karate club (no features)" },
      ],
      'block',
      { label: 'graph' },
    ),
    pIn: slider(0, 0.4, 0.12, {
      step: 0.01,
      label: 'edge probability within a block',
      when: (v) => v.graph === 'block',
    }),
    pOut: slider(0, 0.4, 0.02, {
      step: 0.01,
      label: 'edge probability across blocks',
      when: (v) => v.graph === 'block',
    }),
    signal: slider(0, 3, 1, { step: 0.1, label: 'feature signal μ', when: (v) => v.graph === 'block' }),
    perClass: int(3, { min: 1, max: 15, label: 'labelled nodes per class', suggestions: [1, 3, 10] }),
    steps: int(200, { ge: 1, le: 2000, label: 'training steps (Adam)', suggestions: [50, 200, 500] }),
    stepSize: float(0.02, { gt: 0, le: 1, scale: 'log10', label: 'learning rate', suggestions: [0.005, 0.02, 0.05] }),
    seed: int(1, { min: 1, max: 999, label: 'seed' }),
  })
  const { graph, pIn, pOut, signal, perClass, steps, stepSize, seed } = fig

  const data = useMemo(() => {
    if (graph === 'karate') {
      const d = karateClub()
      return { graph: d.graph, x: d.x, y: toFlat(d.y!), hasFeatures: false }
    }
    const d = contextualBlockModel(stream(seed), { sizes: [BLOCK, BLOCK], pIn, pOut, features: 4, signal })
    return { graph: d.graph, x: d.x, y: toFlat(d.y), hasFeatures: true }
  }, [graph, pIn, pOut, signal, seed])

  const homophily = useMemo(() => {
    const e = data.graph.edges
    return e.length ? e.filter((v) => data.y[v.from] === data.y[v.to]).length / e.length : NaN
  }, [data])
  const layout = useMemo(() => forceLayout(data.graph.nodes, data.graph.edges), [data])
  const split = useMemo(
    () => semiSupervisedSplit(child(stream(seed), 'split'), data.y, perClass),
    [data, seed, perClass],
  )
  const U = split.unlabelled

  const propagated = useMemo(
    () => toFlat(harmonicLabels(data.graph, split.labels, { classes: 2 }).labels),
    [data, split],
  )
  const featuresOnly = useMemo(() => {
    const rows = toArray(data.x) as number[][]
    const model = logisticRegression({ l2: 1 }).fit(
      dataset(tensor(split.labelled.map((i) => rows[i])), tensor(split.labelled.map((i) => data.y[i]))),
    )
    return toFlat(model.decide(data.x))
  }, [data, split])

  // The GCN trains in the compute worker; the task is memoised so the stream restarts only when an input changes.
  const task = useMemo(
    () =>
      call<NodeSnapshot>('neural/graph/nodeClassificationRun', {
        data: graph === 'karate' ? call('data/real/karateClub') : { graph: data.graph, x: data.x, y: tensor(data.y) },
        train: split.labelled,
        kind: 'gcn',
        hidden: HIDDEN,
        steps,
        stepSize,
        every: steps,
        seed,
      }),
    [graph, data, split, steps, stepSize, seed],
  )
  const run = useStreamed(task)
  const last = run.value?.checkpoints.at(-1)
  const finished = run.value?.done === true
  const gcn = useMemo(() => (last ? last.logits.map((z) => (z[1] > z[0] ? 1 : 0)) : null), [last])

  const xa = useAxis({ label: '', nice: false })
  const ya = useAxis({ label: '', nice: false, equal: xa })
  const edges = useMemo(
    () =>
      data.graph.edges.map((e) => ({
        from: [layout[e.from].x, layout[e.from].y] as const,
        to: [layout[e.to].x, layout[e.to].y] as const,
      })),
    [data, layout],
  )

  const panel = (pred: readonly number[] | null, title: string, stale: boolean) => {
    const wrong = pred ? U.filter((i) => pred[i] !== data.y[i]) : []
    return (
      <Plot x={xa} y={ya} title={title} bare>
        <Segments name="edge" segments={edges} muted width={0.6} />
        {wrong.length > 0 && (
          <Points
            name="wrong"
            x={wrong.map((i) => layout[i].x)}
            y={wrong.map((i) => layout[i].y)}
            tone="destructive"
            size={15}
            stale={stale}
          />
        )}
        {pred &&
          [0, 1].map((c) => {
            const rows = U.filter((i) => pred[i] === c)
            return rows.length > 0 ? (
              <Points
                key={c}
                name={`predicted class ${c}`}
                x={rows.map((i) => layout[i].x)}
                y={rows.map((i) => layout[i].y)}
                slot={c}
                stale={stale}
              />
            ) : null
          })}
        <Points
          name="labelled"
          x={split.labelled.map((i) => layout[i].x)}
          y={split.labelled.map((i) => layout[i].y)}
          emphasis
          size={10}
        />
      </Plot>
    )
  }

  const gcnAccuracy = gcn && finished ? formatNumber(accuracyOf(gcn, data.y, U)) : run.error ? 'failed' : 'training…'
  return (
    <Figure
      title="Graph, features, or both"
      state={fig}
      caption={`Each panel draws the same graph, coloured by the predicted class of every unlabelled node; black nodes are labelled, a red halo marks a mistake. Left: harmonic label propagation, which uses only the edges. Right: a two-layer graph convolutional network (${HIDDEN} hidden units, full-batch Adam on the labelled nodes' cross-entropy, trained in your browser's worker), which uses the edges and the node features. The block model has ${2 * BLOCK} nodes in two blocks; nodes of the same block are joined with the first probability and nodes of different blocks with the second, and each node has 4 features whose class mean is μ along one axis plus unit Gaussian noise. With a homophilous graph and weak features the network beats both single sources; with equal edge probabilities the graph carries no class information and the network falls below the features alone. The karate club has no features, each member being its own one-hot vector, so the network can only use the graph.${run.error ? ` (Training failed: ${run.error}.)` : ''}`}
      readouts={
        <>
          <Readout label="edge homophily" value={formatNumber(homophily)} />
          <Readout label="label propagation" value={formatNumber(accuracyOf(propagated, data.y, U))} />
          <Readout
            label="features only (logistic)"
            value={data.hasFeatures ? formatNumber(accuracyOf(featuresOnly, data.y, U)) : 'no features'}
          />
          <Readout label="graph convolutional network" value={gcnAccuracy} />
        </>
      }
    >
      <Plots cols={2}>
        {panel(propagated, 'label propagation (graph only)', false)}
        {panel(gcn, 'graph convolutional network (graph and features)', !finished)}
      </Plots>
    </Figure>
  )
}
