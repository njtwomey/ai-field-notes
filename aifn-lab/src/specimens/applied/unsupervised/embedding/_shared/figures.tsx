import { useMemo, useState } from 'react'
import { dataset } from 'aifn/learning/estimators'
import { digits, gaussians, swissRoll } from 'aifn-applied/data/synthetic'
import { classicalMds, kernelPca, metricMds, pca } from 'aifn-applied/unsupervised/embedding/linear'
import { isomap, laplacianEigenmaps, locallyLinearEmbedding } from 'aifn-applied/unsupervised/embedding/manifold'
import { jointProbabilities, tsneSteps, fuzzyGraph, umapSteps } from 'aifn-applied/unsupervised/embedding/neighbour'
import { rbf } from 'aifn/learning/kernels'
import { pairwiseDistances } from 'aifn/numerics/linalg'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState, variants } from '@lab/state'
import { Curve, Handle, Plot, Points, Readout, useAxis, useScaleColor, Vectors, type Vec2, type Vector } from '@lab/viz'
import { formatValue } from '@lab/views'

// ── PCA on 2-D data ─────────────────────────────────────────────────────────────────────────────────────────────

export function PcaSpecimen() {
  const state = useFigureState({
    data: row('the data', {
      angle: slider(-1.5, 1.5, 0.5, { label: 'orientation (rad)' }),
      ratio: slider(0.05, 1, 0.3, { label: 'minor / major spread' }),
    }),
    qx: slider(-6, 6, 2, { onChart: true }),
    qy: slider(-6, 6, -1, { onChart: true }),
  })
  const { angle, ratio } = state.data
  const query: Vec2 = [state.qx, state.qy]
  const data = useMemo(() => {
    // A Gaussian with standard deviations 2 and 2·ratio along axes rotated by `angle`.
    const c = Math.cos(angle)
    const s = Math.sin(angle)
    const [a, b] = [4, 4 * ratio * ratio]
    const cov = [
      [a * c * c + b * s * s, (a - b) * c * s],
      [(a - b) * c * s, a * s * s + b * c * c],
    ]
    return gaussians(stream('lab/embed/pca'), { means: [[1, 0.5]], covariances: [cov], n: 200 })
  }, [angle, ratio])
  const full = useMemo(() => pca().fit(dataset(data.x)), [data])
  const one = useMemo(() => pca({ components: 1 }).fit(dataset(data.x)), [data])
  const mean = toFlat(full.mean) as Vec2
  const comps = toRows(full.components)
  const sd = toFlat(full.explainedVariance).map(Math.sqrt)
  const vectors: Vector[] = comps.map((c, j) => ({
    from: mean,
    to: [mean[0] + 2 * sd[j] * c[0], mean[1] + 2 * sd[j] * c[1]],
    slot: j + 2,
    label: `PC${j + 1}`,
  }))
  const q = fromData(Float64Array.from(query), [1, 2])
  const score = toFlat(one.transform(q))[0]
  const back = toFlat(one.inverseTransform(one.transform(q))) as Vec2
  const cloud = useMemo(() => {
    const rows = toRows(data.x)
    return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
  }, [data])
  const x0 = useAxis({ label: 'x₀', range: [-6, 6] })
  const x1 = useAxis({ label: 'x₁', range: [-6, 6], equal: x0 })
  return (
    <Figure
      title="PCA axes of a 2-D cloud"
      purpose="The principal axes are the eigenvectors of the sample covariance, found from the SVD of the centred data; projecting onto the first keeps the most variance a single direction can."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout
            label="explained variance ratio"
            value={toFlat(full.explainedVarianceRatio)
              .map((v) => v.toFixed(3))
              .join(', ')}
          />
          <Readout label="PC1 score of the query" value={formatValue(score)} />
          <Readout
            label="reconstruction error"
            value={formatValue(Math.hypot(query[0] - back[0], query[1] - back[1]))}
          />
        </>
      }
      caption="Arrows run two standard deviations along each principal axis. Drag the query point: its projection onto PC1 (the one-component reconstruction) is joined to it by the segment, which is perpendicular to PC1. As the two spreads become equal the axes lose any preferred direction and the ratio tends to ½."
    >
      <Plot x={x0} y={x1}>
        <Points name="points" x={cloud.x} y={cloud.y} muted />
        <Vectors vectors={vectors} />
        <Curve name="projection" x={[query[0], back[0]]} y={[query[1], back[1]]} dashed slot={1} live />
        <Points name="reconstruction" x={[back[0]]} y={[back[1]]} slot={1} live />
        <Handle {...state.handle(['qx', 'qy'], { label: 'query' })} />
      </Plot>
    </Figure>
  )
}

// ── Manifold learning on a Swiss roll ───────────────────────────────────────────────────────────────────────────

const METHODS = {
  pca: { label: 'PCA', run: (x: Tensor) => pca({ components: 2 }).fit(dataset(x)).transform(x) },
  mds: { label: 'classical MDS', run: (x: Tensor) => classicalMds(pairwiseDistances(x)).embedding },
  smacof: { label: 'metric MDS (SMACOF)', run: (x: Tensor) => metricMds({ maxSteps: 100 }).fit(dataset(x)).embedding },
  kpca: {
    label: 'kernel PCA (RBF, ℓ = 4)',
    run: (x: Tensor) => kernelPca({ kernel: rbf({ lengthscale: 4 }) }).fit(dataset(x)).embedding,
  },
  isomap: { label: 'Isomap (k = 8)', run: (x: Tensor) => isomap({ neighbours: 8 }).fit(dataset(x)).embedding },
  eigenmaps: {
    label: 'Laplacian eigenmaps (k = 10)',
    run: (x: Tensor) => laplacianEigenmaps({ neighbours: 10 }).fit(dataset(x)).embedding,
  },
  lle: {
    label: 'LLE (k = 10)',
    run: (x: Tensor) => locallyLinearEmbedding({ neighbours: 10 }).fit(dataset(x)).embedding,
  },
} as const
type MethodName = keyof typeof METHODS

export function ManifoldSpecimen() {
  const state = useFigureState({
    method: choice(
      (Object.keys(METHODS) as MethodName[]).map((value) => ({ value, label: METHODS[value].label })),
      'isomap',
      { label: 'method' },
    ),
  })
  const method = state.method as MethodName
  const data = useMemo(() => swissRoll(stream('lab/embed/roll'), { n: 300, noise: 0.1, height: 8 }), [])
  const y = useMemo(() => {
    const rows = toRows(METHODS[method].run(data.x))
    return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
  }, [method, data])
  const t = useMemo(() => toFlat(data.t!), [data])
  const colour = useScaleColor('sequential')
  const [lo, hi] = [Math.min(...t), Math.max(...t)]
  const colours = useMemo(() => t.map((v) => colour((v - lo) / (hi - lo))), [t, colour, lo, hi])
  const z0 = useAxis({ label: 'z₀', hold: 'initial', key: method })
  const z1 = useAxis({ label: 'z₁', hold: 'initial', key: method, equal: z0 })
  return (
    <Figure
      title="Unrolling a Swiss roll"
      purpose="Every method maps the 3-D roll to the plane; the colour is the position along the roll, which a method that follows the manifold spreads out in order."
      state={state}
      defaultSize="L"
      caption="Linear methods (PCA, classical and metric MDS on straight-line distances) flatten the roll onto itself and mix the colours. Isomap measures distances along the neighbour graph and unrolls it into a strip; Laplacian eigenmaps and LLE keep neighbourhoods but distort the strip's proportions."
    >
      <Plot x={z0} y={z1}>
        <Points name="points" x={y.x} y={y.y} colors={colours} />
      </Plot>
    </Figure>
  )
}

// ── t-SNE and UMAP, step by step ────────────────────────────────────────────────────────────────────────────────

export function NeighbourEmbeddingSpecimen() {
  const state = useFigureState({
    method: variants(
      {
        tsne: { label: 't-SNE (exact)', params: { perplexity: slider(3, 40, 15, { label: 'perplexity', step: 1 }) } },
        umap: { label: 'UMAP', params: { neighbours: slider(3, 40, 12, { label: 'neighbours', step: 1 }) } },
      },
      { label: '1 · method', choiceLabel: 'method' },
    ),
  })
  const method = state.method.key
  const perplexity = state.method.key === 'tsne' ? state.method.values.perplexity : 15
  const neighbours = state.method.key === 'umap' ? state.method.values.neighbours : 12
  const [step, setStep] = useState(0)
  const data = useMemo(() => {
    const d = digits(stream('lab/embed/digits'), { perClass: 18, flip: 0.08, noise: 0.15 })
    const labels = toFlat(d.y!)
    const keep = labels.map((_, i) => i).filter((i) => labels[i] < 8)
    const rows = toRows(d.x)
    return {
      x: fromData(Float64Array.from(keep.flatMap((i) => rows[i])), [keep.length, rows[0].length]),
      y: keep.map((i) => labels[i]),
    }
  }, [])
  const run = useMemo(() => {
    if (method === 'tsne') {
      const t = trace(tsneSteps(jointProbabilities(data.x, perplexity)), {}, 1000, {
        stream: stream('lab/tsne'),
        every: 10,
        record: { cost: (s) => s.kl },
      })
      return {
        index: t.index,
        steps: t.steps.map((s) => toRows(s.embedding)),
        cost: toFlat(t.series.cost),
        label: 'KL(P ‖ Q)',
      }
    }
    const t = trace(umapSteps(fuzzyGraph(data.x, neighbours), { epochs: 200 }), {}, 200, {
      stream: stream('lab/umap'),
      every: 2,
    })
    return {
      index: t.index,
      steps: t.steps.map((s) => toRows(s.embedding)),
      cost: t.steps.map((s) => s.alpha),
      label: 'learning rate',
    }
  }, [method, perplexity, neighbours, data])
  const k = Math.min(Math.round(step / (method === 'tsne' ? 10 : 2)), run.steps.length - 1)
  const e = run.steps[k]
  const layout = useMemo(() => ({ x: e.map((r) => r[0]), y: e.map((r) => r[1]) }), [e])
  // Only relative positions matter, so the axes refit to every frame.
  const z0 = useAxis({ label: 'z₀' })
  const z1 = useAxis({ label: 'z₁', equal: z0 })
  return (
    <Figure
      title="t-SNE and UMAP layouts, iteration by iteration"
      purpose="Both place points so that neighbours in the input stay neighbours in the plane: t-SNE by gradient descent on KL(P ‖ Q), UMAP by sampled attractive and repulsive moves on a fuzzy neighbour graph."
      state={state}
      defaultSize="L"
      controls={
        <>
          <ControlRow label="2 · iterations">
            <Player
              value={k}
              onChange={(i) => setStep(run.index[i])}
              count={run.steps.length}
              format={(i) => String(run.index[i])}
              label={method === 'tsne' ? 'iteration' : 'epoch'}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label={method === 'tsne' ? 'iteration' : 'epoch'} value={run.index[k]} />
          <Readout label={run.label} value={formatValue(run.cost[k])} />
          <Readout label="points" value={e.length} />
        </>
      }
      caption="Noisy 5 × 7 digit glyphs (35 dimensions), classes 0 to 7, coloured by digit. t-SNE's first 250 iterations exaggerate P, which pulls clusters apart early; UMAP starts from a spectral layout and its learning rate falls to zero. Press play from iteration 0 (the random or spectral start); the axes refit to each frame, since only relative positions matter."
    >
      <Plot x={z0} y={z1} legend={false}>
        <Points
          name="digits"
          x={layout.x}
          y={layout.y}
          group={data.y}
          groupNames={Array.from({ length: 8 }, (_, c) => `digit ${c}`)}
        />
      </Plot>
    </Figure>
  )
}
