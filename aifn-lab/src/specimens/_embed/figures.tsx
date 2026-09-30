import { useMemo, useState } from 'react'
import { digits, gaussians, swissRoll } from 'aifn/datasets'
import {
  classicalMds,
  isomap,
  jointProbabilities,
  kernelPca,
  laplacianEigenmaps,
  locallyLinearEmbedding,
  metricMds,
  pca,
  tsneSteps,
  fuzzyGraph,
  umapSteps,
} from 'aifn/embed'
import { rbf } from 'aifn/kernels'
import { pairwiseDistances } from 'aifn/metrics'
import { stream } from 'aifn/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Readout, useScaleColor, XYChart, type Vec2, type Vector } from '@lab/viz'
import { formatValue } from '@lab/views'

// ── PCA on 2-D data ─────────────────────────────────────────────────────────────────────────────────────────────

export function PcaSpecimen() {
  const [angle, setAngle] = useState(0.5)
  const [ratio, setRatio] = useState(0.3)
  const [query, setQuery] = useState<Vec2>([2, -1])
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
  const full = useMemo(() => pca().fit({ x: data.x }), [data])
  const one = useMemo(() => pca({ components: 1 }).fit({ x: data.x }), [data])
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
  const rows = toRows(data.x)
  return (
    <Figure
      title="PCA axes of a 2-D cloud"
      description="The principal axes are the eigenvectors of the sample covariance, found from the SVD of the centred data; projecting onto the first keeps the most variance a single direction can."
      defaultSize="L"
      controls={
        <ControlRow label="the data">
          <Slider label="orientation (rad)" value={angle} min={-1.5} max={1.5} onChange={setAngle} />
          <Slider label="minor / major spread" value={ratio} min={0.05} max={1} onChange={setRatio} />
        </ControlRow>
      }
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
      <XYChart
        aspect="equal"
        xLabel="x₀"
        yLabel="x₁"
        rescaleOnChange={false}
        series={[
          { name: 'points', type: 'scatter', x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), muted: true },
          { name: 'projection', type: 'line', x: [query[0], back[0]], y: [query[1], back[1]], dashed: true, slot: 1 },
          { name: 'reconstruction', type: 'scatter', x: [back[0]], y: [back[1]], slot: 1 },
        ]}
        vectors={vectors}
        handles={[{ kind: 'point', at: query, onDrag: setQuery, label: 'query' }]}
      />
    </Figure>
  )
}

// ── Manifold learning on a Swiss roll ───────────────────────────────────────────────────────────────────────────

const METHODS = {
  pca: { label: 'PCA', run: (x: Tensor) => pca({ components: 2 }).fit({ x }).transform(x) },
  mds: { label: 'classical MDS', run: (x: Tensor) => classicalMds(pairwiseDistances(x)).embedding },
  smacof: { label: 'metric MDS (SMACOF)', run: (x: Tensor) => metricMds({ maxIterations: 100 }).fit({ x }).embedding },
  kpca: {
    label: 'kernel PCA (RBF, ℓ = 4)',
    run: (x: Tensor) => kernelPca({ kernel: rbf({ lengthscale: 4 }) }).fit({ x }).embedding,
  },
  isomap: { label: 'Isomap (k = 8)', run: (x: Tensor) => isomap({ neighbours: 8 }).fit({ x }).embedding },
  eigenmaps: {
    label: 'Laplacian eigenmaps (k = 10)',
    run: (x: Tensor) => laplacianEigenmaps({ neighbours: 10 }).fit({ x }).embedding,
  },
  lle: { label: 'LLE (k = 10)', run: (x: Tensor) => locallyLinearEmbedding({ neighbours: 10 }).fit({ x }).embedding },
} as const
type MethodName = keyof typeof METHODS

export function ManifoldSpecimen() {
  const [method, setMethod] = useState<MethodName>('isomap')
  const data = useMemo(() => swissRoll(stream('lab/embed/roll'), { n: 300, noise: 0.1, height: 8 }), [])
  const y = useMemo(() => toRows(METHODS[method].run(data.x)), [method, data])
  const t = useMemo(() => toFlat(data.t!), [data])
  const colour = useScaleColor('sequential')
  const [lo, hi] = [Math.min(...t), Math.max(...t)]
  const colours = t.map((v) => colour((v - lo) / (hi - lo)))
  return (
    <Figure
      title="Unrolling a Swiss roll"
      description="Every method maps the 3-D roll to the plane; the colour is the position along the roll, which a method that follows the manifold spreads out in order."
      defaultSize="L"
      controls={
        <Select
          label="method"
          value={method}
          onChange={setMethod}
          options={(Object.keys(METHODS) as MethodName[]).map((value) => ({ value, label: METHODS[value].label }))}
        />
      }
      caption="Linear methods (PCA, classical and metric MDS on straight-line distances) flatten the roll onto itself and mix the colours. Isomap measures distances along the neighbour graph and unrolls it into a strip; Laplacian eigenmaps and LLE keep neighbourhoods but distort the strip's proportions."
    >
      <XYChart
        aspect="equal"
        xLabel="z₀"
        yLabel="z₁"
        axisKey={method}
        series={[
          { name: 'points', type: 'scatter', x: y.map((r) => r[0]), y: y.map((r) => r[1]), pointColors: colours },
        ]}
      />
    </Figure>
  )
}

// ── t-SNE and UMAP, step by step ────────────────────────────────────────────────────────────────────────────────

export function NeighbourEmbeddingSpecimen() {
  const [method, setMethod] = useState<'tsne' | 'umap'>('tsne')
  const [perplexity, setPerplexity] = useState(15)
  const [neighbours, setNeighbours] = useState(12)
  const [step, setStep] = useState(1000)
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
  return (
    <Figure
      title="t-SNE and UMAP layouts, iteration by iteration"
      description="Both place points so that neighbours in the input stay neighbours in the plane: t-SNE by gradient descent on KL(P ‖ Q), UMAP by sampled attractive and repulsive moves on a fuzzy neighbour graph."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · method">
            <Select
              label="method"
              value={method}
              onChange={(m) => {
                setMethod(m)
                setStep(m === 'tsne' ? 1000 : 200)
              }}
              options={[
                { value: 'tsne', label: 't-SNE (exact)' },
                { value: 'umap', label: 'UMAP' },
              ]}
            />
            {method === 'tsne' ? (
              <Slider label="perplexity" value={perplexity} min={3} max={40} step={1} onChange={setPerplexity} />
            ) : (
              <Slider label="neighbours" value={neighbours} min={3} max={40} step={1} onChange={setNeighbours} />
            )}
          </ControlRow>
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
      caption="Noisy 5 × 7 digit glyphs (35 dimensions), classes 0 to 7, coloured by digit. t-SNE's first 250 iterations exaggerate P, which pulls clusters apart early; UMAP starts from a spectral layout and its learning rate falls to zero. Axes refit to each frame, since only relative positions matter."
    >
      <XYChart
        aspect="equal"
        xLabel="z₀"
        yLabel="z₁"
        legend={false}
        series={[
          {
            name: 'digits',
            type: 'scatter',
            x: e.map((r) => r[0]),
            y: e.map((r) => r[1]),
            group: data.y,
            groupNames: Array.from({ length: 8 }, (_, c) => `digit ${c}`),
          },
        ]}
      />
    </Figure>
  )
}
