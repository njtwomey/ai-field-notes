import { useMemo, useState } from 'react'
import { digits } from 'aifn-methods/data/synthetic'
import { fuzzyGraph, jointProbabilities, tsneSteps, umapSteps } from 'aifn-methods/unsupervised/embedding/neighbour'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { ControlGroup, Figure, NumberSelector, Player, Plot, Points, Readout, Select, useAxis } from 'aifn-render'

const METHOD_OPTIONS = [
  { value: 'umap', label: 'UMAP (Fuzzy Simplicial Set)' },
  { value: 'tsne', label: 't-SNE (Exact KL Minimisation)' },
]

const FRAME_OPTIONS = [
  { value: 'run', label: 'Fixed across whole run (shows expansion)' },
  { value: 'grow', label: 'Grow only (widens as points separate)' },
  { value: 'step', label: 'Fit each step (normalised view)' },
]

const fmt = (v: number | null | undefined, digits = 3) =>
  v !== null && v !== undefined && Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '—'

export function UmapExplorer() {
  const [method, setMethod] = useState<'umap' | 'tsne'>('umap')
  const [frame, setFrame] = useState<'run' | 'grow' | 'step'>('run')
  const [perplexity, setPerplexity] = useState(15)
  const [neighbours, setNeighbours] = useState(12)
  const [stepIndex, setStepIndex] = useState(0)

  // 35-dimensional 5x7 binary digit glyphs
  const data = useMemo(() => {
    const d = digits(stream('lab/embed/digits'), { perClass: 16, flip: 0.08, noise: 0.15 })
    const labels = toFlat(d.y!)
    const keep = labels.map((_, i) => i).filter((i) => labels[i] < 8)
    const rows = toRows(d.x)
    return {
      x: fromData(Float64Array.from(keep.flatMap((i) => rows[i])), [keep.length, rows[0].length]),
      y: keep.map((i) => labels[i]),
    }
  }, [])

  // Run embedding optimization trace
  const run = useMemo(() => {
    if (method === 'tsne') {
      const t = trace(tsneSteps(jointProbabilities(data.x, perplexity)), {}, 600, {
        stream: stream('umap/tsne'),
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
      stream: stream('umap/umap'),
      every: 2,
    })
    return {
      index: t.index,
      steps: t.steps.map((s) => toRows(s.embedding)),
      cost: t.steps.map((s) => s.alpha),
      label: 'Learning Rate α',
    }
  }, [method, perplexity, neighbours, data])

  const k = Math.min(stepIndex, run.steps.length - 1)
  const e = run.steps[k]
  const layout = useMemo(() => ({ x: e.map((r) => r[0]), y: e.map((r) => r[1]) }), [e])

  // Coordinate bounding range
  const runRange = useMemo((): [number, number] => {
    let lo = Infinity
    let hi = -Infinity
    for (const rows of run.steps) {
      for (const r of rows) {
        lo = Math.min(lo, r[0], r[1])
        hi = Math.max(hi, r[0], r[1])
      }
    }
    const pad = 0.05 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }, [run])

  const key = `${method}:${perplexity}:${neighbours}`
  const z0 = useAxis({
    label: 'z₁ (Embedding Dimension 1)',
    ...(frame === 'run' ? { range: runRange, key } : frame === 'grow' ? { hold: 'union' as const, key } : {}),
    nice: false,
  })
  const z1 = useAxis({
    label: 'z₂ (Embedding Dimension 2)',
    equal: z0,
    ...(frame === 'run' ? { range: runRange, key } : frame === 'grow' ? { hold: 'union' as const, key } : {}),
    nice: false,
  })

  return (
    <Figure
      title="UMAP vs t-SNE: Iteration-by-Iteration Manifold Layout"
      purpose="UMAP constructs a fuzzy simplicial set representation of the high-dimensional data and minimises cross-entropy to place points in 2D, while t-SNE minimises Kullback-Leibler divergence between Student-t and Gaussian affinities."
      defaultSize="L"
      controls={
        <div className="flex flex-col gap-3">
          <ControlGroup title="1 · Manifold Embedding Method">
            <Select
              label="Algorithm"
              value={method}
              onChange={(v) => {
                setMethod(v as 'umap' | 'tsne')
                setStepIndex(0)
              }}
              options={METHOD_OPTIONS}
            />
            {method === 'umap' ? (
              <NumberSelector
                label="k Nearest Neighbours (n_neighbors)"
                value={neighbours}
                onChange={setNeighbours}
                min={3}
                max={40}
                step={1}
                suggestions={[5, 12, 20, 30]}
              />
            ) : (
              <NumberSelector
                label="Perplexity"
                value={perplexity}
                onChange={setPerplexity}
                min={3}
                max={50}
                step={1}
                suggestions={[5, 15, 30, 50]}
              />
            )}
            <Select
              label="Axes Scaling Frame"
              value={frame}
              onChange={(v) => setFrame(v as 'run' | 'grow' | 'step')}
              options={FRAME_OPTIONS}
            />
          </ControlGroup>

          <ControlGroup title="2 · Optimization Playback">
            <Player
              value={k}
              onChange={setStepIndex}
              count={run.steps.length}
              format={(i) => `${method === 'tsne' ? 'Iteration' : 'Epoch'} ${run.index[i]}`}
              label={method === 'tsne' ? 'Iteration' : 'Epoch'}
            />
          </ControlGroup>
        </div>
      }
      readouts={{
        'Optimization Progress': (
          <>
            <Readout label={method === 'tsne' ? 'Iteration' : 'Epoch'} value={String(run.index[k])} />
            <Readout label={run.label} value={fmt(run.cost[k])} />
            <Readout label="Embedded Data Points" value={String(e.length)} />
            <Readout label="Input Dimensionality" value="35D (5×7 pixel glyphs)" />
          </>
        ),
      }}
      caption="35-dimensional noisy handwritten digit glyphs (classes 0 through 7) projected onto 2D. UMAP initializes from a Laplacian spectral layout and anneals its learning rate; t-SNE starts from tiny random positions with early exaggeration to pull clusters apart before expanding into final positions. Use the playback slider to step through the trajectory."
    >
      <Plot x={z0} y={z1}>
        <Points
          name="Digit Glyphs"
          x={layout.x}
          y={layout.y}
          group={data.y}
          groupNames={Array.from({ length: 8 }, (_, c) => `Digit ${c}`)}
          size={8}
        />
      </Plot>
    </Figure>
  )
}
