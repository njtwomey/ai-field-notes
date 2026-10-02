import { useMemo, useState } from 'react'
import { dataset } from 'aifn/learning/estimators'
import {
  BORDER,
  CORE,
  dbscan,
  gaussianMixtureSteps,
  kmeans,
  kmeansSteps,
  linkage,
  optics,
  spectralClustering,
  type CovarianceType,
  type Linkage,
} from 'aifn-applied/unsupervised/clustering'
import { blobs, circles, moons } from 'aifn-applied/data/synthetic'
import { covarianceEllipse } from 'aifn/numerics/geometry'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Bars, Curve, Handle, Plot, Plots, Points, Readout, useAxis, type Vec2 } from '@lab/viz'
import { DendrogramPanel, formatValue } from '@lab/views'

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x0: rows.map((r) => r[0]), x1: rows.map((r) => r[1]) }
}

/** Four blobs of unequal spread: the data of the centroid and mixture figures. */
const blobData = () =>
  blobs(stream('lab/cluster/blobs'), {
    n: [60, 60, 50, 40],
    centers: [
      [-3, -2],
      [2.5, -2.5],
      [0, 2.5],
      [4, 2],
    ],
    sd: [0.8, 1.1, 0.7, 0.5],
  })

// ── k-means ───────────────────────────────────────────────────────────────────────────────────────────────────────

const START: Vec2[] = [
  [-1, 4],
  [0, 3.5],
  [1, 4],
  [2, 3.5],
]

export function KMeansSpecimen() {
  const data = useMemo(() => blobData(), [])
  const [start, setStart] = useState<Vec2[]>(START)
  const [step, setStep] = useState(0)
  const run = useMemo(
    () =>
      trace(kmeansSteps(data.x, { k: 4 }), { centroids: fromData(Float64Array.from(start.flat()), [4, 2]) }, 50, {
        record: { inertia: (s) => s.inertia },
      }),
    [data, start],
  )
  const k = Math.min(step, run.steps.length - 1)
  const state = run.steps[k]
  const cols = useMemo(() => columns(data.x), [data])
  const paths = useMemo(() => run.steps.slice(0, k + 1).map((s) => toRows(s.centroids)), [run, k])
  const best = useMemo(
    () => kmeans({ k: 4, restarts: 10 }).fit(dataset(data.x), { stream: stream('lab/kmeans') }).inertia,
    [data],
  )
  const labels = useMemo(() => toFlat(state.labels), [state])
  const tracks = useMemo(
    () => [0, 1, 2, 3].map((j) => ({ x: paths.map((p) => p[j][0]), y: paths.map((p) => p[j][1]) })),
    [paths],
  )
  const centroids = useMemo(() => ({ x: paths[k].map((c) => c[0]), y: paths[k].map((c) => c[1]) }), [paths, k])
  const x0 = useAxis({ label: 'x₀', range: [-6, 7] })
  const x1 = useAxis({ label: 'x₁', range: [-6, 6], equal: x0 })
  return (
    <Figure
      title="Lloyd's algorithm from centroids you place"
      purpose="Each step moves every centroid to the mean of its points, then reassigns every point to its nearest centroid; the inertia never rises, but where it settles depends on the start."
      defaultSize="L"
      controls={
        <ControlRow label="steps">
          <Player value={k} onChange={setStep} count={run.steps.length} label="step" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="step" value={`${k} of ${run.steps.length - 1}`} />
          <Readout label="inertia" value={formatValue(state.inertia)} />
          <Readout label="best of 10 k-means++ restarts" value={formatValue(best)} />
          <Readout label="converged" value={state.converged ? 'yes' : 'no'} />
          <Readout label="empty clusters" value={state.empty.length ? state.empty.join(', ') : 'none'} />
        </>
      }
      caption="Drag the four starting centroids (the first points of each path; step 0 shows them with their first assignment). The start decides which local optimum Lloyd's algorithm reaches: place two starts inside one blob and the run ends with two centroids sharing it and one straddling two blobs, at a higher inertia than the best k-means++ restart."
    >
      <Plot x={x0} y={x1}>
        <Points
          name="points"
          x={cols.x0}
          y={cols.x1}
          group={labels}
          groupNames={['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']}
        />
        {tracks.map((t, j) => (
          <Curve key={j} name="centroid paths" x={t.x} y={t.y} slot={j} showPoints />
        ))}
        <Points name="centroids" x={centroids.x} y={centroids.y} emphasis />
        {start.map((p, j) => (
          <Handle
            key={j}
            kind="point"
            at={p}
            label={`start ${j}`}
            onDrag={(q) => setStart((s) => s.map((c, i) => (i === j ? q : c)))}
          />
        ))}
      </Plot>
    </Figure>
  )
}

// ── Gaussian mixtures by EM ─────────────────────────────────────────────────────────────────────────────────────

export function MixtureSpecimen() {
  const data = useMemo(() => blobData(), [])
  const figure = useFigureState({
    model: row('1 · model', {
      covariance: choice(['full', 'diagonal', 'spherical'] as CovarianceType[], 'full', { label: 'covariances' }),
    }),
  })
  const covariance = figure.model.covariance as CovarianceType
  const [step, setStep] = useState(0)
  const run = useMemo(
    () =>
      trace(
        gaussianMixtureSteps(data.x, { k: 4, covariance, tolerance: 1e-6 }),
        {
          weights: fromData(Float64Array.from([0.25, 0.25, 0.25, 0.25]), [4]),
          means: fromData(Float64Array.from(START.flat()), [4, 2]),
          covariances: fromData(Float64Array.from([4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4]), [4, 2, 2]),
        },
        200,
        { record: { ll: (s) => s.logLikelihood } },
      ),
    [data, covariance],
  )
  const k = Math.min(step, run.steps.length - 1)
  const state = run.steps[k]
  const cols = useMemo(() => columns(data.x), [data])
  const owner = useMemo(() => toRows(state.responsibilities).map((r) => r.indexOf(Math.max(...r))), [state])
  const ellipses = useMemo(() => {
    const means = toRows(state.means)
    const covs = toFlat(state.covariances)
    return means.flatMap((m, j) =>
      [1, 2].map((sd) => {
        const e = toRows(
          covarianceEllipse(
            m as Vec2,
            [
              [covs[j * 4], covs[j * 4 + 1]],
              [covs[j * 4 + 2], covs[j * 4 + 3]],
            ],
            { k: sd },
          ).points,
        )
        return { j, sd, x: e.map((p) => p[0]), y: e.map((p) => p[1]) }
      }),
    )
  }, [state])
  const ll = useMemo(() => ({ x: Array.from(run.index), y: toFlat(run.series.ll) }), [run])
  const now = useMemo(() => ({ x: [ll.x[k]], y: [ll.y[k]] }), [ll, k])
  const x0 = useAxis({ label: 'x₀', hold: 'initial' })
  const x1 = useAxis({ label: 'x₁', hold: 'initial', equal: x0 })
  const stepAxis = useAxis({ label: 'step', hold: 'initial', key: run })
  const llAxis = useAxis({ label: 'log-likelihood', hold: 'initial', key: run })
  return (
    <Figure
      title="EM for a Gaussian mixture"
      purpose="The E-step gives each point a responsibility for each component; the M-step refits every weight, mean and covariance to those soft assignments, and the log-likelihood rises every step."
      state={figure}
      defaultSize="L"
      controls={
        <ControlRow label="2 · EM steps">
          <Player value={k} onChange={setStep} count={run.steps.length} label="step" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="step" value={`${k} of ${run.steps.length - 1}`} />
          <Readout label="mean log-likelihood" value={formatValue(state.logLikelihood)} />
          <Readout
            label="weights"
            value={toFlat(state.weights)
              .map((w) => w.toFixed(2))
              .join(', ')}
          />
        </>
      }
      caption="Ellipses are the 1σ (solid) and 2σ (dashed) contours of each component; points take the colour of their most responsible component. The mean log-likelihood rises every step. Diagonal covariances keep the ellipses axis-aligned, spherical ones keep them round."
    >
      <Plots rows={2} heights={[3, 1]}>
        <Plot x={x0} y={x1} legend={false}>
          <Points
            name="points"
            x={cols.x0}
            y={cols.x1}
            group={owner}
            groupNames={['component 0', 'component 1', 'component 2', 'component 3']}
          />
          {ellipses.map((e) => (
            <Curve key={`${e.j}-${e.sd}`} name={`component ${e.j}`} x={e.x} y={e.y} slot={e.j} dashed={e.sd === 2} />
          ))}
        </Plot>
        <Plot x={stepAxis} y={llAxis} legend={false}>
          <Curve name="mean log-likelihood" x={ll.x} y={ll.y} />
          <Points name="now" x={now.x} y={now.y} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Agglomerative clustering ────────────────────────────────────────────────────────────────────────────────────

export function DendrogramSpecimen() {
  const figure = useFigureState({
    setup: row('1 · data and linkage', {
      dataset: choice(['blobs', 'moons'], 'blobs', { label: 'data' }),
      method: choice(['single', 'complete', 'average', 'ward'] as Linkage[], 'ward', { label: 'linkage' }),
    }),
  })
  const dataset = figure.setup.dataset as 'blobs' | 'moons'
  const method = figure.setup.method as Linkage
  const data = useMemo(
    () =>
      dataset === 'blobs'
        ? blobs(stream('lab/cluster/dendrogram'), {
            n: 36,
            centers: [
              [-2, 0],
              [2, 0],
              [0, 3],
            ],
            sd: 0.6,
          })
        : moons(stream('lab/cluster/dendrogram-moons'), { n: 40, noise: 0.06 }),
    [dataset],
  )
  const Z = useMemo(() => linkage(data.x, method), [data, method])
  const heights = toRows(Z).map((r) => r[2])
  // The cut starts between the last two merges (two or three clusters); a new tree resets it there.
  const [cut, setCut] = useState<{ tree: Tensor; at: number } | null>(null)
  const at = cut && cut.tree === Z ? cut.at : (heights[heights.length - 2] + heights[heights.length - 3]) / 2
  return (
    <Figure
      title="The dendrogram and the cut"
      purpose="Agglomerative clustering merges the two closest clusters at each step; cutting the tree at a height undoes every merge above it, and the linkage decides which shapes survive."
      state={figure}
      caption="Drag the cut: clusters below it take one colour in the tree and among the points. Single linkage chains along the moons and recovers them; Ward and complete linkage prefer compact groups and split each moon."
    >
      <DendrogramPanel merges={Z} cut={at} onCut={(h) => setCut({ tree: Z, at: h })} points={data.x} />
    </Figure>
  )
}

// ── DBSCAN and OPTICS ───────────────────────────────────────────────────────────────────────────────────────────

export function DbscanSpecimen() {
  const data = useMemo(() => moons(stream('lab/cluster/dbscan'), { n: 200, noise: 0.08 }), [])
  const figure = useFigureState({
    eps: slider(0.03, 0.5, 0.15, { label: 'ε', onChart: true }),
    minSamples: slider(2, 15, 5, { label: 'minSamples', step: 1 }),
  })
  const { eps, minSamples } = figure
  const model = useMemo(() => dbscan({ eps, minSamples }).fit(dataset(data.x)), [data, eps, minSamples])
  const order = useMemo(() => optics({ minSamples }).fit(dataset(data.x)), [data, minSamples])
  const cols = useMemo(() => columns(data.x), [data])
  const groups = useMemo(() => {
    const labels = toFlat(model.labels)
    const roles = toFlat(model.roles)
    const pick = (f: (i: number) => boolean) => {
      const idx = cols.x0.map((_, i) => i).filter(f)
      return { idx, x: idx.map((i) => cols.x0[i]), y: idx.map((i) => cols.x1[i]) }
    }
    const core = pick((i) => roles[i] === CORE)
    return {
      core: { ...core, group: core.idx.map((i) => labels[i] % 8) },
      // Border points as rings: a transparent fill with the ink outline.
      border: (() => {
        const b = pick((i) => roles[i] === BORDER)
        return { ...b, ring: b.idx.map(() => 'rgba(0,0,0,0)') }
      })(),
      noise: pick((i) => labels[i] < 0),
    }
  }, [model, cols])
  const reach = useMemo(() => {
    const r = toFlat(order.reachability)
    const ordering = toFlat(order.ordering)
    const bars = ordering.map((p) => (Number.isFinite(r[p]) ? r[p] : NaN))
    const top = Math.max(...bars.filter(Number.isFinite)) * 1.1
    return { x: ordering.map((_, i) => i), y: bars.map((b) => (Number.isFinite(b) ? b : top)), top }
  }, [order])
  const counts = { core: groups.core.idx.length, border: groups.border.idx.length, noise: groups.noise.idx.length }
  const x0 = useAxis({ label: 'x₀' })
  const x1 = useAxis({ label: 'x₁', equal: x0 })
  const orderAxis = useAxis({ label: 'OPTICS order', range: [-0.5, reach.x.length - 0.5], nice: false })
  const reachAxis = useAxis({ label: 'reachability', range: [0, reach.top] })
  return (
    <Figure
      title="DBSCAN's ε and OPTICS reachability"
      purpose="A core point has at least minSamples points within ε; clusters are chains of core points plus the border points they reach. OPTICS orders the points so that every ε can be read off one reachability plot."
      state={figure}
      defaultSize="L"
      readouts={
        <>
          <Readout label="clusters" value={model.clusters} />
          <Readout label="core" value={counts.core} />
          <Readout label="border" value={counts.border} />
          <Readout label="noise" value={counts.noise} />
        </>
      }
      caption="Drag the ε line on the reachability plot: each valley below the line is a cluster, and bars above it start a new cluster or are noise. Core points are filled by cluster, border points ringed, noise grey. Too small an ε shatters the moons; too large merges them."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={x0} y={x1} legend={false}>
          <Points name="noise" x={groups.noise.x} y={groups.noise.y} muted />
          <Points
            name="core"
            x={groups.core.x}
            y={groups.core.y}
            group={groups.core.group}
            groupNames={Array.from({ length: 8 }, (_, c) => `cluster ${c}`)}
          />
          <Points name="border" x={groups.border.x} y={groups.border.y} colors={groups.border.ring} size={10} />
        </Plot>
        <Plot x={orderAxis} y={reachAxis} legend={false}>
          <Bars name="reachability" x={reach.x} y={reach.y} />
          <Handle {...figure.handle('eps', { label: 'ε', axis: 'y' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Spectral clustering ─────────────────────────────────────────────────────────────────────────────────────────

export function SpectralSpecimen() {
  const figure = useFigureState({ lengthscale: slider(0.03, 1, 0.15, { label: 'affinity lengthscale ℓ' }) })
  const { lengthscale } = figure
  const data = useMemo(() => circles(stream('lab/cluster/spectral'), { n: 200, noise: 0.05, factor: 0.5 }), [])
  const cols = useMemo(() => columns(data.x), [data])
  const km = useMemo(() => kmeans({ k: 2 }).fit(dataset(data.x), { stream: stream('lab/spectral/km') }), [data])
  const sc = useMemo(
    () =>
      spectralClustering({ k: 2, affinity: { kind: 'rbf', lengthscale } }).fit(dataset(data.x), {
        stream: stream('lab/spectral'),
      }),
    [data, lengthscale],
  )
  const kmLabels = useMemo(() => toFlat(km.decide(data.x)), [km, data])
  const scLabels = useMemo(() => toFlat(sc.labels), [sc])
  const ax = useAxis({ label: 'x₀ (k-means)' })
  const ay = useAxis({ label: 'x₁', equal: ax })
  const bx = useAxis({ label: 'x₀ (spectral)' })
  const by = useAxis({ label: 'x₁', equal: bx })
  return (
    <Figure
      title="Spectral clustering against k-means"
      purpose="k-means cuts the plane into convex cells, so it cannot separate nested rings; spectral clustering runs k-means on the top eigenvectors of the normalised affinity, where the rings are far apart."
      state={figure}
      defaultSize="L"
      readouts={
        <>
          <Readout
            label="eigenvalues"
            value={toFlat(sc.eigenvalues)
              .map((v) => v.toFixed(4))
              .join(', ')}
          />
          <Readout label="graph components" value={sc.components} />
        </>
      }
      caption="With a short lengthscale the affinity links only neighbours along each ring, the second eigenvalue is close to 1 and spectral clustering finds the rings. Lengthen ℓ and the affinity links across the gap; the partition degrades towards k-means'."
    >
      <Plots cols={2}>
        <Plot x={ax} y={ay} legend={false}>
          <Points name="points" x={cols.x0} y={cols.x1} group={kmLabels} groupNames={['cluster 0', 'cluster 1']} />
        </Plot>
        <Plot x={bx} y={by} legend={false}>
          <Points name="points" x={cols.x0} y={cols.x1} group={scLabels} groupNames={['cluster 0', 'cluster 1']} />
        </Plot>
      </Plots>
    </Figure>
  )
}
