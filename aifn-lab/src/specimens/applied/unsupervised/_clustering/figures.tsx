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
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type Vec2, type XYSeries } from '@lab/viz'
import { DendrogramView, formatValue } from '@lab/views'

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
  const [step, setStep] = useState(1)
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
  const series: XYSeries[] = [
    {
      name: 'points',
      type: 'scatter',
      x: cols.x0,
      y: cols.x1,
      group: toFlat(state.labels),
      groupNames: ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3'],
    },
    ...[0, 1, 2, 3].map((j) => ({
      name: `path of centroid ${j}`,
      type: 'line' as const,
      x: paths.map((p) => p[j][0]),
      y: paths.map((p) => p[j][1]),
      slot: j,
      showPoints: true,
    })),
    { name: 'centroids', type: 'scatter', x: paths[k].map((c) => c[0]), y: paths[k].map((c) => c[1]), emphasis: true },
  ]
  return (
    <Figure
      title="Lloyd's algorithm from centroids you place"
      description="Each step moves every centroid to the mean of its points, then reassigns every point to its nearest centroid; the inertia never rises, but where it settles depends on the start."
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
      caption="Drag the four starting centroids (the first points of each path). The start decides which local optimum Lloyd's algorithm reaches: place two starts inside one blob and the run ends with two centroids sharing it and one straddling two blobs, at a higher inertia than the best k-means++ restart."
    >
      <XYChart
        series={series}
        aspect="equal"
        xLabel="x₀"
        yLabel="x₁"
        handles={start.map((p, j) => ({
          kind: 'point' as const,
          at: p,
          label: `start ${j}`,
          onDrag: (q: Vec2) => setStart((s) => s.map((c, i) => (i === j ? q : c))),
        }))}
      />
    </Figure>
  )
}

// ── Gaussian mixtures by EM ─────────────────────────────────────────────────────────────────────────────────────

export function MixtureSpecimen() {
  const data = useMemo(() => blobData(), [])
  const [covariance, setCovariance] = useState<CovarianceType>('full')
  const [step, setStep] = useState(3)
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
  const resp = toRows(state.responsibilities)
  const owner = resp.map((r) => r.indexOf(Math.max(...r)))
  const means = toRows(state.means)
  const covs = toFlat(state.covariances)
  const ellipses: XYSeries[] = means.flatMap((m, j) =>
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
      return {
        name: `component ${j}`,
        type: 'line' as const,
        x: e.map((p) => p[0]),
        y: e.map((p) => p[1]),
        slot: j,
        dashed: sd === 2,
      }
    }),
  )
  const ll = toFlat(run.series.ll)
  return (
    <Figure
      title="EM for a Gaussian mixture"
      description="The E-step gives each point a responsibility for each component; the M-step refits every weight, mean and covariance to those soft assignments."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · model">
            <Select
              label="covariances"
              value={covariance}
              onChange={setCovariance}
              options={['full', 'diagonal', 'spherical']}
            />
          </ControlRow>
          <ControlRow label="2 · EM steps">
            <Player value={k} onChange={setStep} count={run.steps.length} label="step" />
          </ControlRow>
        </>
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
      <Subplots rows={2} heightRatios={[3, 1]}>
        <Panel>
          <XYChart
            aspect="equal"
            xLabel="x₀"
            yLabel="x₁"
            series={[
              {
                name: 'points',
                type: 'scatter',
                x: cols.x0,
                y: cols.x1,
                group: owner,
                groupNames: ['component 0', 'component 1', 'component 2', 'component 3'],
              },
              ...ellipses,
            ]}
            legend={false}
            rescaleOnChange={false}
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="step"
            yLabel="log-likelihood"
            series={[
              { name: 'mean log-likelihood', type: 'line', x: Array.from(run.index), y: ll },
              { name: 'now', type: 'scatter', x: [run.index[k]], y: [ll[k]], emphasis: true },
            ]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── Agglomerative clustering ────────────────────────────────────────────────────────────────────────────────────

export function DendrogramSpecimen() {
  const [method, setMethod] = useState<Linkage>('ward')
  const [dataset, setDataset] = useState<'blobs' | 'moons'>('blobs')
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
  // Start the cut between the last two merges, so it shows two or three clusters.
  const [cut, setCut] = useState<number | null>(null)
  const at = cut ?? (heights[heights.length - 2] + heights[heights.length - 3]) / 2
  return (
    <DendrogramView
      title="The dendrogram and the cut"
      description="Agglomerative clustering merges the two closest clusters at each step; cutting the tree at a height undoes every merge above it."
      merges={Z}
      cut={at}
      onCut={setCut}
      points={data.x}
      controls={
        <>
          <Select
            label="linkage"
            value={method}
            onChange={(m) => {
              setMethod(m)
              setCut(null)
            }}
            options={['single', 'complete', 'average', 'ward']}
          />
          <Select
            label="data"
            value={dataset}
            onChange={(d) => {
              setDataset(d)
              setCut(null)
            }}
            options={['blobs', 'moons']}
          />
        </>
      }
      caption="Move the cut: clusters below it take one colour in the tree and among the points. Single linkage chains along the moons and recovers them; Ward and complete linkage prefer compact groups and split each moon."
    />
  )
}

// ── DBSCAN and OPTICS ───────────────────────────────────────────────────────────────────────────────────────────

export function DbscanSpecimen() {
  const data = useMemo(() => moons(stream('lab/cluster/dbscan'), { n: 200, noise: 0.08 }), [])
  const [eps, setEps] = useState(0.15)
  const [minSamples, setMinSamples] = useState(5)
  const model = useMemo(() => dbscan({ eps, minSamples }).fit(dataset(data.x)), [data, eps, minSamples])
  const order = useMemo(() => optics({ minSamples }).fit(dataset(data.x)), [data, minSamples])
  const cols = useMemo(() => columns(data.x), [data])
  const labels = toFlat(model.labels)
  const roles = toFlat(model.roles)
  const pick = (f: (i: number) => boolean) => cols.x0.map((_, i) => i).filter(f)
  const core = pick((i) => roles[i] === CORE)
  const border = pick((i) => roles[i] === BORDER)
  const noise = pick((i) => labels[i] < 0)
  const reach = toFlat(order.reachability)
  const ordering = toFlat(order.ordering)
  const bars = ordering.map((p) => (Number.isFinite(reach[p]) ? reach[p] : NaN))
  const top = Math.max(...bars.filter(Number.isFinite)) * 1.1
  const counts = { core: core.length, border: border.length, noise: noise.length }
  return (
    <Figure
      title="DBSCAN's ε and OPTICS reachability"
      description="A core point has at least minSamples points within ε; clusters are chains of core points plus the border points they reach. OPTICS orders the points so that every ε at once can be read off one reachability plot."
      defaultSize="L"
      controls={
        <>
          <Slider label="ε" value={eps} min={0.03} max={0.5} onChange={setEps} />
          <Slider label="minSamples" value={minSamples} min={2} max={15} step={1} onChange={setMinSamples} />
        </>
      }
      readouts={
        <>
          <Readout label="clusters" value={model.clusters} />
          <Readout label="core" value={counts.core} />
          <Readout label="border" value={counts.border} />
          <Readout label="noise" value={counts.noise} />
        </>
      }
      caption="Drag the ε line on the reachability plot (or use the slider): each valley below the line is a cluster, and bars above it start a new cluster or are noise. Core points are filled by cluster, border points ringed, noise grey. Too small an ε shatters the moons; too large merges them."
    >
      <Subplots rows={2} heightRatios={[3, 2]}>
        <Panel>
          <XYChart
            aspect="equal"
            xLabel="x₀"
            yLabel="x₁"
            series={[
              {
                name: 'noise',
                type: 'scatter',
                x: noise.map((i) => cols.x0[i]),
                y: noise.map((i) => cols.x1[i]),
                muted: true,
              },
              {
                name: 'core',
                type: 'scatter',
                x: core.map((i) => cols.x0[i]),
                y: core.map((i) => cols.x1[i]),
                group: core.map((i) => labels[i] % 8),
                groupNames: Array.from({ length: 8 }, (_, c) => `cluster ${c}`),
              },
              {
                name: 'border',
                type: 'scatter',
                x: border.map((i) => cols.x0[i]),
                y: border.map((i) => cols.x1[i]),
                emphasis: true,
              },
            ]}
            legend={false}
            rescaleOnChange={false}
          />
        </Panel>
        <Panel>
          <XYChart
            xLabel="OPTICS order"
            yLabel="reachability"
            yRange={[0, top]}
            series={[
              {
                name: 'reachability',
                type: 'bar',
                x: ordering.map((_, i) => i),
                y: bars.map((b) => (Number.isFinite(b) ? b : top)),
              },
            ]}
            handles={[{ kind: 'y', at: eps, onDrag: (e) => setEps(Math.min(0.5, Math.max(0.03, e))), label: 'ε' }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── Spectral clustering ─────────────────────────────────────────────────────────────────────────────────────────

export function SpectralSpecimen() {
  const [lengthscale, setLengthscale] = useState(0.15)
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
  const scatter = (labels: Tensor): XYSeries[] => [
    {
      name: 'points',
      type: 'scatter',
      x: cols.x0,
      y: cols.x1,
      group: toFlat(labels),
      groupNames: ['cluster 0', 'cluster 1'],
    },
  ]
  return (
    <Figure
      title="Spectral clustering against k-means"
      description="k-means cuts the plane into convex cells, so it cannot separate nested rings; spectral clustering runs k-means on the top eigenvectors of the normalised affinity, where the rings are far apart."
      defaultSize="L"
      controls={
        <Slider label="affinity lengthscale ℓ" value={lengthscale} min={0.03} max={1} onChange={setLengthscale} />
      }
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
      <Subplots cols={2} sharey>
        <Panel>
          <XYChart
            aspect="equal"
            xLabel="x₀ (k-means)"
            yLabel="x₁"
            series={scatter(km.decide(data.x))}
            legend={false}
          />
        </Panel>
        <Panel>
          <XYChart aspect="equal" xLabel="x₀ (spectral)" series={scatter(sc.labels)} legend={false} />
        </Panel>
      </Subplots>
    </Figure>
  )
}
