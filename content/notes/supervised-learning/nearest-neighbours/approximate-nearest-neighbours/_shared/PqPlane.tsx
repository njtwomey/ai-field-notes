import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { eigSym } from '@/lib/math/mat2'
import { dist2, kmeans1d, nearest1d, type P } from './geometry'

const N = 400
const RANGE: [number, number] = [-5, 5]

const rotate = ([x, y]: P, phi: number): P => [
  x * Math.cos(phi) - y * Math.sin(phi),
  x * Math.sin(phi) + y * Math.cos(phi),
]

/** Four elongated clusters strung along a line at 35 degrees. */
function makeData(): P[] {
  const g = rng(11)
  const a = (35 * Math.PI) / 180
  const u: P = [Math.cos(a), Math.sin(a)]
  const v: P = [-u[1], u[0]]
  const offsets = [-2.7, -0.9, 0.9, 2.7]
  return Array.from({ length: N }, (): P => {
    const t = offsets[Math.floor(g.uniform() * 4)] + 0.3 * g.normal()
    const s = 0.8 * g.normal()
    return [t * u[0] + s * v[0], t * u[1] + s * v[1]]
  })
}

/**
 * Product quantisation with two one-dimensional sub-quantisers, after rotating the data by -phi. Returns the two
 * codebooks, each point's codes and its reconstruction in the original coordinates.
 */
function pq(data: P[], k: number, phi: number) {
  const z = data.map((p) => rotate(p, -phi))
  const books = [0, 1].map((m) =>
    kmeans1d(
      z.map((p) => p[m]),
      k,
    ),
  )
  const codes = z.map((p) => [nearest1d(books[0], p[0]), nearest1d(books[1], p[1])])
  const zhat = codes.map(([i, j]): P => [books[0][i], books[1][j]])
  return { books, codes, zhat, recon: zhat.map((p) => rotate(p, phi)) }
}

/** OPQ in the plane: start from the principal axes, then alternate PQ codebooks and the best rotation. */
function learnRotation(data: P[], k: number): number {
  const n = data.length
  const sxx = data.reduce((s, p) => s + p[0] * p[0], 0) / n
  const sxy = data.reduce((s, p) => s + p[0] * p[1], 0) / n
  const syy = data.reduce((s, p) => s + p[1] * p[1], 0) / n
  const [e] = eigSym(sxx, sxy, syy).vectors
  let phi = Math.atan2(e[1], e[0])
  for (let it = 0; it < 15; it++) {
    const { zhat } = pq(data, k, phi)
    // Orthogonal Procrustes in 2-D: the rotation taking the codewords zhat closest to the data.
    let dot = 0
    let cross = 0
    data.forEach((x, i) => {
      dot += x[0] * zhat[i][0] + x[1] * zhat[i][1]
      cross += x[1] * zhat[i][0] - x[0] * zhat[i][1]
    })
    phi = Math.atan2(cross, dot)
  }
  return phi
}

/** Product quantisation in the plane, with an optional learned rotation (OPQ), and asymmetric distances to a query. */
export function PqPlane({ rotateByDefault = false }: { rotateByDefault?: boolean }) {
  const kParam = useParam(4, { min: 2, max: 8, step: 1 })
  const k = kParam.value
  const [learn, setLearn] = useState(rotateByDefault)
  const [query, setQuery] = useState<P>([1.2, -1.6])
  const data = useMemo(() => makeData(), [])

  const model = useMemo(() => {
    const phi = learn ? learnRotation(data, k) : 0
    const q = pq(data, k, phi)
    const mse = q.recon.reduce((s, r, i) => s + dist2(r, data[i]), 0) / N
    const baseline = learn ? pq(data, k, 0) : q
    const mse0 = baseline.recon.reduce((s, r, i) => s + dist2(r, data[i]), 0) / N
    const grid = q.books[0].flatMap((a) => q.books[1].map((b) => rotate([a, b], phi)))
    const used = new Set(q.codes.map(([i, j]) => i * k + j)).size
    return { phi, ...q, mse, mse0, grid, used }
  }, [data, k, learn])

  // Asymmetric distance: the exact query against each point's reconstruction.
  const adc = model.recon.map((r) => dist2(r, query))
  const exact = data.map((p) => dist2(p, query))
  const argmin = (v: number[]) => v.reduce((b, x, i) => (x < v[b] ? i : b), 0)
  const trueNn = argmin(exact)
  const adcNn = argmin(adc)
  const truthRank = adc.filter((d) => d < adc[trueNn]).length + 1

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'data', type: 'scatter', x: data.map((p) => p[0]), y: data.map((p) => p[1]), slot: 0 },
      {
        name: `codewords (${k} × ${k})`,
        type: 'scatter',
        x: model.grid.map((p) => p[0]),
        y: model.grid.map((p) => p[1]),
        emphasis: true,
      },
    ],
    [data, model, k],
  )
  const marks: XYSeries[] = [
    { name: 'true nearest', type: 'scatter', x: [data[trueNn][0]], y: [data[trueNn][1]], slot: 1 },
    { name: 'nearest by ADC', type: 'scatter', x: [data[adcNn][0]], y: [data[adcNn][1]], slot: 2 },
  ]
  const segments = useMemo((): Segment[] => data.map((p, i) => ({ from: p, to: model.recon[i] })), [data, model])
  const handles: Handle[] = [{ kind: 'point', at: query, label: 'query', onDrag: (p) => setQuery(p) }]
  const bits = 2 * Math.log2(k)

  return (
    <Interactive
      title={learn ? 'Optimised product quantisation in the plane' : 'Product quantisation in the plane'}
      caption={`Each 2-D point is split into two 1-D sub-vectors, and each sub-vector is replaced by the nearest of k values learnt by k-means on that coordinate. The k × k codewords (diamonds) form a grid; thin lines join each point to its reconstruction. The clusters lie along a diagonal, so an axis-aligned grid wastes most codewords on empty space. Switch on the learned rotation to align the grid with the data (OPQ). Drag the query: the asymmetric distance compares the exact query with each reconstruction, and its nearest point can differ from the true nearest.`}
      controls={
        <>
          <ParamSlider label="codewords per sub-quantiser k" param={kParam} format={(v) => String(v)} withArrows />
          <ParamSwitch label="learned rotation (OPQ)" checked={learn} onChange={setLearn} />
        </>
      }
      readout={
        <>
          <Readout label="code size" value={`${formatNumber(bits)} bits`} />
          <Readout label="mean squared error" value={formatNumber(model.mse)} />
          {learn && <Readout label="without rotation" value={formatNumber(model.mse0)} />}
          {learn && <Readout label="rotation" value={`${formatNumber((model.phi * 180) / Math.PI)}°`} />}
          <Readout label="codewords used" value={`${model.used} of ${k * k}`} />
          <Readout label="ADC rank of true nearest" value={String(truthRank)} />
        </>
      }
    >
      <XYChart
        series={[...series, ...marks]}
        segments={segments}
        xRange={RANGE}
        yRange={RANGE}
        equalAspect
        xLabel="x₁"
        yLabel="x₂"
        handles={handles}
      />
    </Interactive>
  )
}
