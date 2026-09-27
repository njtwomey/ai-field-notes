import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber } from '@/components/viz'
import { rng } from '@/lib/math'

const N = 60
const D = 1000
const BINS = 40
const RANGE: [number, number] = [0, 2]

const MATRICES = [
  { value: 'gauss', label: 'Gaussian' },
  { value: 'sign', label: 'random signs' },
  { value: 'achlioptas', label: 'Achlioptas (2/3 zeros)' },
  { value: 'very-sparse', label: 'very sparse (s = √D)' },
] as const
type MatrixKind = (typeof MATRICES)[number]['value']

const DATA = [
  { value: 'dense', label: 'dense points' },
  { value: 'sparse', label: 'sparse points (5 non-zeros)' },
] as const
type DataKind = (typeof DATA)[number]['value']

/** Fourth moment of one matrix entry, scaled to unit variance: it sets the extra variance on sparse vectors. */
const KURTOSIS: Record<MatrixKind, number> = { gauss: 3, sign: 1, achlioptas: 3, 'very-sparse': Math.sqrt(D) }

function points(kind: DataKind): number[][] {
  const r = rng(11)
  return Array.from({ length: N }, () => {
    if (kind === 'dense') return Array.from({ length: D }, () => r.normal())
    const x = new Array<number>(D).fill(0)
    for (let m = 0; m < 5; m++) x[Math.floor(r.uniform() * D)] = r.normal()
    return x
  })
}

/** Entries of a D × k matrix with mean 0 and variance 1 (divided by √k when applied), stored as sparse columns. */
function matrix(kind: MatrixKind, k: number, seed: number): { rows: number[]; vals: number[] }[] {
  const r = rng(seed)
  const s = kind === 'achlioptas' ? 3 : kind === 'very-sparse' ? Math.sqrt(D) : 1
  return Array.from({ length: k }, () => {
    const rows: number[] = []
    const vals: number[] = []
    for (let j = 0; j < D; j++) {
      if (kind === 'gauss') {
        rows.push(j)
        vals.push(r.normal())
      } else {
        const u = r.uniform()
        // Non-zero with probability 1/s, then ±√s with equal probability: mean 0, variance 1, fourth moment s.
        if (u < 1 / s) {
          rows.push(j)
          vals.push(u < 1 / (2 * s) ? Math.sqrt(s) : -Math.sqrt(s))
        }
      }
    }
    return { rows, vals }
  })
}

/** The distortion ε that the Johnson–Lindenstrauss bound k ≥ 4 ln n / (ε²/2 − ε³/3) guarantees at this k. */
function jlEpsilon(k: number, n: number): number {
  // At ε = 1 the bound needs k ≥ 24 ln n; below that it guarantees nothing.
  if (k < 24 * Math.log(n)) return Infinity
  let lo = 0
  let hi = 1
  for (let it = 0; it < 60; it++) {
    const e = (lo + hi) / 2
    if ((4 * Math.log(n)) / ((e * e) / 2 - (e * e * e) / 3) > k) lo = e
    else hi = e
  }
  return hi
}

export function DistortionHistogram() {
  const [k, setK] = useState(100)
  const [kind, setKind] = useState<MatrixKind>('gauss')
  const [dataKind, setDataKind] = useState<DataKind>('dense')
  const x = useMemo(() => points(dataKind), [dataKind])
  const result = useMemo(() => {
    const cols = matrix(kind, k, 5)
    const y = x.map((p) => cols.map((c) => c.rows.reduce((s, j, m) => s + c.vals[m] * p[j], 0) / Math.sqrt(k)))
    const ratios: number[] = []
    let predicted = 0
    for (let i = 0; i < N; i++)
      for (let j = i + 1; j < N; j++) {
        let before = 0
        let fourth = 0
        for (let m = 0; m < D; m++) {
          const u = x[i][m] - x[j][m]
          before += u * u
          fourth += u ** 4
        }
        let after = 0
        for (let m = 0; m < k; m++) after += (y[i][m] - y[j][m]) ** 2
        ratios.push(after / before)
        // Var(‖Ru‖²/‖u‖²) = (2 + (κ − 3) Σu⁴/‖u‖⁴) / k for entries with fourth moment κ.
        predicted += (2 + (KURTOSIS[kind] - 3) * (fourth / (before * before))) / k
      }
    const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length
    const sd = Math.sqrt(ratios.reduce((s, v) => s + (v - mean) ** 2, 0) / ratios.length)
    const worst = Math.max(...ratios.map((v) => Math.abs(v - 1)))
    const width = (RANGE[1] - RANGE[0]) / BINS
    const counts = new Array<number>(BINS).fill(0)
    for (const v of ratios) counts[Math.min(BINS - 1, Math.max(0, Math.floor((v - RANGE[0]) / width)))]++
    return {
      centres: counts.map((_, b) => RANGE[0] + (b + 0.5) * width),
      density: counts.map((c) => c / (ratios.length * width)),
      sd,
      predicted: Math.sqrt(predicted / ratios.length),
      worst,
    }
  }, [x, kind, k])

  return (
    <Interactive
      title="How much a random projection distorts squared distances"
      caption="60 points in 1000 dimensions are projected to k dimensions. The histogram shows, for all 1770 pairs, the squared distance after projection divided by the squared distance before. The spread shrinks as 1/√k for every matrix on dense points. On sparse points, very sparse matrices often miss the few coordinates where two points differ, and the spread grows, while random signs give a smaller spread than Gaussian entries."
      controls={
        <>
          <ParamSlider label="k (target dimension)" value={k} onChange={setK} min={5} max={400} step={5} />
          <ParamChoice label="matrix" value={kind} onChange={setKind} options={MATRICES} />
          <ParamChoice label="data" value={dataKind} onChange={setDataKind} options={DATA} />
        </>
      }
      readout={
        <>
          <Readout label="spread of ratios (sd)" value={formatNumber(result.sd)} />
          <Readout label="predicted sd" value={formatNumber(result.predicted)} />
          <Readout label="worst pair |ratio − 1|" value={formatNumber(result.worst)} />
          <Readout
            label="JL guarantee ε at this k"
            value={Number.isFinite(jlEpsilon(k, N)) ? formatNumber(jlEpsilon(k, N)) : 'none (k < 99)'}
          />
        </>
      }
    >
      <XYChart
        height={300}
        xRange={RANGE}
        xLabel="‖R(x − x′)‖² / ‖x − x′‖²"
        yLabel="density"
        series={[{ name: 'pairs', type: 'bar', x: result.centres, y: result.density, slot: 0 }]}
      />
    </Interactive>
  )
}
