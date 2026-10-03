import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

/** Rows of Q and K: n queries attend over n keys. */
const N = 16
const MAX_LOG_D = 10
const MAX_D = 2 ** MAX_LOG_D

/** Dimensions plotted: the small powers of two, then every multiple of 32 up to 1024. */
const DIMS = [1, 2, 4, 8, 16, ...Array.from({ length: MAX_D / 32 }, (_, i) => 32 * (i + 1))]

const KEY_INDEX = Array.from({ length: N }, (_, j) => j + 1)

type Draw = {
  /** Standard deviation of the n × n entries of QKᵀ, at each d in DIMS. */
  rawSd: number[]
  /** Logits QKᵀ at each d in DIMS, keyed by d. Row 0 feeds the softmax panels. */
  logits: Map<number, Float64Array>
}

function sd(xs: ArrayLike<number>) {
  let mean = 0
  for (let i = 0; i < xs.length; i++) mean += xs[i]
  mean /= xs.length
  let v = 0
  for (let i = 0; i < xs.length; i++) v += (xs[i] - mean) ** 2
  return Math.sqrt(v / xs.length)
}

function softmax(xs: number[]) {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / z)
}

/**
 * Q, K ∈ ℝ^{n × 1024} with independent N(0, 1) entries. The logits for dimension d use the first d columns, so the
 * running sum over columns gives QKᵀ for every d in one pass (n² · 1024 multiplications).
 */
function draw(seed: number): Draw {
  const r = rng(seed)
  const q = Array.from({ length: N }, () => Float64Array.from({ length: MAX_D }, () => r.normal()))
  const k = Array.from({ length: N }, () => Float64Array.from({ length: MAX_D }, () => r.normal()))
  const acc = new Float64Array(N * N)
  const wanted = new Set(DIMS)
  const logits = new Map<number, Float64Array>()
  for (let c = 0; c < MAX_D; c++) {
    for (let i = 0; i < N; i++) {
      const qi = q[i][c]
      for (let j = 0; j < N; j++) acc[i * N + j] += qi * k[j][c]
    }
    if (wanted.has(c + 1)) logits.set(c + 1, acc.slice())
  }
  return { rawSd: DIMS.map((d) => sd(logits.get(d)!)), logits }
}

const SQRT_CURVE_X = Array.from({ length: 257 }, (_, i) => (i * MAX_D) / 256)
const SQRT_CURVE_Y = SQRT_CURVE_X.map(Math.sqrt)

/** Spread of the logits QKᵀ against the head dimension, raw and divided by √d, and the softmax of one row. */
export function ScalingDemo() {
  const logD = useParam(6, { min: 0, max: MAX_LOG_D, step: 1 })
  const [seed, setSeed] = useState(1)
  const sample = useMemo(() => draw(seed), [seed])
  const d = 2 ** logD.value

  const scaledSd = useMemo(() => sample.rawSd.map((s, i) => s / Math.sqrt(DIMS[i])), [sample])

  const row = useMemo(() => {
    const s = Array.from(sample.logits.get(d)!.subarray(0, N))
    const raw = softmax(s)
    const scaled = softmax(s.map((x) => x / Math.sqrt(d)))
    const i = DIMS.indexOf(d)
    return { raw, scaled, rawSd: sample.rawSd[i], scaledSd: scaledSd[i] }
  }, [sample, scaledSd, d])

  const setLogD = logD.set
  const handles = useMemo(
    () => [{ kind: 'x' as const, at: d, label: 'd', onDrag: (x: number) => setLogD(Math.log2(Math.max(x, 1))) }],
    [d, setLogD],
  )

  const rawSeries = useMemo<XYSeries[]>(
    () => [
      { name: '√d', type: 'line', x: SQRT_CURVE_X, y: SQRT_CURVE_Y, muted: true, dashed: true },
      { name: 'std of QKᵀ', type: 'scatter', x: DIMS, y: sample.rawSd, slot: 0 },
      { name: `d = ${d}`, type: 'scatter', x: [d], y: [row.rawSd], emphasis: true },
    ],
    [sample, row, d],
  )
  const scaledSeries = useMemo<XYSeries[]>(
    () => [
      { name: '1', type: 'line', x: [0, MAX_D], y: [1, 1], muted: true, dashed: true },
      { name: 'std of QKᵀ/√d', type: 'scatter', x: DIMS, y: scaledSd, slot: 1 },
      { name: `d = ${d}`, type: 'scatter', x: [d], y: [row.scaledSd], emphasis: true },
    ],
    [scaledSd, row, d],
  )
  const rawBars = useMemo<XYSeries[]>(
    () => [{ name: 'weight, raw', type: 'bar', x: KEY_INDEX, y: row.raw, slot: 0 }],
    [row],
  )
  const scaledBars = useMemo<XYSeries[]>(
    () => [{ name: 'weight, scaled', type: 'bar', x: KEY_INDEX, y: row.scaled, slot: 1 }],
    [row],
  )

  const panel = 'text-center text-xs text-muted-foreground'
  return (
    <Interactive
      title="Why divide by √d"
      caption={
        <>
          Q and K are 16 × d matrices with independent standard normal entries. Top: the standard deviation of the 256
          entries of QKᵀ at each d, raw (left, dashed √d) and divided by √d (right, dashed 1). Bottom: the softmax of
          the first row of logits over the 16 keys at the chosen d. Change d with the slider or drag the vertical line
          in either top chart; draw new matrices to see the scatter across seeds.
        </>
      }
      controls={
        <>
          <ParamSlider label="dimension d" param={logD} format={(v) => String(2 ** v)} withArrows />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New Q and K</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="std of logits, raw" value={formatNumber(row.rawSd)} />
          <Readout label="std of logits, scaled" value={formatNumber(row.scaledSd)} />
          <Readout label="largest weight, raw" value={formatNumber(Math.max(...row.raw))} />
          <Readout label="largest weight, scaled" value={formatNumber(Math.max(...row.scaled))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2">
        <div>
          <p className={panel}>Raw logits QKᵀ</p>
          <XYChart
            series={rawSeries}
            xLabel="d"
            yLabel="std of logits"
            xRange={[0, MAX_D]}
            yRange={[0, 40]}
            handles={handles}
            height={230}
          />
        </div>
        <div>
          <p className={panel}>Scaled logits QKᵀ/√d</p>
          <XYChart
            series={scaledSeries}
            xLabel="d"
            yLabel="std of logits"
            xRange={[0, MAX_D]}
            yRange={[0, 2]}
            handles={handles}
            height={230}
          />
        </div>
        <div>
          <p className={panel}>Softmax of row 1, raw</p>
          <XYChart series={rawBars} xLabel="key" yLabel="weight" xRange={[0.5, N + 0.5]} yRange={[0, 1]} height={200} />
        </div>
        <div>
          <p className={panel}>Softmax of row 1, scaled</p>
          <XYChart
            series={scaledBars}
            xLabel="key"
            yLabel="weight"
            xRange={[0.5, N + 0.5]}
            yRange={[0, 1]}
            height={200}
          />
        </div>
      </div>
    </Interactive>
  )
}
