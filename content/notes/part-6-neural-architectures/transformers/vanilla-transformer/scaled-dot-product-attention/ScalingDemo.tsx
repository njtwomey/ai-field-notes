import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

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
  const r = stream(seed)
  const q = Array.from({ length: N }, () => Float64Array.from({ length: MAX_D }, () => normal(r)))
  const k = Array.from({ length: N }, () => Float64Array.from({ length: MAX_D }, () => normal(r)))
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
  const state = useFigureState({
    logD: int(6, { min: 0, max: MAX_LOG_D, step: 1, label: 'dimension d', format: (v) => String(2 ** v) }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const sample = useMemo(() => draw(state.seed), [state.seed])
  const d = 2 ** state.logD

  const scaledSd = useMemo(() => sample.rawSd.map((s, i) => s / Math.sqrt(DIMS[i])), [sample])

  const row = useMemo(() => {
    const s = Array.from(sample.logits.get(d)!.subarray(0, N))
    const raw = softmax(s)
    const scaled = softmax(s.map((x) => x / Math.sqrt(d)))
    const i = DIMS.indexOf(d)
    return { raw, scaled, rawSd: sample.rawSd[i], scaledSd: scaledSd[i] }
  }, [sample, scaledSd, d])

  const { set } = state
  const handles = useMemo(
    () => [{ kind: 'x' as const, at: d, label: 'd', onDrag: (x: number) => set('logD', Math.log2(Math.max(x, 1))) }],
    [d, set],
  )

  const rawSeries = useMemo(
    () =>
      [
        { name: '√d', x: SQRT_CURVE_X, y: SQRT_CURVE_Y, muted: true, dashed: true },
        { name: 'std of QKᵀ', x: DIMS, y: sample.rawSd, slot: 0 },
        { name: `d = ${d}`, x: [d], y: [row.rawSd], emphasis: true },
      ] as const,
    [sample, row, d],
  )
  const scaledSeries = useMemo(
    () =>
      [
        { name: '1', x: [0, MAX_D], y: [1, 1], muted: true, dashed: true },
        { name: 'std of QKᵀ/√d', x: DIMS, y: scaledSd, slot: 1 },
        { name: `d = ${d}`, x: [d], y: [row.scaledSd], emphasis: true },
      ] as const,
    [scaledSd, row, d],
  )
  const rawBars = useMemo(() => [{ name: 'weight, raw', x: KEY_INDEX, y: row.raw, slot: 0 }] as const, [row])
  const scaledBars = useMemo(() => [{ name: 'weight, scaled', x: KEY_INDEX, y: row.scaled, slot: 1 }] as const, [row])

  const panel = 'text-center text-xs text-muted-foreground'
  const xAxis = useAxis({ label: 'd', range: [0, MAX_D] })
  const yAxis = useAxis({ label: 'std of logits', range: [0, 40] })
  const xAxis2 = useAxis({ label: 'd', range: [0, MAX_D] })
  const yAxis2 = useAxis({ label: 'std of logits', range: [0, 2] })
  const xAxis3 = useAxis({ label: 'key', range: [0.5, N + 0.5] })
  const yAxis3 = useAxis({ label: 'weight', range: [0, 1] })
  const xAxis4 = useAxis({ label: 'key', range: [0.5, N + 0.5] })
  const yAxis4 = useAxis({ label: 'weight', range: [0, 1] })
  return (
    <Figure
      title="Why divide by √d"
      state={state}
      caption={
        <>
          Q and K are 16 × d matrices with independent standard normal entries. Top: the standard deviation of the 256
          entries of QKᵀ at each d, raw (left, dashed √d) and divided by √d (right, dashed 1). Bottom: the softmax of
          the first row of logits over the 16 keys at the chosen d. Change d with the slider or drag the vertical line
          in either top chart; draw new matrices to see the scatter across seeds.
        </>
      }

      readouts={
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
          <Plot x={xAxis} y={yAxis} height={230}>
            <Curve {...rawSeries[0]} />
            <Points {...rawSeries[1]} />
            <Points {...rawSeries[2]} />
            {(handles ?? []).map((h, i) => (
              <Handle key={i} {...h} />
            ))}
          </Plot>
        </div>
        <div>
          <p className={panel}>Scaled logits QKᵀ/√d</p>
          <Plot x={xAxis2} y={yAxis2} height={230}>
            <Curve {...scaledSeries[0]} />
            <Points {...scaledSeries[1]} />
            <Points {...scaledSeries[2]} />
            {(handles ?? []).map((h, i) => (
              <Handle key={i} {...h} />
            ))}
          </Plot>
        </div>
        <div>
          <p className={panel}>Softmax of row 1, raw</p>
          <Plot x={xAxis3} y={yAxis3} height={200}>
            <Bars {...rawBars[0]} />
          </Plot>
        </div>
        <div>
          <p className={panel}>Softmax of row 1, scaled</p>
          <Plot x={xAxis4} y={yAxis4} height={200}>
            <Bars {...scaledBars[0]} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
