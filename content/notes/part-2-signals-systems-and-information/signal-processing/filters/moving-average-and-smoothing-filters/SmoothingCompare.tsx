import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { convolve } from 'aifn/foundation/convolution'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { db, response as freqz } from '../_shared/design'
import { normal, stream } from 'aifn/foundation/random'

const N = 300

/**
 * Savitzky–Golay smoothing coefficients: fit a polynomial of the given degree to 2m + 1 points by least squares and
 * take its value at the centre, h = e₀ᵀ (VᵀV)⁻¹ Vᵀ with V_ij = (i − m)^j.
 */
function savgol(m: number, degree: number): number[] {
  const p = degree + 1
  const rows = Array.from({ length: 2 * m + 1 }, (_, i) => Array.from({ length: p }, (_, j) => (i - m) ** j))
  // Normal matrix G = VᵀV, then solve G c = e₀ for c; the coefficients are V c.
  const G = Array.from({ length: p }, (_, j) =>
    Array.from({ length: p }, (_, k) => rows.reduce((s, r) => s + r[j] * r[k], 0)),
  )
  const rhs = Array.from({ length: p }, (_, j) => (j === 0 ? 1 : 0))
  // Gaussian elimination with partial pivoting.
  const A = G.map((row, i) => [...row, rhs[i]])
  for (let col = 0; col < p; col++) {
    let pivot = col
    for (let r = col + 1; r < p; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r
    ;[A[col], A[pivot]] = [A[pivot], A[col]]
    for (let r = 0; r < p; r++) {
      if (r === col) continue
      const f = A[r][col] / A[col][col]
      for (let k = col; k <= p; k++) A[r][k] -= f * A[col][k]
    }
  }
  const c = A.map((row, i) => row[p] / row[i])
  return rows.map((r) => r.reduce((s, v, j) => s + v * c[j], 0))
}

/** Moving average against Savitzky–Golay on a narrow peak and a broad bump, with their frequency responses. */
export function SmoothingCompare() {
  const state = useFigureState({
    half: int(7, {
      min: 1,
      max: 25,
      step: 1,
      label: 'half-width m (window 2m + 1)',
      format: (v) => `${v}  (${2 * v + 1} points)`,
    }),
    degree: int(2, { min: 2, max: 6, step: 2, label: 'Savitzky–Golay degree', format: (v) => String(v) }),
  })
  const L = 2 * state.half + 1

  const r = useMemo(() => {
    const g = stream(8)
    const clean = Array.from(
      { length: N },
      (_, n) => Math.exp(-0.5 * ((n - 90) / 4) ** 2) + 0.6 * Math.exp(-0.5 * ((n - 200) / 25) ** 2),
    )
    const x = clean.map((v) => v + 0.08 * normal(g))
    const ma = new Array<number>(L).fill(1 / L)
    const sg = savgol(state.half, Math.min(state.degree, L - 1))
    // Centre the convolution so outputs line up with inputs (both filters are symmetric).
    const apply = (h: number[]) =>
      toFlat(convolve(fromData(Float64Array.from(x)), fromData(Float64Array.from(h))) as Tensor).slice(
        state.half,
        state.half + N,
      )
    const resp = (h: number[]) => freqz(h, [1], 512)
    const rm = resp(ma)
    const rs = resp(sg)
    return {
      x,
      clean,
      ma: apply(ma),
      sg: apply(sg),
      w: rm.omega.map((v) => v / Math.PI),
      maDb: rm.magnitude.map((m) => db(m, -60)),
      sgDb: rs.magnitude.map((m) => db(m, -60)),
      maPeak: Math.max(...apply(ma).slice(70, 110)),
      sgPeak: Math.max(...apply(sg).slice(70, 110)),
      noiseGain: [ma, sg].map((h) => h.reduce((s, v) => s + v * v, 0)),
    }
  }, [state.half, state.degree, L])

  const n = Array.from({ length: N }, (_, i) => i)
  const time = [
    { name: 'noisy input', x: n, y: r.x, muted: true },
    { name: `moving average (${L} points)`, x: n, y: r.ma, slot: 2 },
    { name: `Savitzky–Golay (${L} points, degree ${state.degree})`, x: n, y: r.sg, slot: 0 },
    { name: 'clean', x: n, y: r.clean, slot: 1, dashed: true },
  ] as const
  const freq = [
    { name: 'moving average', x: r.w, y: r.maDb, slot: 2 },
    { name: 'Savitzky–Golay', x: r.w, y: r.sgDb, slot: 0 },
  ] as const

  const xAxis = useAxis({ label: 'n', range: [0, N - 1] })
  const yAxis = useAxis({ label: 'amplitude', range: [-0.3, 1.2] })
  const xAxis2 = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'magnitude (dB)', range: [-60, 5] })
  return (
    <Figure
      title="Moving average against Savitzky–Golay"
      state={state}
      caption="A narrow peak (height 1) and a broad bump in noise, smoothed with the same window length. The moving average flattens the narrow peak; Savitzky–Golay fits a local polynomial and keeps the peak's height much better, at the price of passing more high-frequency noise. The frequency responses show why: Savitzky–Golay has a flatter passband and a less attenuating stopband. Both filters are symmetric, so neither shifts the features."

      readouts={
        <>
          <Readout label="peak height, moving average" value={formatNumber(r.maPeak)} />
          <Readout label="peak height, Savitzky–Golay" value={formatNumber(r.sgPeak)} />
          <Readout
            label="white-noise variance gain Σh² (MA, SG)"
            value={r.noiseGain.map((v) => v.toFixed(3)).join(', ')}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...time[0]} />
        <Curve {...time[1]} />
        <Curve {...time[2]} />
        <Curve {...time[3]} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={200}>
        <Curve {...freq[0]} />
        <Curve {...freq[1]} />
      </Plot>
    </Figure>
  )
}
