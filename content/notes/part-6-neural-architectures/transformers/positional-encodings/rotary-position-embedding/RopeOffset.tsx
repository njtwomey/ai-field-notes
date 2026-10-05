import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const D = 64
const MAX_OFFSET = 48
const M1 = 100
const OFFSETS = Array.from({ length: 2 * MAX_OFFSET + 1 }, (_, i) => i - MAX_OFFSET)

type Method = 'rope' | 'added'
const METHODS = [
  { value: 'rope', label: 'RoPE (rotate q and k)' },
  { value: 'added', label: 'sinusoid added to x' },
] as const

/** A query and a correlated key, so the score at offset 0 is large: k = q + 0.7 · noise. */
function draw() {
  const r = stream(7)
  const q = Array.from({ length: D }, () => normal(r))
  const k = q.map((v) => v + 0.7 * normal(r))
  return { q, k }
}
const SAMPLE = draw()

const freqs = (base: number) => Array.from({ length: D / 2 }, (_, i) => base ** ((-2 * i) / D))

/** Rotate each pair (x_{2i}, x_{2i+1}) by the angle t · ω_i. */
function rotate(x: number[], t: number, w: number[]): number[] {
  const out = x.slice()
  w.forEach((wi, i) => {
    const c = Math.cos(t * wi)
    const s = Math.sin(t * wi)
    out[2 * i] = c * x[2 * i] - s * x[2 * i + 1]
    out[2 * i + 1] = s * x[2 * i] + c * x[2 * i + 1]
  })
  return out
}

/** Sinusoidal encoding at position t: (sin tω_i, cos tω_i) pairs. */
const sinusoid = (t: number, w: number[]) => w.flatMap((wi) => [Math.sin(t * wi), Math.cos(t * wi)])

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)

/** Attention score q_m · k_n / √d as a function of the offset n − m, for two absolute query positions. */
function scores(method: Method, m: number, w: number[]): number[] {
  const { q, k } = SAMPLE
  const pm = sinusoid(m, w)
  const qm = method === 'rope' ? rotate(q, m, w) : q.map((v, i) => v + pm[i])
  return OFFSETS.map((off) => {
    const n = m + off
    const pn = sinusoid(n, w)
    const kn = method === 'rope' ? rotate(k, n, w) : k.map((v, i) => v + pn[i])
    return dot(qm, kn) / Math.sqrt(D)
  })
}

export function RopeOffset() {
  const state = useFigureState({
    method: choice<Method>(METHODS, 'rope', { label: 'encoding' }),
    m2: int(700, { min: 100, max: 4000, step: 10, label: 'second query position m' }),
    logBase: float(4, {
      min: 1,
      max: 5,
      step: 0.1,
      label: 'base',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => Math.round(10 ** v).toLocaleString(),
    }),
  })
  const base = 10 ** state.logBase

  const r = useMemo(() => {
    const w = freqs(base)
    const a = scores(state.method, M1, w)
    const b = scores(state.method, state.m2, w)
    const gap = Math.max(...a.map((v, i) => Math.abs(v - b[i])))
    return { a, b, gap }
  }, [state.method, state.m2, base])

  const series = [
    { name: `query at m = ${M1}`, x: OFFSETS, y: r.a, slot: 0 },
    { name: `query at m = ${state.m2}`, x: OFFSETS, y: r.b, slot: 1, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'offset n − m', hold: 'union' })
  const yAxis = useAxis({ label: 'score', hold: 'union' })
  return (
    <Figure
      title="Score against offset: rotary versus added encodings"
      state={state}
      caption="One fixed query vector and one fixed key vector (d = 64) are placed at positions m and n = m + offset, and the score q·k/√d is plotted against the offset. With RoPE the two curves coincide for every choice of m: the score depends only on n − m. With a sinusoid added to the query and key vectors (identity projections), moving the query to another absolute position changes the curve."

      readouts={<Readout label="largest gap between the curves" value={formatNumber(r.gap)} />}
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
