import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

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
  const r = rng(7)
  const q = Array.from({ length: D }, () => r.normal())
  const k = q.map((v) => v + 0.7 * r.normal())
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
  const [method, setMethod] = useState<Method>('rope')
  const [m2, setM2] = useState(700)
  const [logBase, setLogBase] = useState(4)
  const base = 10 ** logBase

  const r = useMemo(() => {
    const w = freqs(base)
    const a = scores(method, M1, w)
    const b = scores(method, m2, w)
    const gap = Math.max(...a.map((v, i) => Math.abs(v - b[i])))
    return { a, b, gap }
  }, [method, m2, base])

  const series: XYSeries[] = [
    { name: `query at m = ${M1}`, type: 'line', x: OFFSETS, y: r.a, slot: 0 },
    { name: `query at m = ${m2}`, type: 'line', x: OFFSETS, y: r.b, slot: 1, dashed: true },
  ]

  return (
    <Interactive
      title="Score against offset: rotary versus added encodings"
      caption="One fixed query vector and one fixed key vector (d = 64) are placed at positions m and n = m + offset, and the score q·k/√d is plotted against the offset. With RoPE the two curves coincide for every choice of m: the score depends only on n − m. With a sinusoid added to the query and key vectors (identity projections), moving the query to another absolute position changes the curve."
      controls={
        <>
          <ParamChoice label="encoding" value={method} onChange={setMethod} options={METHODS} />
          <ParamSlider label="second query position m" value={m2} onChange={setM2} min={100} max={4000} step={10} />
          <ParamSlider
            label="base"
            value={logBase}
            onChange={setLogBase}
            min={1}
            max={5}
            step={0.1}
            format={(v) => Math.round(10 ** v).toLocaleString()}
          />
        </>
      }
      readout={<Readout label="largest gap between the curves" value={formatNumber(r.gap)} />}
    >
      <XYChart series={series} xLabel="offset n − m" yLabel="score" height={300} />
    </Interactive>
  )
}
