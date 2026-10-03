import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

const D = 64
const BASE = 10000
const M1 = 256
const MAX_R = 128
const DISTANCES = Array.from({ length: MAX_R + 1 }, (_, r) => r)
const GRID = 32
const POSITIONS = Array.from({ length: GRID }, (_, i) => i)

type Scheme = 'none' | 'sinusoid' | 'shaw' | 't5' | 'rope' | 'xpos' | 'alibi'
const SCHEMES = [
  { value: 'none', label: 'none' },
  { value: 'sinusoid', label: 'sinusoid' },
  { value: 'shaw', label: 'Shaw' },
  { value: 't5', label: 'T5' },
  { value: 'rope', label: 'RoPE' },
  { value: 'xpos', label: 'xPos' },
  { value: 'alibi', label: 'ALiBi' },
] as const
const CAUSAL: Record<Scheme, boolean> = {
  none: false,
  sinusoid: false,
  shaw: false,
  t5: false,
  rope: false,
  xpos: true,
  alibi: true,
}

/** Fixed content: a query and a correlated key (k = q + 0.7·noise), plus Shaw's offset vectors, seeded. */
const SAMPLE = (() => {
  const r = rng(11)
  const q = Array.from({ length: D }, () => r.normal())
  const k = q.map((v) => v + 0.7 * r.normal())
  const shaw = Array.from({ length: 2 * 16 + 1 }, () => Array.from({ length: D }, () => 0.5 * r.normal()))
  return { q, k, shaw }
})()

const THETA = Array.from({ length: D / 2 }, (_, i) => BASE ** ((-2 * i) / D))
/** xPos decay per pair, (i/(d/2) + 0.4) / 1.4, applied with exponent position/512 as in the reference code. */
const ZETA = THETA.map((_, i) => (i / (D / 2) + 0.4) / 1.4)
const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)

/** Rotate pair i by t·θ_i and scale it by ζ_i^(t/512)·sign when xPos is on. */
function rotate(x: number[], t: number, decay: 0 | 1 | -1): number[] {
  const out = x.slice()
  THETA.forEach((w, i) => {
    const s = decay === 0 ? 1 : ZETA[i] ** ((decay * t) / 512)
    const c = Math.cos(t * w)
    const n = Math.sin(t * w)
    out[2 * i] = s * (c * x[2 * i] - n * x[2 * i + 1])
    out[2 * i + 1] = s * (n * x[2 * i] + c * x[2 * i + 1])
  })
  return out
}
const sinusoid = (t: number) => THETA.flatMap((w) => [Math.sin(t * w), Math.cos(t * w)])

/** T5's bidirectional bucket for offset n − m: 16 buckets per direction, 8 exact, log-spaced up to 128. */
function t5Bucket(offset: number): number {
  const side = offset > 0 ? 16 : 0
  const r = Math.abs(offset)
  if (r < 8) return side + r
  return side + Math.min(15, 8 + Math.floor((Math.log(r / 8) / Math.log(16)) * 8))
}
/** Illustrative learned biases: keys before the query favoured near, falling with bucket; keys after slightly lower. */
const t5Bias = (bucket: number) => (bucket < 16 ? 1.5 - 0.15 * bucket : 1.2 - 0.15 * (bucket - 16))

/** Score of the fixed query at position m against the fixed key at position n, under a scheme. */
function score(scheme: Scheme, m: number, n: number, slope: number): number {
  const { q, k, shaw } = SAMPLE
  const content = () => dot(q, k) / Math.sqrt(D)
  switch (scheme) {
    case 'none':
      return content()
    case 'sinusoid': {
      const pm = sinusoid(m)
      const pn = sinusoid(n)
      return (
        dot(
          q.map((v, i) => v + pm[i]),
          k.map((v, i) => v + pn[i]),
        ) / Math.sqrt(D)
      )
    }
    case 'shaw': {
      const a = shaw[Math.max(-16, Math.min(16, n - m)) + 16]
      return (
        dot(
          q,
          k.map((v, i) => v + a[i]),
        ) / Math.sqrt(D)
      )
    }
    case 't5':
      return content() + t5Bias(t5Bucket(n - m))
    case 'rope':
      return dot(rotate(q, m, 0), rotate(k, n, 0)) / Math.sqrt(D)
    case 'xpos':
      return dot(rotate(q, m, 1), rotate(k, n, -1)) / Math.sqrt(D)
    case 'alibi':
      return content() - slope * (m - n)
  }
}

/** Score against a fixed key under several schemes: pick one, move the query, and see whether only the offset matters. */
export function PositionExplorer() {
  const [scheme, setScheme] = useState<Scheme>('rope')
  const m2 = useParam(1500, { min: M1, max: 4000, step: 10 })
  const head = useParam(3, { min: 1, max: 8, step: 1 })
  const slope = 2 ** -head.value
  const causal = CAUSAL[scheme]

  const r = useMemo(() => {
    const a = DISTANCES.map((d) => score(scheme, M1, M1 - d, slope))
    const b = DISTANCES.map((d) => score(scheme, m2.value, m2.value - d, slope))
    const gap = Math.max(...a.map((v, i) => Math.abs(v - b[i])))
    const grid = POSITIONS.map((i) => POSITIONS.map((j) => (causal && j > i ? NaN : score(scheme, i, j, slope))))
    const finite = grid.flat().filter((v) => !Number.isNaN(v))
    const lo = Math.min(...finite)
    const hi = Math.max(...finite)
    // Masked cells (key after query) are drawn at the bottom of the colour scale: their weight is 0.
    const z = grid.map((row) => row.map((v) => (Number.isNaN(v) ? lo : v)))
    return { a, b, gap, z, lo, hi: hi > lo ? hi : lo + 1 }
  }, [scheme, m2.value, slope, causal])
  const range = useMemo<[number, number]>(() => [r.lo, r.hi], [r.lo, r.hi])

  const series: XYSeries[] = [
    { name: `query at position ${M1}`, type: 'line', x: DISTANCES, y: r.a, slot: 0 },
    { name: `query at position ${m2.value}`, type: 'line', x: DISTANCES, y: r.b, slot: 1, dashed: true },
  ]

  return (
    <Interactive
      title="Position scheme explorer"
      caption="A fixed query and a fixed, similar key (d = 64) are placed at positions m and n = m − r, and each chart shows their attention logit. Top: logit against the distance r, for two absolute query positions; the curves coincide exactly when the scheme is relative. Bottom: the logit for every query position i and key position j from 0 to 31. A relative scheme gives a matrix that is constant along each diagonal. Shaw's offset vectors and T5's bucket biases are learned in practice; here they are fixed illustrative values. ALiBi and xPos are causal: cells with j > i are masked and drawn at the bottom of the scale."
      controls={
        <>
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamChoice label="scheme" value={scheme} onChange={setScheme} options={SCHEMES} />
          </div>
          <ParamSlider label="second query position m" param={m2} />
          {scheme === 'alibi' && (
            <ParamSlider label="ALiBi head h of 8" param={head} format={(v) => `${v} (slope 1/${2 ** v})`} />
          )}
        </>
      }
      readout={
        <>
          <Readout label="largest gap between the two curves" value={formatNumber(r.gap)} />
          <Readout label="depends on" value={r.gap < 1e-6 ? 'offset only' : 'absolute positions too'} />
        </>
      }
    >
      <XYChart series={series} xLabel="distance r = m − n" yLabel="logit" height={260} />
      <Heatmap
        x={POSITIONS}
        y={POSITIONS}
        z={r.z}
        range={range}
        xLabel="key position j"
        yLabel="query position i"
        valueLabel="logit"
        height={320}
      />
    </Interactive>
  )
}
