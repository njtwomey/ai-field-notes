import { useMemo } from 'react'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const FINE = 2 ** 16
const KS = Array.from({ length: 16 }, (_, i) => i + 1)
const MAX_PATHS = 30

type Stats = { quot: number; qv: number }[]
const statsCache = new Map<number, Stats>()

/**
 * For path p on [0, 1] and every resolution k: mean |ΔW|/h and Σ(ΔW)² at step h = 2^−k. Path p has its own random
 * stream (path 0 keeps seed 7), so adding paths leaves the existing ones unchanged. Cached: each path costs 2^16 draws.
 */
function pathStats(p: number): Stats {
  const hit = statsCache.get(p)
  if (hit) return hit
  const { normal } = rng(p === 0 ? 7 : 7000 + p)
  const b = new Float64Array(FINE + 1)
  const sd = Math.sqrt(1 / FINE)
  for (let i = 1; i <= FINE; i++) b[i] = b[i - 1] + sd * normal()
  const out = KS.map((kk) => {
    const m = 2 ** kk
    const stride = FINE / m
    let quot = 0
    let qv = 0
    for (let j = 1; j <= m; j++) {
      const d = b[j * stride] - b[(j - 1) * stride]
      quot += Math.abs(d) * m
      qv += d * d
    }
    return { quot: quot / m, qv }
  })
  statsCache.set(p, out)
  return out
}

/**
 * Brownian paths on [0, 1], each viewed at steps h = 2^−k. The difference quotient ΔW/h has typical size √(2/(πh)) and
 * never settles, while the sum of (ΔW)² settles at 1 on every path: (dW)² behaves like dt.
 */
export function RoughPath() {
  const k = useParam(4, { min: 1, max: 16, step: 1 })
  const count = useParam(10, { min: 1, max: MAX_PATHS, step: 1 })

  // Statistics at every resolution, computed once per path; the k slider only picks a column.
  const stats = useMemo(() => Array.from({ length: count.value }, (_, p) => pathStats(p)), [count.value])
  const avg = useMemo(
    () =>
      KS.map((_, i) => ({
        quot: stats.reduce((s, st) => s + st[i].quot, 0) / stats.length,
        qv: stats.reduce((s, st) => s + st[i].qv, 0) / stats.length,
      })),
    [stats],
  )

  const series = useMemo<XYSeries[]>(() => {
    const many = stats.length > 1
    const out: XYSeries[] = []
    for (const st of stats)
      out.push({
        name: many ? 'mean |ΔW| / h, each path' : 'mean |ΔW| / h',
        type: 'line',
        x: KS,
        y: st.map((s) => s.quot),
        slot: 0,
        thin: many,
      })
    for (const st of stats)
      out.push({
        name: many ? 'Σ (ΔW)², each path' : 'Σ (ΔW)²',
        type: 'line',
        x: KS,
        y: st.map((s) => s.qv),
        slot: 1,
        thin: many,
      })
    if (many)
      out.push(
        { name: 'mean |ΔW| / h, average', type: 'line', x: KS, y: avg.map((s) => s.quot), slot: 0 },
        { name: 'Σ (ΔW)², average', type: 'line', x: KS, y: avg.map((s) => s.qv), slot: 1 },
      )
    out.push(
      {
        name: '√(2 / (π h))',
        type: 'line',
        x: KS,
        y: KS.map((kk) => Math.sqrt((2 * 2 ** kk) / Math.PI)),
        dashed: true,
        muted: true,
      },
      { name: 'T = 1', type: 'line', x: [1, 16], y: [1, 1], dashed: true, emphasis: true },
    )
    return out
  }, [stats, avg])

  const now = avg[k.value - 1]
  const qvs = stats.map((st) => st[k.value - 1].qv)
  const many = count.value > 1

  return (
    <Interactive
      title="Why ordinary calculus fails on a Brownian path"
      caption="Brownian paths on [0, 1], each sampled with step h = 2⁻ᵏ; the paths slider sets how many. With several paths each one is a light line and the solid lines average them. The average slope |ΔW|/h grows like 1/√h without limit on every path, so no path has a derivative. The sum of squared increments settles at 1, the length of the interval, and the paths bunch ever closer around it as h shrinks: over a step of length h, (ΔW)² is about h. Drag the vertical line or use the slider to change k. The y axis is logarithmic."
      controls={
        <>
          <ParamSlider label="k (step h = 2⁻ᵏ)" param={k} withArrows />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="h" value={formatNumber(2 ** -k.value)} />
          <Readout label={many ? 'mean |ΔW| / h (average)' : 'mean |ΔW| / h'} value={formatNumber(now.quot)} />
          <Readout label={many ? 'Σ (ΔW)² (average)' : 'Σ (ΔW)²'} value={formatNumber(now.qv)} />
          {many && (
            <Readout
              label="Σ (ΔW)² range over paths"
              value={`${formatNumber(Math.min(...qvs))} – ${formatNumber(Math.max(...qvs))}`}
            />
          )}
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="k"
        yLabel="value"
        yLog
        series={series}
        xRange={[1, 16]}
        yRange={[0.02, 300]}
        handles={[{ kind: 'x', at: k.value, label: 'k', onDrag: (x) => k.set(x) }]}
      />
    </Interactive>
  )
}

const box = { shape: 'box' as const, w: 3.4, h: 1.1 }
const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'bm', x: 1.5, y: 1, ...box, label: 'Brownian motion', tone: 'neutral' },
    { id: 'ito', x: 5.5, y: 1, ...box, label: 'Itô integral', tone: 0 },
    { id: 'lemma', x: 9.5, y: 1, ...box, label: "Itô's lemma", tone: 0 },
    { id: 'sde', x: 13.5, y: 1, ...box, label: 'SDEs', tone: 0 },
    { id: 'fp', x: 9.5, y: 3.5, ...box, label: 'Fokker–Planck', tone: 1 },
    { id: 'ou', x: 13.5, y: 3.5, ...box, label: 'Ornstein–Uhlenbeck', tone: 1 },
    { id: 'em', x: 17.5, y: 3.5, ...box, label: 'Euler–Maruyama', tone: 1 },
    { id: 'rev', x: 5.5, y: 6, ...box, label: 'reverse-time SDE', tone: 2 },
    { id: 'pf', x: 9.5, y: 6, ...box, label: 'probability-flow ODE', tone: 2 },
    { id: 'tw', x: 13.5, y: 6, ...box, label: "Tweedie's formula", tone: 2 },
    { id: 'diff', x: 9.5, y: 8.5, shape: 'box', w: 5, h: 1.1, label: 'diffusion models', highlight: true },
  ],
  edges: [
    { from: 'bm', to: 'ito' },
    { from: 'ito', to: 'lemma' },
    { from: 'lemma', to: 'sde' },
    { from: 'sde:s', to: 'ou:n' },
    { from: 'sde:e', to: 'em:n', via: [[17.5, 1]] },
    { from: 'lemma:s', to: 'fp:n' },
    { from: 'fp:w', to: 'rev:n', via: [[5.5, 3.5]] },
    { from: 'fp:s', to: 'pf:n' },
    { from: 'ou:s', to: 'tw:n', dashed: true },
    {
      from: 'rev:s',
      to: 'diff:n',
      via: [
        [5.5, 7.25],
        [9.5, 7.25],
      ],
    },
    { from: 'pf:s', to: 'diff:n' },
    {
      from: 'tw:s',
      to: 'diff:n',
      via: [
        [13.5, 7.25],
        [9.5, 7.25],
      ],
    },
  ],
}

/** The order of ideas in this section, from Brownian motion to diffusion models. */
export function SectionMap() {
  return (
    <Interactive
      title="The route from Brownian motion to diffusion models"
      caption="Each box is a note in this section; an arrow means the idea at its tail is used to build the idea at its head. Itô's lemma drives everything below it: it gives the Fokker–Planck equation for densities, which in turn gives the reverse-time SDE and the probability-flow ODE. The Ornstein–Uhlenbeck process is the forward noising process of a diffusion model, and Tweedie's formula turns its denoiser into a score."
    >
      <Diagram
        spec={spec}
        ariaLabel="Map of the stochastic calculus section: Brownian motion, Itô integral, Itô's lemma, SDEs, then Fokker–Planck, Ornstein–Uhlenbeck and Euler–Maruyama, then reverse-time SDE, probability-flow ODE and Tweedie's formula, all feeding diffusion models"
      />
    </Interactive>
  )
}
