import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { grid, normalLogPdf, normalPdf } from '../_shared/ep'

// Integrate well beyond the plotted range: a broad q (variance about 5.5) has mass past ±8 that would bias D_α.
const XS = grid(-14, 14, 1121)
const H = XS[1] - XS[0]
const X_RANGE: [number, number] = [-8, 8]
const Y_RANGE: [number | undefined, number | undefined] = [0, undefined]
const TARGET = XS.map((x) => 0.35 * normalPdf(x, -3, 0.36) + 0.65 * normalPdf(x, 2, 0.64))
const LOG_TARGET = TARGET.map((p) => Math.log(Math.max(p, 1e-300)))
type Start = 'wide' | 'left' | 'right'
const STARTS: Record<Start, [number, number]> = {
  wide: [0.25, Math.log(2.5)],
  left: [-3, Math.log(0.6)],
  right: [2, Math.log(0.8)],
}
const START_OPTIONS = [
  { value: 'wide' as const, label: 'wide' },
  { value: 'left' as const, label: 'left mode' },
  { value: 'right' as const, label: 'right mode' },
]

/** The α-divergence D_α(p ‖ q) for a Gaussian q = N(m, s²), by the trapezoid rule; α = 0 and 1 are the two KLs. */
function divergence(alpha: number, [m, logS]: [number, number]): number {
  const v = Math.exp(2 * logS)
  let acc = 0
  XS.forEach((x, i) => {
    const lq = normalLogPdf(x, m, v)
    const p = TARGET[i]
    const q = Math.exp(lq)
    if (alpha >= 1) acc += p > 0 ? p * (LOG_TARGET[i] - lq) : 0
    else if (alpha <= 0) acc += q > 0 ? q * (lq - LOG_TARGET[i]) : 0
    else
      acc += (alpha * p + (1 - alpha) * q - Math.exp(alpha * LOG_TARGET[i] + (1 - alpha) * lq)) / (alpha * (1 - alpha))
  })
  return acc * H
}

/**
 * Nelder–Mead in two dimensions: a local minimiser, so the answer depends on the start, as it does for VB. The objective
 * is flat near its minimum, so a stop on function values alone ends early; the search also requires the simplex to
 * shrink and restarts once from its best point.
 */
function minimise(f: (p: [number, number]) => number, start: [number, number]): [number, number] {
  return nelderMead(f, nelderMead(f, start))
}

function nelderMead(f: (p: [number, number]) => number, start: [number, number]): [number, number] {
  let simplex: [number, number][] = [start, [start[0] + 0.5, start[1]], [start[0], start[1] + 0.3]]
  let values = simplex.map(f)
  for (let it = 0; it < 2000; it++) {
    const order = [0, 1, 2].sort((a, b) => values[a] - values[b])
    simplex = order.map((i) => simplex[i])
    values = order.map((i) => values[i])
    const spread = Math.max(...simplex.map((p) => Math.hypot(p[0] - simplex[0][0], p[1] - simplex[0][1])))
    if (Math.abs(values[2] - values[0]) < 1e-12 && spread < 1e-6) break
    const c: [number, number] = [(simplex[0][0] + simplex[1][0]) / 2, (simplex[0][1] + simplex[1][1]) / 2]
    const along = (t: number): [number, number] => [
      c[0] + t * (simplex[2][0] - c[0]),
      c[1] + t * (simplex[2][1] - c[1]),
    ]
    const r = along(-1)
    const fr = f(r)
    if (fr < values[0]) {
      const e = along(-2)
      const fe = f(e)
      ;[simplex[2], values[2]] = fe < fr ? [e, fe] : [r, fr]
    } else if (fr < values[1]) {
      ;[simplex[2], values[2]] = [r, fr]
    } else {
      const k = along(0.5)
      const fk = f(k)
      if (fk < values[2]) [simplex[2], values[2]] = [k, fk]
      else {
        simplex = simplex.map((p) => [(p[0] + simplex[0][0]) / 2, (p[1] + simplex[0][1]) / 2] as [number, number])
        values = simplex.map(f)
      }
    }
  }
  return simplex[values.indexOf(Math.min(...values))]
}

/**
 * The Gaussian closest to a bimodal target under the α-divergence. α = 1 is KL(p ‖ q), EP's local objective; α = 0 is
 * KL(q ‖ p), the objective of variational Bayes.
 */
export function AlphaProjection() {
  const alpha = useParam(1, { min: 0, max: 1, step: 0.05 })
  const [start, setStart] = useState<Start>('wide')

  const r = useMemo(() => {
    const [m, logS] = minimise((p) => divergence(alpha.value, p), STARTS[start])
    const v = Math.exp(2 * logS)
    return { m, v, d: divergence(alpha.value, [m, logS]), q: XS.map((x) => normalPdf(x, m, v)) }
  }, [alpha.value, start])

  const series: XYSeries[] = [
    { name: 'target p', type: 'line', x: XS, y: TARGET, emphasis: true },
    { name: 'closest Gaussian q', type: 'line', x: XS, y: r.q, slot: 0 },
  ]
  const label =
    alpha.value >= 1
      ? 'KL(p ‖ q): EP'
      : alpha.value <= 0
        ? 'KL(q ‖ p): VB'
        : alpha.value === 0.5
          ? 'Hellinger (α = ½)'
          : 'α-divergence'

  return (
    <Interactive
      title="Mass-covering to mode-seeking: the α-divergence"
      caption="A two-mode target p and the Gaussian q that minimises D_α(p ‖ q), found by local search from the chosen start. At α = 1 (EP's divergence, KL(p ‖ q)) q matches the mean and variance of p and covers both modes, with much of its mass in the valley. As α falls, q narrows. Started wide, the search locks onto a mode below about α = 0.3; started at a mode, it stays there for α up to about 0.5. Which mode depends on the start. α = 0 is KL(q ‖ p), the divergence that variational Bayes minimises."
      controls={
        <>
          <ParamSlider label="α" param={alpha} />
          <ParamChoice label="search starts at" value={start} onChange={setStart} options={START_OPTIONS} />
        </>
      }
      readout={
        <>
          <Readout label="divergence" value={label} />
          <Readout label="q mean, variance" value={`${formatNumber(r.m)}, ${formatNumber(r.v)}`} />
          <Readout label="D_α at the optimum" value={formatNumber(r.d)} />
        </>
      }
    >
      <XYChart series={series} xLabel="θ" yLabel="density" xRange={X_RANGE} yRange={Y_RANGE} />
    </Interactive>
  )
}
