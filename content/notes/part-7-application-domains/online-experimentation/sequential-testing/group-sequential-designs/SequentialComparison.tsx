import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { TABLES, type Alpha } from './tables'
import { normalCdf, normalQuantile } from 'aifn/numerics/special'

const Z_MAX = 6
const ALPHAS: { value: Alpha; label: string }[] = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]

/** Two-sided Lan–DeMets O'Brien–Fleming-type spending: 2 − 2Φ(z_{1−α/4}/√t) per tail. */
const obfSpent = (t: number, a: number) => 2 * (2 - 2 * normalCdf(normalQuantile(1 - a / 4) / Math.sqrt(t)))
/** Lan–DeMets Pocock-type spending: α·log(1 + (e − 1)t). */
const pocockSpent = (t: number, a: number) => a * Math.log(1 + (Math.E - 1) * t)

/**
 * mSPRT boundary on the z-scale at look k of K with batch variance 1: V = 1/k is the variance of the running mean
 * difference and τ = (z_{1−α/2} + z_{0.8})/√K is the effect the fixed-horizon test of all K batches detects with 80%
 * power. Rejects when Z² ≥ (V + τ²)/τ² · (2 log(1/α) + log((V + τ²)/V)).
 */
function msprtBoundary(k: number, K: number, a: number): number {
  const tau2 = (normalQuantile(1 - a / 2) + normalQuantile(0.8)) ** 2 / K
  const v = 1 / k
  return Math.sqrt(((v + tau2) / tau2) * (2 * Math.log(1 / a) + Math.log((v + tau2) / v)))
}

const clip = (c: number) => (c > Z_MAX ? NaN : c)

/**
 * Four ways to analyse an A/A test at K equally spaced looks: the fixed-sample threshold at every look, Lan–DeMets
 * Pocock-type and O'Brien–Fleming-type alpha spending, and the mixture sequential probability ratio test.
 */
export function SequentialComparison() {
  const state = useFigureState({
    looks: int(10, { min: 1, max: 50, step: 1, label: 'looks K' }),
    alpha: choice<Alpha>(ALPHAS, '0.05', { label: 'α' }),
  })
  const K = state.looks

  const r = useMemo(() => {
    const a = Number(state.alpha)
    const table = TABLES[state.alpha]
    const ks = Array.from({ length: K }, (_, i) => i + 1)
    const z = normalQuantile(1 - a / 2)
    const bounds = {
      naive: ks.map(() => z),
      pocock: table.pocock[K - 1],
      obf: table.obf[K - 1],
      msprt: ks.map((k) => msprtBoundary(k, K, a)),
    }
    const spent = {
      naive: table.naive.slice(0, K),
      pocock: ks.map((k) => pocockSpent(k / K, a)),
      obf: ks.map((k) => obfSpent(k / K, a)),
      msprt: table.msprt[K - 1],
    }
    const xEnd = Math.max(K, 2)
    const boundary = [
      { name: 'fixed threshold every look', x: ks, y: bounds.naive, slot: 0 },
      { name: 'Pocock-type', x: ks, y: bounds.pocock.map(clip), slot: 1 },
      { name: "O'Brien–Fleming-type", x: ks, y: bounds.obf.map(clip), slot: 2 },
      { name: 'mSPRT', x: ks, y: bounds.msprt.map(clip), slot: 3 },
    ] as const
    const rate = [
      { name: 'fixed threshold every look', x: ks, y: spent.naive, slot: 0 },
      { name: 'Pocock-type', x: ks, y: spent.pocock, slot: 1 },
      { name: "O'Brien–Fleming-type", x: ks, y: spent.obf, slot: 2 },
      { name: 'mSPRT', x: ks, y: spent.msprt, slot: 3 },
      { name: 'α', x: [1, xEnd], y: [a, a], emphasis: true, dashed: true },
    ] as const
    return { boundary, rate, final: { naive: spent.naive[K - 1], msprt: spent.msprt[K - 1] }, xEnd }
  }, [K, state.alpha])

  const xAxis = useAxis({ label: 'look', range: [1, r.xEnd] })
  const yAxis = useAxis({ label: 'boundary for |Z|', range: [0, Z_MAX] })
  const xAxis2 = useAxis({ label: 'look', range: [1, r.xEnd] })
  const yAxis2 = useAxis({ label: 'P(false positive by this look)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Four stopping rules on an A/A test"
      state={state}
      caption="An A/A test analysed at K equally spaced looks and stopped at the first look where |Z| reaches the boundary. Left: each rule's boundary; O'Brien–Fleming-type boundaries above the plotted range are not drawn. Right: the probability, with no true effect, of having stopped by each look. The fixed-sample threshold z₁₋α/₂ exceeds α from the second look on. The two alpha-spending designs spend exactly α by the last look, Pocock-type evenly and O'Brien–Fleming-type mostly at the end. The mSPRT, with its mixing scale set to the effect the full fixed-horizon test detects with 80% power, stays far below α because its guarantee covers checking after every observation."

      readouts={
        <>
          <Readout label="fixed threshold: false-positive rate" value={formatNumber(r.final.naive)} />
          <Readout label="alpha spending: false-positive rate" value={formatNumber(Number(state.alpha))} />
          <Readout label="mSPRT: false-positive rate" value={formatNumber(r.final.msprt)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve {...r.boundary[0]} />
          <Curve {...r.boundary[1]} />
          <Curve {...r.boundary[2]} />
          <Curve {...r.boundary[3]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...r.rate[0]} />
          <Curve {...r.rate[1]} />
          <Curve {...r.rate[2]} />
          <Curve {...r.rate[3]} />
          <Curve {...r.rate[4]} />
        </Plot>
      </div>
    </Figure>
  )
}
