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
} from 'aifn-render'
import { normalCdf, normalQuantile } from '@/lib/math/special'
import { TABLES, type Alpha } from './tables'

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
  const looks = useParam(10, { min: 1, max: 50, step: 1 })
  const [alpha, setAlpha] = useState<Alpha>('0.05')
  const K = looks.value

  const r = useMemo(() => {
    const a = Number(alpha)
    const table = TABLES[alpha]
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
    const boundary: XYSeries[] = [
      { name: 'fixed threshold every look', type: 'line', x: ks, y: bounds.naive, slot: 0 },
      { name: 'Pocock-type', type: 'line', x: ks, y: bounds.pocock.map(clip), slot: 1 },
      { name: "O'Brien–Fleming-type", type: 'line', x: ks, y: bounds.obf.map(clip), slot: 2 },
      { name: 'mSPRT', type: 'line', x: ks, y: bounds.msprt.map(clip), slot: 3 },
    ]
    const rate: XYSeries[] = [
      { name: 'fixed threshold every look', type: 'line', x: ks, y: spent.naive, slot: 0 },
      { name: 'Pocock-type', type: 'line', x: ks, y: spent.pocock, slot: 1 },
      { name: "O'Brien–Fleming-type", type: 'line', x: ks, y: spent.obf, slot: 2 },
      { name: 'mSPRT', type: 'line', x: ks, y: spent.msprt, slot: 3 },
      { name: 'α', type: 'line', x: [1, xEnd], y: [a, a], emphasis: true, dashed: true },
    ]
    return { boundary, rate, final: { naive: spent.naive[K - 1], msprt: spent.msprt[K - 1] }, xEnd }
  }, [K, alpha])

  return (
    <Interactive
      title="Four stopping rules on an A/A test"
      caption="An A/A test analysed at K equally spaced looks and stopped at the first look where |Z| reaches the boundary. Left: each rule's boundary; O'Brien–Fleming-type boundaries above the plotted range are not drawn. Right: the probability, with no true effect, of having stopped by each look. The fixed-sample threshold z₁₋α/₂ exceeds α from the second look on. The two alpha-spending designs spend exactly α by the last look, Pocock-type evenly and O'Brien–Fleming-type mostly at the end. The mSPRT, with its mixing scale set to the effect the full fixed-horizon test detects with 80% power, stays far below α because its guarantee covers checking after every observation."
      controls={
        <>
          <ParamSlider label="looks K" param={looks} />
          <ParamChoice label="α" value={alpha} onChange={setAlpha} options={ALPHAS} />
        </>
      }
      readout={
        <>
          <Readout label="fixed threshold: false-positive rate" value={formatNumber(r.final.naive)} />
          <Readout label="alpha spending: false-positive rate" value={formatNumber(Number(alpha))} />
          <Readout label="mSPRT: false-positive rate" value={formatNumber(r.final.msprt)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel="look"
          yLabel="boundary for |Z|"
          series={r.boundary}
          xRange={[1, r.xEnd]}
          yRange={[0, Z_MAX]}
        />
        <XYChart
          height={320}
          xLabel="look"
          yLabel="P(false positive by this look)"
          series={r.rate}
          xRange={[1, r.xEnd]}
          yRange={[0, undefined]}
        />
      </div>
    </Interactive>
  )
}
