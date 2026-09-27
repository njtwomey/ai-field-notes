import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const M = linspace(-3, 3, 301)

type Surrogate = { name: string; phi: (m: number) => number; argmin: (eta: number) => number }

// Minimisers of the conditional risk ηφ(α) + (1 − η)φ(−α), derived in the note.
const SURROGATES: Surrogate[] = [
  { name: 'logistic (base 2)', phi: (m) => Math.log1p(Math.exp(-m)) / Math.LN2, argmin: (e) => Math.log(e / (1 - e)) },
  { name: 'hinge', phi: (m) => Math.max(0, 1 - m), argmin: (e) => (e > 0.5 ? 1 : e < 0.5 ? -1 : 0) },
  { name: 'exponential', phi: (m) => Math.exp(-m), argmin: (e) => 0.5 * Math.log(e / (1 - e)) },
  { name: 'squared hinge', phi: (m) => Math.max(0, 1 - m) ** 2, argmin: (e) => 2 * e - 1 },
]

/** The 0–1 loss and convex surrogates against the margin, and the conditional risk each one minimises. */
export function SurrogateLosses() {
  const eta = useParam(0.8, { min: 0.05, max: 0.95, step: 0.05 })

  const losses = useMemo(
    (): XYSeries[] => [
      { name: '0–1', type: 'line', x: M, y: M.map((m) => (m <= 0 ? 1 : 0)), emphasis: true },
      ...SURROGATES.map((s, i): XYSeries => ({ name: s.name, type: 'line', x: M, y: M.map(s.phi), slot: i })),
    ],
    [],
  )
  const risks = useMemo((): XYSeries[] => {
    const e = eta.value
    return [
      {
        name: '0–1',
        type: 'line',
        x: M,
        y: M.map((a) => (a > 0 ? 1 - e : a < 0 ? e : 0.5)),
        emphasis: true,
      },
      ...SURROGATES.map((s, i): XYSeries => ({
        name: s.name,
        type: 'line',
        x: M,
        y: M.map((a) => e * s.phi(a) + (1 - e) * s.phi(-a)),
        slot: i,
      })),
    ]
  }, [eta.value])

  return (
    <Interactive
      title="Surrogate losses and the conditional risk they minimise"
      caption="Left: each loss against the margin m = y·f(x); every surrogate is convex and lies on or above the 0–1 step. The logistic loss is drawn in base 2, so that it passes through 1 at m = 0. Right: the expected loss at a point where P(y = +1 | x) = η, as a function of the score α. Every surrogate's minimum lies on the same side of zero as the Bayes decision, sign(2η − 1). The logistic, exponential and squared-hinge minima also move with η, so they encode the probability; the hinge minimum sits at ±1 whatever η is."
      controls={<ParamSlider label="η = P(y = +1 | x)" param={eta} />}
      readout={
        <>
          {SURROGATES.map((s) => (
            <Readout key={s.name} label={`${s.name} α*`} value={formatNumber(s.argmin(eta.value))} />
          ))}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={losses}
          xLabel="margin m = y f(x)"
          yLabel="loss"
          xRange={[-3, 3]}
          yRange={[0, 4]}
          height={320}
        />
        <XYChart
          series={risks}
          xLabel="score α"
          yLabel="conditional risk ηφ(α) + (1 − η)φ(−α)"
          xRange={[-3, 3]}
          yRange={[0, 4]}
          height={320}
        />
      </div>
    </Interactive>
  )
}
