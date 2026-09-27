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
import { rng } from '@/lib/math'
import { gram, makeKernel, samples } from '../../_shared/gp'
import { anomalyScore, classModel } from '../_shared/hgp'

const T = 40
const PHASE = Array.from({ length: T }, (_, i) => i / T)
const PER_CLASS = 6
const PHASE_RANGE: [number, number] = [0, 1]
const STAR_RANGE: [number, number] = [0, 19]
const CLASSES = ['Cepheid-like', 'RR Lyrae-like', 'eclipsing binary'] as const

/** Smoothed sawtooth: a fast rise to maximum and a slow decline, as in a Cepheid. */
const cepheid = (p: number) =>
  0.7 * [1, 2, 3, 4].reduce((s, k) => s + ((k % 2 ? 1 : -1) * Math.sin(2 * Math.PI * k * p)) / k, 0)
/** A steeper sawtooth with more harmonics, similar in shape to the Cepheid template. */
const rrLyrae = (p: number) =>
  0.8 *
  [1, 2, 3, 4, 5, 6, 7, 8].reduce((s, k) => s + ((k % 2 ? 1 : -1) * Math.sin(2 * Math.PI * k * (p - 0.05))) / k, 0)
const dip = (p: number, c: number, w: number) => {
  const d = ((p - c + 1.5) % 1) - 0.5
  return Math.exp(-0.5 * (d / w) ** 2)
}
/** Constant light with a deep primary and a shallow secondary eclipse half a period apart. */
const eclipsing = (p: number) => 0.6 - 1.4 * dip(p, 0.25, 0.05) - 0.7 * dip(p, 0.75, 0.05)
const TEMPLATES = [cepheid, rrLyrae, eclipsing]

/** Hyperparameters of the class model; the synthetic stars are generated from the same values. */
const SG = 1
const ELL_G = 0.1
const SF = 0.3
const ELL_F = 0.15
const SN = 0.2
const KG = makeKernel('matern32', { ell: ELL_G, sf: SG })
const KF = makeKernel('matern32', { ell: ELL_F, sf: SF })

/** Each star's own smooth deviation from its class template and its noise, fixed once so only the anomaly moves. */
const DEVIATIONS: number[][][] = (() => {
  const r = rng(3)
  const cov = gram(KF, PHASE, PHASE)
  return CLASSES.map(() =>
    samples(
      PHASE.map(() => 0),
      cov,
      Array.from({ length: PER_CLASS }, () => PHASE.map(() => r.normal())),
    ).map((h) => h.map((v) => v + SN * r.normal())),
  )
})()

type Kind = 'blend' | 'shift' | 'dip'
const KIND_OPTIONS = [
  { value: 'blend' as const, label: 'blend toward eclipse' },
  { value: 'shift' as const, label: 'phase shift' },
  { value: 'dip' as const, label: 'extra dip' },
]

/** The last Cepheid-like star, distorted by amount d. */
function anomaly(kind: Kind, d: number): number[] {
  const own = DEVIATIONS[0][PER_CLASS - 1]
  return PHASE.map((p, i) => {
    switch (kind) {
      case 'blend':
        return (1 - d) * cepheid(p) + d * eclipsing(p) + own[i]
      case 'shift':
        return cepheid((p - 0.5 * d + 1) % 1) + own[i]
      case 'dip':
        return cepheid(p) - 1.2 * d * dip(p, 0.6, 0.04) + own[i]
    }
  })
}

/**
 * Synthetic phase-folded light curves from three classes. Each star is scored by the negative log predictive density
 * of its whole curve under a hierarchical GP fitted to the other stars of its class.
 */
export function StarAnomaly() {
  const d = useParam(0.5, { min: 0, max: 1, step: 0.01 })
  const [kind, setKind] = useState<Kind>('blend')

  const r = useMemo(() => {
    const curves = CLASSES.map((_, c) =>
      DEVIATIONS[c].map((h, i) =>
        c === 0 && i === PER_CLASS - 1 ? anomaly(kind, d.value) : PHASE.map((p, k) => TEMPLATES[c](p) + h[k]),
      ),
    )
    // Leave one out: each star is scored under the class model of the other five.
    const scores = curves.map((ys) =>
      ys.map((y, i) =>
        anomalyScore(
          classModel(
            KG,
            KF,
            SN * SN,
            PHASE,
            ys.filter((_, m) => m !== i),
          ),
          y,
        ),
      ),
    )
    const held = classModel(KG, KF, SN * SN, PHASE, curves[0].slice(0, PER_CLASS - 1))
    const sd = held.chol.map((row) => Math.sqrt(row.reduce((s, v) => s + v * v, 0)))
    return { curves, scores, held, sd }
  }, [kind, d.value])

  const anomalyScoreValue = r.scores[0][PER_CLASS - 1]
  const normal = r.scores.flatMap((s, c) => (c === 0 ? s.slice(0, PER_CLASS - 1) : s))
  const rank = 1 + normal.filter((s) => s > anomalyScoreValue).length

  const curveSeries: XYSeries[] = [
    ...r.curves[0]
      .slice(0, PER_CLASS - 1)
      .map((y, i): XYSeries => ({ name: `Cepheid-like star ${i + 1}`, type: 'line', x: PHASE, y, muted: true })),
    { name: 'class model mean', type: 'line', x: PHASE, y: r.held.mean, emphasis: true },
    {
      name: 'class model ± 2 sd',
      type: 'line',
      x: PHASE,
      y: r.held.mean.map((m, i) => m + 2 * r.sd[i]),
      emphasis: true,
      dashed: true,
    },
    {
      name: 'class model − 2 sd',
      type: 'line',
      x: PHASE,
      y: r.held.mean.map((m, i) => m - 2 * r.sd[i]),
      emphasis: true,
      dashed: true,
    },
    { name: 'distorted star', type: 'line', x: PHASE, y: r.curves[0][PER_CLASS - 1], slot: 3 },
  ]

  const scoreSeries: XYSeries[] = [
    ...CLASSES.map((name, c): XYSeries => ({
      name,
      type: 'scatter',
      x: r.scores[c].map((_, i) => c * PER_CLASS + i + 1).filter((_, i) => c !== 0 || i < PER_CLASS - 1),
      y: r.scores[c].filter((_, i) => c !== 0 || i < PER_CLASS - 1),
      slot: c,
    })),
    { name: 'distorted star', type: 'scatter', x: [PER_CLASS], y: [anomalyScoreValue], slot: 3 },
  ]

  return (
    <Interactive
      title="Scoring phase-folded light curves with a class HGP"
      caption="Eighteen synthetic stars, six per class, each on 40 phase points. Left: the five undistorted Cepheid-like stars (grey), the predictive mean and ±2 sd band of a new replicate under the HGP fitted to them (ink), and the sixth star distorted by the slider. Right: every star's anomaly score S(y) = −ln p(y | other stars of its class), computed leave-one-out. Both kernels are Matérn 3/2 and the hyperparameters are the generating values. The phase-shift distortion moves the curve by d/2 of a cycle; a small shift is flagged more strongly than a large change of shape, because the model compares curves point by point in phase."
      controls={
        <>
          <ParamChoice label="distortion" value={kind} onChange={setKind} options={KIND_OPTIONS} />
          <ParamSlider label="distortion amount d" param={d} />
        </>
      }
      readout={
        <>
          <Readout label="score of distorted star" value={formatNumber(anomalyScoreValue)} />
          <Readout label="highest score among the other 17" value={formatNumber(Math.max(...normal))} />
          <Readout label="rank of distorted star" value={`${rank} of 18`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={curveSeries} xLabel="phase φ" yLabel="brightness" xRange={PHASE_RANGE} height={320} />
        <XYChart series={scoreSeries} xLabel="star" yLabel="anomaly score S" xRange={STAR_RANGE} height={320} />
      </div>
    </Interactive>
  )
}
