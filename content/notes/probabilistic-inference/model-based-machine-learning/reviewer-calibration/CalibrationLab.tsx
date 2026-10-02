import { useMemo, useState } from 'react'
import { Interactive, ParamButton, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'
import { calibrate, makeConference, scoreReviews, spearman, topK } from '../_shared/reviewers'

const MU = 5
const NOISE_SD = 0.5
const PRIOR = { mu: MU, qualityVar: 1, biasVar: 1, noiseVar: NOISE_SD ** 2 }
const CONF = makeConference(rng(2).normal, NOISE_SD)
const START = [-1.5, -1, -0.5, 0.5, 1, 1.5]
const REVIEWERS = START.map((_, r) => r + 1)
const ACCEPT = 4
const round = (v: number) => Math.round(v * 20) / 20
const TRUTH = CONF.quality.map((q) => MU + q)
const BEST = topK(TRUTH, ACCEPT)
const hits = (xs: number[]) => [...topK(xs, ACCEPT)].filter((i) => BEST.has(i)).length

/** Drag each reviewer's bias; compare raw mean scores with calibrated qualities against the truth. */
export function CalibrationLab() {
  const [bias, setBias] = useState(START)

  const { raw, cal } = useMemo(() => {
    const reviews = scoreReviews(CONF, bias, MU)
    const raw = CONF.quality.map((_, p) => {
      const s = reviews.filter((r) => r.paper === p)
      return s.reduce((a, r) => a + r.score, 0) / s.length
    })
    return { raw, cal: calibrate(reviews, CONF.quality.length, bias.length, PRIOR) }
  }, [bias])

  const biasSeries = useMemo(
    (): XYSeries[] => [
      { name: 'true bias', type: 'scatter', x: REVIEWERS, y: bias, slot: 0 },
      { name: 'inferred bias', type: 'scatter', x: REVIEWERS, y: cal.bias, emphasis: true },
    ],
    [bias, cal],
  )
  const handles = useMemo(
    (): Handle[] =>
      bias.map((b, r) => ({
        kind: 'point',
        at: [r + 1, b],
        label: `reviewer ${r + 1}`,
        onDrag: ([, y]) => setBias((old) => old.map((v, j) => (j === r ? round(Math.max(-2.5, Math.min(2.5, y))) : v))),
      })),
    [bias],
  )

  const paperSeries = useMemo((): XYSeries[] => {
    const lo = Math.min(...TRUTH) - 0.5
    const hi = Math.max(...TRUTH) + 0.5
    return [
      { name: 'raw mean score', type: 'scatter', x: TRUTH, y: raw, slot: 1 },
      { name: 'calibrated quality', type: 'scatter', x: TRUTH, y: cal.quality, slot: 2 },
      { name: 'perfect', type: 'line', x: [lo, hi], y: [lo, hi], muted: true },
    ]
  }, [raw, cal])

  return (
    <Interactive
      title="Raw and calibrated paper scores"
      caption="Twelve papers, six reviewers, three reviews per paper. Reviewers 1 to 3 mostly review the six best papers and reviewers 4 to 6 the six worst, as happens when reviewers are matched to sub-areas. Left: drag each reviewer's bias, the amount they add to every score; the diamonds are the biases the model infers. Right: each paper's raw mean score and its calibrated quality (posterior mean) against the true quality. With harsh reviewers on strong papers and lenient ones on weak papers, raw means squeeze the field together and scramble the ranking; the calibrated estimates undo it."
      controls={<ParamButton onClick={() => setBias(START)}>reset biases</ParamButton>}
      readout={
        <>
          <Readout label="rank correlation, raw" value={formatNumber(spearman(raw, TRUTH))} />
          <Readout label="rank correlation, calibrated" value={formatNumber(spearman(cal.quality, TRUTH))} />
          <Readout label={`true top ${ACCEPT} found, raw`} value={hits(raw)} />
          <Readout label={`true top ${ACCEPT} found, calibrated`} value={hits(cal.quality)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart
          series={biasSeries}
          xLabel="reviewer"
          yLabel="bias (score points)"
          xRange={[0.5, 6.5]}
          yRange={[-2.5, 2.5]}
          handles={handles}
          height={300}
        />
        <XYChart series={paperSeries} xLabel="true quality" yLabel="estimate" height={300} />
      </div>
    </Interactive>
  )
}
