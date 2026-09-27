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
import { grid } from '../_shared/gaussian'
import { CLICK_MODEL, labelProbabilities, scoreDensities } from '../_shared/clicks'

const S = grid(-1, 2, 301)

/** Combine a document's click record with a judge's label and read off the relevance score and label probabilities. */
export function ClickScore() {
  const logExams = useParam(2, { min: 0, max: 4, step: 0.1 })
  const rate = useParam(0.1, { min: 0, max: 1, step: 0.01 })
  const [label, setLabel] = useState('none')
  const exams = Math.round(10 ** logExams.value)
  const clicks = Math.round(rate.value * exams)
  const lab = label === 'none' ? null : Number(label)

  const d = useMemo(() => scoreDensities(S, clicks, exams, lab), [clicks, exams, lab])
  const probs = labelProbabilities(d.clicks.mean, d.clicks.variance)
  const series = useMemo((): XYSeries[] => {
    const out: XYSeries[] = [
      { name: 'prior', type: 'line', x: S, y: d.prior, muted: true },
      { name: 'clicks only', type: 'line', x: S, y: d.fromClicks, slot: 0 },
    ]
    if (lab !== null) out.push({ name: 'clicks and label', type: 'line', x: S, y: d.both, slot: 1, area: true })
    CLICK_MODEL.thresholds.forEach((t, i) =>
      out.push({ name: `threshold ${i + 1}`, type: 'line', x: [t, t], y: [0, 12], dashed: true, muted: true }),
    )
    return out
  }, [d, lab])

  return (
    <Interactive
      title="A relevance score from clicks and a judgement"
      caption="Set how often the document was examined and what fraction of examinations ended in a click. The click record becomes a Gaussian observation of the latent score, as sharp as the number of examinations allows. Add a judge's label to see the two sources combined: a label that disagrees with many clicks barely moves the score, which flags the label for review. The dashed lines are the learned thresholds between labels 0, 1 and 2. Parameters are those Infer.NET learns on its example data."
      controls={
        <>
          <ParamSlider label="examinations (log₁₀)" param={logExams} format={() => String(exams)} />
          <ParamSlider label="click rate" param={rate} format={(v) => v.toFixed(2)} />
          <ParamChoice
            label="judge's label"
            value={label}
            onChange={setLabel}
            options={[
              { value: 'none', label: 'none' },
              { value: '0', label: '0 not relevant' },
              { value: '1', label: '1 possibly' },
              { value: '2', label: '2 relevant' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="clicks / exams" value={`${clicks} / ${exams}`} />
          <Readout
            label="score from clicks"
            value={`${formatNumber(d.clicks.mean)} ± ${formatNumber(Math.sqrt(d.clicks.variance))}`}
          />
          {lab !== null && <Readout label="with label" value={`${formatNumber(d.mean)} ± ${formatNumber(d.sd)}`} />}
          <Readout label="P(label 0, 1, 2 | clicks)" value={probs.map((p) => formatNumber(p)).join(', ')} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="latent relevance score"
        yLabel="density"
        xRange={[-1, 2]}
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}
