import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { grid } from '../_shared/gaussian'
import { CLICK_MODEL, labelProbabilities, scoreDensities } from '../_shared/clicks'

const S = grid(-1, 2, 301)

/** Combine a document's click record with a judge's label and read off the relevance score and label probabilities. */
export function ClickScore() {
  const state = useFigureState({
    exams: int(100, {
      ge: 1,
      le: 10000,
      scale: 'log10',
      suggestions: [1, 10, 100, 1000, 10000],
      label: 'examinations',
    }),
    rate: slider(0, 1, 0.1, { step: 0.01, label: 'click rate', format: (v) => v.toFixed(2) }),
    label: choice(
      [
        { value: 'none', label: 'none' },
        { value: '0', label: '0 not relevant' },
        { value: '1', label: '1 possibly' },
        { value: '2', label: '2 relevant' },
      ],
      'none',
      { label: "judge's label" },
    ),
  })
  const exams = state.exams
  const clicks = Math.round(state.rate * exams)
  const label = state.label
  const lab = label === 'none' ? null : Number(label)

  const d = useMemo(() => scoreDensities(S, clicks, exams, lab), [clicks, exams, lab])
  const probs = labelProbabilities(d.clicks.mean, d.clicks.variance)
  const series = useMemo((): SeriesSpec[] => {
    const out: SeriesSpec[] = [
      { name: 'prior', type: 'line', x: S, y: d.prior, muted: true },
      { name: 'clicks only', type: 'line', x: S, y: d.fromClicks, slot: 0 },
    ]
    if (lab !== null) out.push({ name: 'clicks and label', type: 'line', x: S, y: d.both, slot: 1, area: true })
    CLICK_MODEL.thresholds.forEach((t, i) =>
      out.push({ name: `threshold ${i + 1}`, type: 'line', x: [t, t], y: [0, 12], dashed: true, muted: true }),
    )
    return out
  }, [d, lab])

  const xAxis = useAxis({ label: 'latent relevance score', range: [-1, 2] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="A relevance score from clicks and a judgement"
      state={state}
      caption="Set how often the document was examined and what fraction of examinations ended in a click. The click record becomes a Gaussian observation of the latent score, as sharp as the number of examinations allows. Add a judge's label to see the two sources combined: a label that disagrees with many clicks barely moves the score, which flags the label for review. The dashed lines are the learned thresholds between labels 0, 1 and 2. Parameters are those Infer.NET learns on its example data."
      readouts={
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
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
