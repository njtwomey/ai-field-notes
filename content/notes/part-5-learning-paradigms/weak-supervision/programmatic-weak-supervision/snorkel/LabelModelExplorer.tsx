import { useMemo, useState } from 'react'
import { Bars, ControlRow, Figure, Plot, Plots, Points, Readout, Select, formatNumber, useAxis } from 'aifn-render'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { labellingFunctions } from 'aifn-methods/data/synthetic'
import { labelModelReport } from 'aifn-methods/learning/weak-supervision'

const NOISE_PRESETS = [
  { value: 'clean', label: 'High quality LFs (accuracy 70% – 95%)' },
  { value: 'noisy', label: 'Noisy LFs (accuracy 55% – 80%)' },
  { value: 'adversarial', label: 'Near-random LFs (accuracy 48% – 70%)' },
]

export function LabelModelExplorer() {
  const [numLfs, setNumLfs] = useState(10)
  const [noisePreset, setNoisePreset] = useState<'clean' | 'noisy' | 'adversarial'>('noisy')
  const [correlatedCopies, setCorrelatedCopies] = useState(0)

  const report = useMemo(() => {
    const s = stream(`snorkel-lf-${noisePreset}-${correlatedCopies}-${numLfs}`)
    const n = 400
    const classes = 2

    let low = 0.55
    let high = 0.8
    if (noisePreset === 'clean') {
      low = 0.7
      high = 0.95
    } else if (noisePreset === 'adversarial') {
      low = 0.48
      high = 0.7
    }

    const data = labellingFunctions(s, {
      n,
      classes,
      functions: numLfs,
      copies: correlatedCopies,
      accuracy: [low, high],
    })

    const truth = Array.from(toFlat(data.y))
    const res = labelModelReport(data.votes, classes, truth, { steps: 25 })
    return { data, report: res, truth }
  }, [numLfs, noisePreset, correlatedCopies])

  const res = report.report

  const totalFunctions = numLfs + correlatedCopies
  const lfIndices = useMemo(() => Array.from({ length: totalFunctions }, (_, i) => i), [totalFunctions])

  const empiricalAccuracies = res.empiricalAccuracy
  const estimatedAccuracies = res.labelModelAccuracy

  const methodIndices = [0, 1, 2]
  const overallAccuracies = [res.accuracy.majority, res.accuracy.dawidSkene, res.accuracy.labelModel]

  const accAxisX = useAxis({ label: 'Labelling Function Index', range: [-0.5, totalFunctions - 0.5] })
  const accAxisY = useAxis({ label: 'Estimated vs Empirical Accuracy', range: [0.4, 1.0] })

  const methodAxisX = useAxis({ label: 'Method (0: Majority, 1: Dawid-Skene, 2: Snorkel LM)', range: [-0.5, 2.5] })
  const methodAxisY = useAxis({ label: 'Training Label Accuracy', range: [0.5, 1.0] })

  const labelDensity = useMemo(() => {
    const flatVotes = toFlat(report.data.votes)
    let nonAbstain = 0
    for (let i = 0; i < flatVotes.length; i++) {
      if (flatVotes[i] >= 0) nonAbstain++
    }
    return nonAbstain / report.data.votes.shape[0]
  }, [report])

  return (
    <Figure
      title="Snorkel Data Programming Label Model"
      purpose="Compare generative covariance label modeling against majority voting and Dawid-Skene aggregation under noisy and correlated labeling functions."
      caption={
        'Snorkel generative label model vs Majority Vote and Dawid–Skene (Ratner et al., 2017; Ratner et al., 2020). Left: true empirical LF accuracy (blue) against Snorkel’s unsupervised estimates (gold) learned purely from mutual agreements and disagreements without ground truth. Right: resulting accuracy of training labels. When correlated copies exist, majority vote is duped by redundancy, whereas the generative model downweights correlated functions.'
      }
    >
      <ControlRow label="Labelling functions">
        <Select
          label="Total LFs"
          value={String(numLfs)}
          options={[
            { value: '6', label: '6 LFs' },
            { value: '10', label: '10 LFs' },
            { value: '16', label: '16 LFs' },
          ]}
          onChange={(v) => setNumLfs(Number(v))}
        />
        <Select
          label="LF accuracy range"
          value={noisePreset}
          options={NOISE_PRESETS}
          onChange={(v) => setNoisePreset(v as 'clean' | 'noisy' | 'adversarial')}
        />
        <Select
          label="Correlated duplicate copies"
          value={String(correlatedCopies)}
          options={[
            { value: '0', label: '0 (independent)' },
            { value: '2', label: '2 duplicates' },
            { value: '4', label: '4 duplicates' },
          ]}
          onChange={(v) => setCorrelatedCopies(Number(v))}
        />
      </ControlRow>

      <Plots>
        <Plot x={accAxisX} y={accAxisY} title="LF accuracy: true (dots) vs Snorkel estimated (bars)">
          <Bars x={lfIndices} y={estimatedAccuracies} slot={1} width={0.5} />
          <Points x={lfIndices} y={empiricalAccuracies} slot={0} size={6} />
        </Plot>

        <Plot x={methodAxisX} y={methodAxisY} title="End training label accuracy: Majority vs DS vs Snorkel">
          <Bars x={methodIndices} y={overallAccuracies} slot={0} width={0.6} />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Snorkel label accuracy"
          value={formatNumber(Number((res.accuracy.labelModel * 100).toFixed(1))) + '%'}
        />
        <Readout
          label="Majority vote accuracy"
          value={formatNumber(Number((res.accuracy.majority * 100).toFixed(1))) + '%'}
        />
        <Readout
          label="Modelling advantage"
          value={formatNumber(Number(((res.accuracy.labelModel - res.accuracy.majority) * 100).toFixed(1))) + '%'}
        />
        <Readout label="Label density d_Λ" value={formatNumber(Number(labelDensity.toFixed(2))) + ' votes/item'} />
      </ControlRow>
    </Figure>
  )
}
