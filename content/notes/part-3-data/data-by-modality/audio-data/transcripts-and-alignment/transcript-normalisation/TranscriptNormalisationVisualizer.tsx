import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TranscriptNormalisationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'wer_sensitivity' | 'numeral_rules' | 'disfluency_impact'>(
      [
        {
          value: 'wer_sensitivity',
          label: 'Evaluation Sensitivity: Normalization Pipeline vs Measured WER (Radford 2023)',
        },
        {
          value: 'numeral_rules',
          label: 'Non-Standard Words: Digit Formats vs Verbalized Spoken Length (Sproat 2001)',
        },
        { value: 'disfluency_impact', label: 'Disfluency Mismatch: False Deletions vs Ground Truth Retention' },
      ],
      'wer_sensitivity',
      { label: 'normalization perspective' },
    ),
    acousticErrorBase: slider(4, 20, 8, {
      step: 1,
      label: 'underlying acoustic WER (%)',
    }),
    nswDensity: slider(2, 16, 8, {
      step: 2,
      label: 'non-standard words / numbers per 100 words',
    }),
  })

  const { visualizerMode, acousticErrorBase, nswDensity } = state

  const results = useMemo(() => {
    if (visualizerMode === 'wer_sensitivity') {
      // 5 progressive normalization stages:
      // 1: Raw verbatim (punctuated, capitalized, digits)
      // 2: Lowercase only
      // 3: Punctuation stripped
      // 4: Number verbalization ($5 -> five dollars)
      // 5: Full Whisper normalizer (spelling harmonization + contraction expansion)
      const stages = [1, 2, 3, 4, 5]
      // Penalty added by formatting mismatches depends on nswDensity
      const formatPenalty = nswDensity * 0.9
      const werScores = [
        acousticErrorBase + formatPenalty * 2.2,
        acousticErrorBase + formatPenalty * 1.6,
        acousticErrorBase + formatPenalty * 1.0,
        acousticErrorBase + formatPenalty * 0.35,
        acousticErrorBase, // True underlying acoustic WER
      ]
      const cleanAcousticBaseline = stages.map(() => acousticErrorBase)

      const rawWER = werScores[0]
      const fullyNormalizedWER = werScores[4]
      const phantomGap = rawWER - fullyNormalizedWER

      return {
        x: stages,
        y: werScores,
        baseline: cleanAcousticBaseline,
        stat1: `Acoustic base WER: ${acousticErrorBase}%`,
        stat2: `Raw unnormalized WER: ${rawWER.toFixed(1)}%`,
        stat3: `Phantom WER gap: +${phantomGap.toFixed(1)}%`,
        warning:
          phantomGap >= 8
            ? 'Severe evaluation distortion: Over 50% of reported word errors stem from formatting discrepancies, not acoustic mistakes'
            : 'Standardized normalization isolates genuine acoustic speech recognition performance',
        axisXLabel: 'Normalization Stage (1=Raw, 3=No Punct, 4=Numbers, 5=Whisper Norm)',
        axisYLabel: 'Measured Word Error Rate WER (%)',
        xRange: [0.8, 5.2] as [number, number],
        yRange: [0, Math.max(...werScores) * 1.15] as [number, number],
      }
    } else if (visualizerMode === 'numeral_rules') {
      // Numeric formats: 1: Cardinal (42), 2: Ordinal (1st), 3: Currency ($100), 4: Year (1998), 5: Decimals (3.14), 6: Time (8:45)
      const formats = [1, 2, 3, 4, 5, 6]
      const writtenTokenCount = [1, 1, 1, 1, 1, 1]
      // Verbalized spoken word counts:
      // "forty two" (2), "first" (1), "one hundred dollars" (3), "nineteen ninety eight" (3), "three point one four" (4), "eight forty five" (3)
      const spokenWordCounts = [2, 1, 3, 3, 4, 3]

      return {
        x: formats,
        y: spokenWordCounts,
        baseline: writtenTokenCount,
        stat1: 'Non-Standard Words (NSWs)',
        stat2: 'Mean word expansion: 2.7x',
        stat3: 'Verbalization Active',
        warning:
          'A single written digit token ($100) expands into multiple spoken words; evaluating without expansion creates false substitution errors',
        axisXLabel: 'Numeric Format Category (1=Card, 2=Ord, 3=$, 4=Year, 5=Dec, 6=Time)',
        axisYLabel: 'Equivalent Spoken Word Count',
        xRange: [0.5, 6.5] as [number, number],
        yRange: [0, 5.5] as [number, number],
      }
    } else {
      // Disfluency density per 100 words (0 to 12 filled pauses: um, uh, repetitions)
      const disfluencyCounts = [1, 2, 4, 6, 8, 10, 12]
      // If reference includes disfluencies but ASR omits them: Deletion error rate increases linearly
      const deletionErrorRate = disfluencyCounts.map((d) => (d / (100 + d)) * 100)
      const zeroDisfluencyBaseline = disfluencyCounts.map(() => 0)

      const currDelError = (nswDensity / (100 + nswDensity)) * 100

      return {
        x: disfluencyCounts,
        y: deletionErrorRate,
        baseline: zeroDisfluencyBaseline,
        stat1: `Disfluency rate: ${nswDensity} per 100 words`,
        stat2: `Induced deletion error: ${currDelError.toFixed(1)}%`,
        stat3: 'Disfluency Mismatch',
        warning:
          'If human reference transcripts retain "um" and "uh" but the ASR model suppresses them, deletion error increases directly',
        axisXLabel: 'Spoken Disfluencies per 100 Words (um, uh, repeats)',
        axisYLabel: 'Artificial Deletion Error Rate (%)',
        xRange: [0.5, 12.5] as [number, number],
        yRange: [0, 15] as [number, number],
      }
    }
  }, [visualizerMode, acousticErrorBase, nswDensity])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Transcript Normalisation: Normalization Pipelines, Numeral Verbalization, and WER Sensitivity"
      state={state}
      hoverReadout={false}
      caption="The impact of text normalization on ASR benchmarks: The dramatic swing in measured Word Error Rate across preprocessing stages, numeric verbalization expansion, and disfluency deletion penalties."
      readouts={
        <>
          <Readout label="Benchmark Setup" value={results.stat1} />
          <Readout label="Observed Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'wer_sensitivity'
                ? 'Phantom WER'
                : visualizerMode === 'numeral_rules'
                  ? 'Expansion Status'
                  : 'Deletion Penalty'
            }
            value={results.stat3}
            color={visualizerMode === 'wer_sensitivity' ? 'var(--red-600)' : 'var(--blue-600)'}
          />
          <Readout label="Evaluation Rule" value={results.warning} color="var(--amber-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Transcript Normalization Profile">
        <Curve
          name={
            visualizerMode === 'wer_sensitivity'
              ? 'True Acoustic Error Floor (Fully Normalized)'
              : visualizerMode === 'numeral_rules'
                ? 'Raw Written Representation (1 Token)'
                : 'Zero Deletion Benchmark'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'wer_sensitivity'
              ? 'Measured Word Error Rate (WER)'
              : visualizerMode === 'numeral_rules'
                ? 'Expanded Spoken Word Count'
                : 'False Deletion Error Rate (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Evaluation Stages" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
