import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function PiiDetectionRedactionVisualizer() {
  const state = useFigureState({
    detectorType: choice<'regex_rules' | 'ner_model' | 'hybrid_ensemble'>(
      [
        { value: 'hybrid_ensemble', label: 'Hybrid Ensemble (Regex + NER + Context Verifier)' },
        { value: 'ner_model', label: 'Neural NER Token Classifier (RoBERTa / Presidio)' },
        { value: 'regex_rules', label: 'Regex Patterns & Checksums (Strict syntactic rules)' },
      ],
      'hybrid_ensemble',
      { label: 'detection strategy' },
    ),
    sanitizationAction: choice<'surrogate' | 'hard_mask'>(
      [
        { value: 'surrogate', label: 'Surrogate replacement (Synthetic entities: preserves LM perplexity)' },
        { value: 'hard_mask', label: 'Hard token masking ([REDACTED]: degrades language syntax)' },
      ],
      'surrogate',
      { label: 'sanitization mechanism' },
    ),
    corpusVolumeThousand: slider(10, 500, 100, {
      step: 10,
      label: 'corpus volume (thousands of documents)',
    }),
  })

  const results = useMemo(() => {
    // Total PII entities assumes ~0.5 PII entities per document
    const totalPii = state.corpusVolumeThousand * 500

    let precision = 0.98
    let recall = 0.65

    if (state.detectorType === 'regex_rules') {
      precision = 0.99
      recall = 0.62 // Misses unstructured names, clinic names, unusual address formats
    } else if (state.detectorType === 'ner_model') {
      precision = 0.93
      recall = 0.89 // Catches names, but misses rare spellings and edge phone numbers
    } else {
      precision = 0.96
      recall = 0.975 // State-of-the-art hybrid ensemble
    }

    const detectedPii = Math.round(totalPii * recall)
    const falseNegatives = totalPii - detectedPii // Leaked PII!
    const falsePositives = Math.round((detectedPii * (1 - precision)) / precision)

    // Language modeling perplexity degradation
    // Hard masking creates unnatural syntax breaks; surrogate replacement maintains smooth grammar
    const perplexityPenalty =
      state.sanitizationAction === 'hard_mask' ? 4.2 + (detectedPii / 100000) * 1.5 : 0.3 + (detectedPii / 100000) * 0.1

    // Curve data: Recall vs Residual Leaks as volume scales from 10k to 500k
    const volGrid: number[] = []
    const leaksGrid: number[] = []
    for (let v = 10; v <= 500; v += 20) {
      volGrid.push(v)
      const leaks = Math.round(v * 500 * (1 - recall))
      leaksGrid.push(leaks)
    }

    return {
      totalPii,
      detectedPii,
      falseNegatives,
      falsePositives,
      precision,
      recall,
      perplexityPenalty,
      volGrid,
      leaksGrid,
      currentLeaks: falseNegatives,
    }
  }, [state.detectorType, state.sanitizationAction, state.corpusVolumeThousand])

  const xAxis = useAxis({
    label: 'corpus document volume (thousands of records)',
    range: [0, 500],
  })
  const yAxis = useAxis({
    label: 'residual exposed PII entities (false negatives)',
    range: [0, Math.max(1000, results.leaksGrid[results.leaksGrid.length - 1] * 1.1)],
  })

  return (
    <Figure
      title="Automated PII detection and redaction: recall limits, residual leakage, and syntax preservation"
      state={state}
      hoverReadout={false}
      caption="Evaluating PII detection pipelines on pretraining text corpora. Because detection recall is never 100%, scaling corpus volume linearly multiplies the absolute number of leaked PII records: even an ensemble achieving 97.5% recall leaves over 1,250 real personal identities exposed per 100,000 documents. Furthermore, surrogate replacement preserves natural language perplexity, whereas hard masking ([REDACTED]) degrades syntactic dependencies."
      readouts={
        <>
          <Readout label="detector recall" value={`${(results.recall * 100).toFixed(1)}%`} />
          <Readout label="detector precision" value={`${(results.precision * 100).toFixed(1)}%`} />
          <Readout
            label="unredacted PII leakage"
            value={`${results.falseNegatives.toLocaleString()} exposed entities`}
          />
          <Readout
            label="LM perplexity impact"
            value={
              state.sanitizationAction === 'surrogate'
                ? `Minimal (+${results.perplexityPenalty.toFixed(2)} PPL)`
                : `Degraded (+${results.perplexityPenalty.toFixed(2)} PPL)`
            }
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Unsanitized PII entity leakage as a function of corpus scale">
        <Curve
          name="unredacted PII leakage trajectory (false negatives)"
          x={results.volGrid}
          y={results.leaksGrid}
          slot={1}
          width={2.5}
        />
        <Points
          name="current operating scale leakage"
          x={[state.corpusVolumeThousand]}
          y={[results.currentLeaks]}
          slot={1}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
