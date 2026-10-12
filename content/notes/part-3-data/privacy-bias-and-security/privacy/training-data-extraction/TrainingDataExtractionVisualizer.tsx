import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TrainingDataExtractionVisualizer() {
  const state = useFigureState({
    modelScale: choice<'7b' | '70b'>(
      [
        { value: '70b', label: '70B Parameter Model (High memorization capacity)' },
        { value: '7b', label: '7B Parameter Model (Moderate memorization capacity)' },
      ],
      '70b',
      { label: 'model capacity' },
    ),
    dedupState: choice<'raw' | 'deduplicated'>(
      [
        { value: 'raw', label: 'Raw Web Crawl (Unfiltered duplicates C in [1, 50])' },
        { value: 'deduplicated', label: 'Deduplicated Corpus (Near-dedup caps C = 1: Lee 2022)' },
      ],
      'raw',
      { label: 'corpus preprocessing' },
    ),
    duplicationCount: slider(1, 40, 10, { step: 1, label: 'raw sequence duplicate count C' }),
  })

  const results = useMemo(() => {
    // Generate extraction probability curves across duplication count C in [1, 40]
    const cGrid: number[] = []
    const prob70bGrid: number[] = []
    const prob7bGrid: number[] = []

    for (let c = 1; c <= 40; c++) {
      cGrid.push(c)
      // Super-linear scaling: P(extract | C) = 1 - exp(-k * C^1.4)
      const p70 = Math.min(100, 100 * (1 - Math.exp(-0.045 * c ** 1.35)))
      const p7 = Math.min(100, 100 * (1 - Math.exp(-0.015 * c ** 1.25)))
      prob70bGrid.push(p70)
      prob7bGrid.push(p7)
    }

    // Evaluated at user slider
    const effectiveC = state.dedupState === 'deduplicated' ? 1 : state.duplicationCount
    const is70b = state.modelScale === '70b'

    const currentProb = is70b
      ? Math.min(100, 100 * (1 - Math.exp(-0.045 * effectiveC ** 1.35)))
      : Math.min(100, 100 * (1 - Math.exp(-0.015 * effectiveC ** 1.25)))

    let riskLevel = 'Low Risk (< 5% extraction)'
    if (currentProb > 60) {
      riskLevel = 'Critical (High Verbatim Extraction Risk)'
    } else if (currentProb > 25) {
      riskLevel = 'Elevated Risk'
    }

    return {
      cGrid,
      activeGrid: is70b ? prob70bGrid : prob7bGrid,
      effectiveC,
      currentProb,
      riskLevel,
    }
  }, [state.modelScale, state.dedupState, state.duplicationCount])

  const axisX = useAxis({
    label: 'sequence occurrence frequency in training corpus C (duplicates)',
    range: [1, 40],
  })
  const axisY = useAxis({
    label: 'verbatim training data extraction probability (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Training-data extraction: duplicate amplification and super-linear memorization"
      state={state}
      hoverReadout={false}
      caption="Evaluating training data extraction risk in large language models. The probability of extracting a verbatim training sequence scales super-linearly with its duplication count C in the pretraining corpus. Larger models (70B) memorize duplicated sequences at significantly lower thresholds. Applying MinHash deduplication caps sequence multiplicity to C=1, suppressing extraction risk by over 80%."
      readouts={
        <>
          <Readout label="effective duplicate count" value={`${results.effectiveC} copies in corpus`} />
          <Readout label="verbatim extraction probability" value={`${results.currentProb.toFixed(1)}%`} />
          <Readout label="privacy risk classification" value={results.riskLevel} />
          <Readout
            label="deduplication benefit"
            value={
              state.dedupState === 'deduplicated' ? 'Protected by Deduplication (C=1)' : 'Unmitigated Duplication Risk'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Extraction probability as a function of training duplication count">
        <Curve
          name="verbatim extraction probability curve P(extract | C)"
          x={results.cGrid}
          y={results.activeGrid}
          slot={0}
          width={2.5}
        />
        <Points
          name="current sequence operating point"
          x={[results.effectiveC]}
          y={[results.currentProb]}
          slot={1}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
