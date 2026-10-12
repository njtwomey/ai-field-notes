import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ContentFilteringVisualizer() {
  const state = useFigureState({
    filteringStrategy: choice<'blocklist' | 'classifier' | 'contextual'>(
      [
        { value: 'classifier', label: 'Toxicity Classifier (e.g. Perspective API threshold: Gehman et al. 2020)' },
        { value: 'blocklist', label: 'Static Keyword Blocklist (e.g. C4 "bad-words.py": Dodge et al. 2021)' },
        { value: 'contextual', label: 'Dialect-Aware Contextual Filter (Disambiguated reclamation)' },
      ],
      'classifier',
      { label: 'filtering mechanism' },
    ),
    threshold: slider(0.1, 0.9, 0.5, {
      step: 0.05,
      label: 'classifier toxicity threshold (tau)',
    }),
  })

  const results = useMemo(() => {
    // Generate FPR curves for AAVE vs SAE across threshold tau in [0.1, 0.9]
    const tauGrid: number[] = []
    const aaveFprGrid: number[] = []
    const saeFprGrid: number[] = []

    for (let t = 0.1; t <= 0.9; t += 0.05) {
      tauGrid.push(t)
      let aFpr = 0
      let sFpr = 0

      if (state.filteringStrategy === 'blocklist') {
        // Static blocklist has fixed high error rates independent of threshold
        aFpr = 14.5
        sFpr = 2.1
      } else if (state.filteringStrategy === 'contextual') {
        // Contextual reduces dialect disparity gap
        aFpr = 18.0 * Math.exp(-3.2 * t) + 1.2
        sFpr = 8.0 * Math.exp(-3.5 * t) + 0.5
      } else {
        // Standard classifier exhibits strong false positive penalty on AAVE
        aFpr = 48.0 * Math.exp(-2.8 * t) + 3.0
        sFpr = 16.0 * Math.exp(-3.5 * t) + 0.8
      }

      aaveFprGrid.push(aFpr)
      saeFprGrid.push(sFpr)
    }

    const t = state.threshold
    let trueToxicRemoved = 0
    let currentAaveFpr = 0
    let currentSaeFpr = 0
    let lgbtqExclusionRate = 0

    if (state.filteringStrategy === 'blocklist') {
      trueToxicRemoved = 62.0
      currentAaveFpr = 14.5
      currentSaeFpr = 2.1
      lgbtqExclusionRate = 34.0 // High false positive on sex-ed and queer terms
    } else if (state.filteringStrategy === 'contextual') {
      trueToxicRemoved = Math.min(98, 100 * (1.0 - 0.45 * t))
      currentAaveFpr = 18.0 * Math.exp(-3.2 * t) + 1.2
      currentSaeFpr = 8.0 * Math.exp(-3.5 * t) + 0.5
      lgbtqExclusionRate = 4.2
    } else {
      trueToxicRemoved = Math.min(99, 105 * (1.0 - 0.55 * t))
      currentAaveFpr = 48.0 * Math.exp(-2.8 * t) + 3.0
      currentSaeFpr = 16.0 * Math.exp(-3.5 * t) + 0.8
      lgbtqExclusionRate = 18.5 * Math.exp(-1.8 * t)
    }

    const disparityRatio = currentAaveFpr / Math.max(0.1, currentSaeFpr)

    return {
      tauGrid,
      aaveFprGrid,
      saeFprGrid,
      currentTau: t,
      trueToxicRemoved,
      currentAaveFpr,
      currentSaeFpr,
      lgbtqExclusionRate,
      disparityRatio,
    }
  }, [state.filteringStrategy, state.threshold])

  const axisX = useAxis({
    label: 'classifier toxicity threshold tau',
    range: [0.1, 0.9],
  })
  const axisY = useAxis({
    label: 'false positive discard rate on benign text (%)',
    range: [0, 45],
  })

  return (
    <Figure
      title="Toxicity filtering, dialect disparities, and collateral corpus erasure"
      state={state}
      hoverReadout={false}
      caption="Evaluating collateral damage in pre-training data sanitization. Keyword blocklists (e.g. C4 bad-words list: Dodge et al. 2021) and automated toxicity classifiers (Gehman et al. 2020) disproportionately tag African American Vernacular English (AAVE) and LGBTQ+ health/identity documentation as toxic. While aggressive thresholds eliminate overt hate speech, they induce a 3-4x higher false positive rate on minority dialects, systematically erasing marginalized communities from language model pre-training corpora."
      readouts={
        <>
          <Readout label="toxic text eliminated (recall)" value={`${results.trueToxicRemoved.toFixed(1)}%`} />
          <Readout label="AAVE false discard rate" value={`${results.currentAaveFpr.toFixed(1)}%`} />
          <Readout
            label="disparity ratio (AAVE / SAE)"
            value={`${results.disparityRatio.toFixed(1)}x higher penalty`}
          />
          <Readout label="LGBTQ+ resource exclusion" value={`${results.lgbtqExclusionRate.toFixed(1)}% purged`} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Benign false discard rate on AAVE vs standard English">
        <Curve
          name="African American Vernacular English (AAVE) false discard"
          x={results.tauGrid}
          y={results.aaveFprGrid}
          slot={1}
          width={2.5}
        />
        <Curve
          name="Standard American English (SAE) false discard"
          x={results.tauGrid}
          y={results.saeFprGrid}
          slot={0}
          width={2}
          dashed
        />
        <Points
          name="current AAVE operating point"
          x={[results.currentTau]}
          y={[results.currentAaveFpr]}
          slot={1}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
