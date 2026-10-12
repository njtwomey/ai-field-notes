import { useMemo } from 'react'
import { Bars, choice, Figure, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const ARCHETYPES = {
  fineweb: {
    label: 'FineWeb (aggressive web decanting)',
    baseTokens: 100,
    heuristicRetention: 0.58,
    qualityRetention: 0.72,
    dedupRetention: 0.62,
    piiRetention: 0.98,
  },
  refinedweb: {
    label: 'RefinedWeb (strict dedup & filtering)',
    baseTokens: 100,
    heuristicRetention: 0.52,
    qualityRetention: 0.68,
    dedupRetention: 0.55,
    piiRetention: 0.97,
  },
  c4: {
    label: 'C4 (heuristic-only pipeline)',
    baseTokens: 100,
    heuristicRetention: 0.61,
    qualityRetention: 0.9,
    dedupRetention: 0.85,
    piiRetention: 0.96,
  },
} as const

type ArchetypeKey = keyof typeof ARCHETYPES

const STAGES = ['1. Raw Crawl', '2. Heuristics', '3. Quality Filter', '4. Deduplication', '5. PII Scrubbing']

export function CorpusFunnel() {
  const state = useFigureState({
    archetype: choice<ArchetypeKey>(
      [
        { value: 'fineweb', label: 'FineWeb (high-grade web filtering)' },
        { value: 'refinedweb', label: 'RefinedWeb (strict MinHash dedup)' },
        { value: 'c4', label: 'C4 (heuristic cleanup)' },
      ],
      'fineweb',
      { label: 'pipeline profile' },
    ),
    qualityCutoff: slider(0.5, 0.95, 0.75, { step: 0.05, label: 'quality filter pass rate' }),
    dedupStrictness: slider(0.4, 0.9, 0.6, { step: 0.05, label: 'dedup retention rate' }),
  })

  const { archetype, qualityCutoff, dedupStrictness } = state
  const config = ARCHETYPES[archetype]

  const stageData = useMemo(() => {
    const raw = config.baseTokens
    const afterHeuristics = raw * config.heuristicRetention
    const afterQuality = afterHeuristics * (config.qualityRetention * (qualityCutoff / 0.75))
    const afterDedup = afterQuality * (config.dedupRetention * (dedupStrictness / 0.6))
    const finalTokens = afterDedup * config.piiRetention

    const values = [raw, afterHeuristics, afterQuality, afterDedup, finalTokens]
    return {
      values,
      retentionRate: (finalTokens / raw) * 100,
      tokensDiscarded: raw - finalTokens,
    }
  }, [config, qualityCutoff, dedupStrictness])

  const xAxis = useAxis({
    label: 'pipeline stage',
    categories: STAGES,
  })
  const yAxis = useAxis({ label: 'relative token volume (%)', range: [0, 110] })

  const xs = [0, 1, 2, 3, 4]

  return (
    <Figure
      title="Token attrition through the web corpus curation funnel"
      state={state}
      hoverReadout={false}
      caption="Token volume attrition across five successive stages of web corpus construction, starting from raw web crawl dumps (100%). Heuristic cleaning strips boilerplate and navigation fragments. Perplexity and fastText quality filters discard machine-translated and low-quality text. MinHash and exact line deduplication eliminate widespread web repetition, followed by PII scrubbing. Adjust the quality and deduplication strictness sliders to see their compounding impact on yield."
      readouts={
        <>
          <Readout label="final retention" value={`${stageData.retentionRate.toFixed(1)}%`} />
          <Readout label="tokens discarded" value={`${stageData.tokensDiscarded.toFixed(1)}%`} />
          <Readout label="dominant reduction" value="heuristics & dedup" />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="remaining tokens at each pipeline stage">
        <Bars name="remaining tokens" x={xs} y={stageData.values} width={0.6} slot={0} />
      </Plot>
    </Figure>
  )
}
