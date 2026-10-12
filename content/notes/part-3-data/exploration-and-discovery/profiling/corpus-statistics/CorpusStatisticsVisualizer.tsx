import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function CorpusStatisticsVisualizer() {
  const state = useFigureState({
    corpusMode: choice<'length' | 'zipf' | 'heaps'>(
      [
        { value: 'length', label: 'Document Length & Context Truncation' },
        { value: 'zipf', label: 'Zipf’s Law (Term Frequency vs Rank)' },
        { value: 'heaps', label: 'Heaps’ Law (Vocabulary Growth Scaling)' },
      ],
      'length',
      { label: 'corpus profiling metric' },
    ),
    contextLimit: slider(256, 4096, 1024, {
      step: 256,
      label: 'transformer context limit (L_max)',
    }),
    zipfSlope: slider(0.7, 1.4, 1.0, {
      step: 0.05,
      label: 'Zipf power-law exponent (s)',
    }),
    heapsExponent: slider(0.35, 0.7, 0.52, {
      step: 0.02,
      label: 'Heaps law exponent (beta)',
    }),
  })

  const { corpusMode, contextLimit, zipfSlope, heapsExponent } = state

  const results = useMemo(() => {
    if (corpusMode === 'length') {
      // Document length distribution (lognormal density sampled at 20 length points from 100 to 5000)
      const lengths = [
        100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2400, 2800, 3200, 3600, 4000, 4500, 5000,
      ]
      // Lognormal parameters: mu = 6.8 (mean ~900 tokens), sigma = 0.8
      const densities = lengths.map((l) => {
        const logl = Math.log(l)
        const d = (1 / (l * 0.8 * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow(logl - 6.8, 2) / (2 * 0.64))
        return d * 150000 // scaled frequency %
      })

      // Truncation line at contextLimit
      const cutoff = lengths.map((l) => (l === contextLimit ? 50 : 0))

      // Estimate truncated document fraction
      const truncatedPct =
        lengths
          .filter((l) => l > contextLimit)
          .reduce((sum, _, i) => sum + (densities[lengths.length - 1 - i] || 0), 0) / 10

      return {
        x: lengths,
        y: densities,
        baseline: cutoff,
        truncatedPct: Math.min(95, Math.max(1, truncatedPct)),
      }
    } else if (corpusMode === 'zipf') {
      // Ranks 1 to 20 on log scale (log10(rank) vs log10(freq))
      // freq(r) = C / r^s
      const ranks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
      const logRanks = ranks.map((r) => Math.log10(r))
      const logFreqs = ranks.map((r) => 5.0 - zipfSlope * Math.log10(r))
      const idealSlope1 = ranks.map((r) => 5.0 - 1.0 * Math.log10(r))

      return {
        x: logRanks,
        y: logFreqs,
        baseline: idealSlope1,
        truncatedPct: 0,
      }
    } else {
      // Heaps law: V(N) = 30 * N^beta for N in [1k, 1M]
      // N sampled at 12 log points: 1k to 1M
      const nValues = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
      const logN = nValues.map((n) => Math.log10(n))
      const vocabSize = nValues.map((n) => Math.log10(30 * Math.pow(n, heapsExponent)))
      const linearBound = nValues.map((n) => Math.log10(n * 0.3))

      return {
        x: logN,
        y: vocabSize,
        baseline: linearBound,
        truncatedPct: 0,
      }
    }
  }, [corpusMode, contextLimit, zipfSlope, heapsExponent])

  const axisX = useAxis({
    label:
      corpusMode === 'length'
        ? 'Document Sequence Length (Tokens)'
        : corpusMode === 'zipf'
          ? 'Log10 Term Rank (1 to 1,000)'
          : 'Log10 Total Token Volume (1k to 1M Tokens)',
    range: corpusMode === 'length' ? [0, 5200] : corpusMode === 'zipf' ? [0, 3.2] : [3.0, 6.2],
  })

  const axisY = useAxis({
    label:
      corpusMode === 'length'
        ? 'Relative Token Frequency / Density'
        : corpusMode === 'zipf'
          ? 'Log10 Empirical Frequency'
          : 'Log10 Unique Vocabulary Size (Types)',
    range: corpusMode === 'length' ? [0, 80] : corpusMode === 'zipf' ? [1.0, 5.5] : [2.0, 5.5],
  })

  return (
    <Figure
      title="Corpus Statistics & Lexical Scaling Explorer"
      state={state}
      hoverReadout={false}
      caption="Profiling textual corpora tracks sequence length distributions, context window truncation loss, Zipfian word frequency dropoff, and Heaps law sublinear vocabulary growth."
      readouts={
        <>
          {corpusMode === 'length' && (
            <>
              <Readout label="Context Cutoff" value={`${contextLimit} tokens`} />
              <Readout
                label="Truncated Tail Loss"
                value={`${results.truncatedPct.toFixed(1)}% of docs clipped`}
                color={results.truncatedPct > 20 ? 'var(--red-600)' : 'var(--blue-600)'}
              />
              <Readout
                label="Padding Efficiency"
                value={contextLimit > 2048 ? 'High padding overhead on short texts' : 'Balanced context'}
              />
            </>
          )}
          {corpusMode === 'zipf' && (
            <>
              <Readout label="Power-Law Exponent (s)" value={`${zipfSlope.toFixed(2)}`} />
              <Readout label="Hapax Legomena Mass" value="~50% of distinct words appear exactly once" />
            </>
          )}
          {corpusMode === 'heaps' && (
            <>
              <Readout label="Heaps Growth Exponent" value={`${heapsExponent.toFixed(2)}`} />
              <Readout
                label="Vocab at 1M Tokens"
                value={`~${Math.round(Math.pow(10, results.y[results.y.length - 1])).toLocaleString()} subword types`}
              />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Corpus Statistical Profile">
        <Curve name="Empirical Distribution / Fitted Curve" x={results.x} y={results.y} slot={0} width={2.5} />
        <Points
          name="Measurement Points"
          x={results.x}
          y={results.y}
          slot={corpusMode === 'length' && results.truncatedPct > 20 ? 1 : 0}
          size={7}
        />
        {corpusMode !== 'length' && (
          <Curve name="Theoretical Reference Slope" x={results.x} y={results.baseline} slot={2} dashed width={1.8} />
        )}
      </Plot>
    </Figure>
  )
}
