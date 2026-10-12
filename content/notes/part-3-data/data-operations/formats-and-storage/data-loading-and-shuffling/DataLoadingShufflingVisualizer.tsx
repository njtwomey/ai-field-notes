import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataLoadingShufflingVisualizer() {
  const state = useFigureState({
    strategy: choice<'length_bucketing' | 'naive_batching' | 'shuffle_window'>(
      [
        {
          value: 'length_bucketing',
          label: 'Length Bucketing (Sort & cluster variable sequences: slashes padding waste)',
        },
        {
          value: 'naive_batching',
          label: 'Naive Uniform Batching (Random batching padded to batch max: heavy padding)',
        },
        { value: 'shuffle_window', label: 'Shuffle Window Buffer (In-memory ring buffer vs correlation decay)' },
      ],
      'length_bucketing',
      { label: 'loading optimization regime' },
    ),
    parameterLevel: slider(1, 16, 8, {
      step: 1,
      label: 'bucketing granularity (number of discrete length bins K)',
    }),
  })

  const results = useMemo(() => {
    // Generate padding overhead or autocorrelation decay across sequence index or lag
    const xGrid: number[] = []
    const yGrid: number[] = []

    const K = state.parameterLevel

    for (let i = 1; i <= 16; i += 1) {
      xGrid.push(i)
      let metric = 0

      if (state.strategy === 'length_bucketing') {
        // As bucket count K increases, padding waste drops toward zero
        // Curve plotted against bucket count i in [1, 16]
        metric = 55.0 * Math.exp(-0.25 * (i - 1)) + 5.0
      } else if (state.strategy === 'naive_batching') {
        // Constant high padding waste ~55-60% across naive batching
        metric = 58.0 - 0.2 * i
      } else {
        // Autocorrelation decay across lag i under buffer size K * 200
        const tau = K * 1.5
        metric = 100.0 * Math.exp(-i / tau)
      }

      yGrid.push(metric)
    }

    let currentMetric = 0
    let computeUtilization = 0
    let effectiveSpeedup = '1.0x'
    let statusLabel = ''

    if (state.strategy === 'length_bucketing') {
      currentMetric = 55.0 * Math.exp(-0.25 * (K - 1)) + 5.0
      computeUtilization = 100.0 - currentMetric
      const speedup = 1.0 / (1.0 - (58.0 - currentMetric) / 100.0)
      effectiveSpeedup = `${speedup.toFixed(2)}x compute speedup`
      statusLabel = `Padding tokens: ${currentMetric.toFixed(1)}% of total sequence volume`
    } else if (state.strategy === 'naive_batching') {
      currentMetric = 58.0
      computeUtilization = 42.0
      effectiveSpeedup = '1.00x baseline'
      statusLabel = 'Severe padding waste: 58% of matrix FLOPs spent on zero pad tokens'
    } else {
      currentMetric = 100.0 * Math.exp(-1.0 / (K * 1.5))
      computeUtilization = Math.min(99, 50.0 + K * 3.0)
      effectiveSpeedup = `${(K * 250).toLocaleString()} samples in buffer`
      statusLabel = `Lag-1 autocorrelation: ${(currentMetric / 100).toFixed(2)} (0 = ideal uniform shuffle)`
    }

    return {
      xGrid,
      yGrid,
      currentX: K,
      currentMetric,
      computeUtilization,
      effectiveSpeedup,
      statusLabel,
    }
  }, [state.strategy, state.parameterLevel])

  const axisX = useAxis({
    label: state.strategy === 'shuffle_window' ? 'batch lag distance' : 'length bucket count K',
    range: [1, 16],
  })
  const axisY = useAxis({
    label: state.strategy === 'shuffle_window' ? 'sample autocorrelation (%)' : 'padding token overhead (%)',
    range: [0, 65],
  })

  return (
    <Figure
      title="Training loader dynamics: length bucketing, padding waste, and shuffle buffers"
      state={state}
      hoverReadout={false}
      caption="Dynamics of data loader optimization in deep learning pipelines. In variable-length sequence models (transformers, speech architectures), naive random batching pads every sequence to the maximum length in that batch, squandering over 55% of accelerator FLOPs on uninformative pad tokens. Length bucketing sorts samples into discrete length bins, reducing padding waste below 8% and yielding a >2x training speedup. In streaming sharded loaders, shuffle buffer capacity dictates how rapidly batch autocorrelation decays toward ideal uniform permutations."
      readouts={
        <>
          <Readout label="optimization state" value={results.statusLabel} />
          <Readout label="useful compute FLOPs" value={`${results.computeUtilization.toFixed(1)}%`} />
          <Readout label="throughput impact" value={results.effectiveSpeedup} />
        </>
      }
    >
      <Plot
        x={axisX}
        y={axisY}
        title={
          state.strategy === 'shuffle_window'
            ? 'Autocorrelation decay across batch lags'
            : 'Padding waste reduction as length buckets increase'
        }
      >
        <Curve
          name="overhead metric curve"
          x={results.xGrid}
          y={results.yGrid}
          slot={state.strategy === 'naive_batching' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current configuration"
          x={[results.currentX]}
          y={[results.currentMetric]}
          slot={state.strategy === 'naive_batching' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
