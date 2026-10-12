import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function LargeScaleProcessingVisualizer() {
  const state = useFigureState({
    processingParadigm: choice<'count_min_sketch' | 'join_skew' | 'out_of_core'>(
      [
        {
          value: 'count_min_sketch',
          label: 'Count-Min Sketch (Sub-linear memory frequency estimation: Cormode & Muthukrishnan 2005)',
        },
        { value: 'join_skew', label: 'Join Key Skew & Stragglers (Zipfian power-law key distribution)' },
        { value: 'out_of_core', label: 'Out-of-Core Memory Streaming (Chunked disk spilling vs RAM bound)' },
      ],
      'count_min_sketch',
      { label: 'large-scale computational challenge' },
    ),
    scaleParameter: slider(100, 2000, 500, {
      step: 100,
      label: 'configuration scale (sketch width w / partition count)',
    }),
  })

  const results = useMemo(() => {
    // Generate trade-off curves across parameter W in [100, 2000]
    const wGrid: number[] = []
    const yGrid: number[] = []

    for (let w = 100; w <= 2000; w += 100) {
      wGrid.push(w)
      let metric = 0

      if (state.processingParadigm === 'count_min_sketch') {
        // Point query overestimation error epsilon = e / w * 100%
        metric = (Math.E / w) * 100.0
      } else if (state.processingParadigm === 'join_skew') {
        // Maximum partition straggler runtime under skewed distribution (seconds)
        metric = 15.0 + 350.0 / Math.pow(w / 100, 0.4)
      } else {
        // Peak resident set size RAM footprint (MB)
        metric = 32.0 + (w / 2000) * 180.0
      }

      yGrid.push(metric)
    }

    const currentW = state.scaleParameter
    let currentMetric = 0
    let memoryFootprint = ''
    let operationalStatus = ''

    if (state.processingParadigm === 'count_min_sketch') {
      currentMetric = (Math.E / currentW) * 100.0
      const depth = 5
      const ramBytes = currentW * depth * 4
      memoryFootprint = `${(ramBytes / 1024).toFixed(1)} KB (vs 8.0 GB exact hash map)`
      operationalStatus = `Frequency error bound: ε = ${(currentMetric / 100).toFixed(4)} with 99.3% confidence`
    } else if (state.processingParadigm === 'join_skew') {
      currentMetric = 15.0 + 350.0 / Math.pow(currentW / 100, 0.4)
      memoryFootprint = `${currentW} distributed partitions`
      operationalStatus = `Longest straggler partition: ${currentMetric.toFixed(0)}s (bottlenecked by top join key)`
    } else {
      currentMetric = 32.0 + (currentW / 2000) * 180.0
      memoryFootprint = `${currentMetric.toFixed(0)} MB resident memory`
      operationalStatus = `Stable execution: 100% immune to out-of-core OOM crashes`
    }

    return {
      wGrid,
      yGrid,
      currentW,
      currentMetric,
      memoryFootprint,
      operationalStatus,
    }
  }, [state.processingParadigm, state.scaleParameter])

  const axisX = useAxis({
    label:
      state.processingParadigm === 'count_min_sketch'
        ? 'sketch width table columns w'
        : 'distributed partitions / buffer scale',
    range: [100, 2000],
  })
  const axisY = useAxis({
    label:
      state.processingParadigm === 'count_min_sketch'
        ? 'point query error bound epsilon (%)'
        : 'straggler runtime / buffer MB',
    range: [0, Math.max(120, Math.ceil(results.yGrid[0] || 100))],
  })

  return (
    <Figure
      title="Large-scale data processing: sketches, distributed skew, and out-of-core pipelines"
      state={state}
      hoverReadout={false}
      caption="Evaluating architectural strategies for data exceeding main memory. Streaming sketches (Count-Min Sketch: Cormode & Muthukrishnan 2005) estimate frequency statistics in constant sub-linear memory with provable error bounds ε = e/w, avoiding multi-gigabyte hash tables. In distributed clusters (MapReduce: Dean & Ghemawat 2004), skewed join keys produce severe stragglers where a single partition bottlenecks the entire cluster. Out-of-core streaming engines process terabyte-scale datasets on a single machine by streaming chunked columnar blocks through bounded RAM buffers."
      readouts={
        <>
          <Readout label="memory footprint" value={results.memoryFootprint} />
          <Readout label="operational metric" value={`${results.currentMetric.toFixed(2)}`} />
          <Readout label="runtime status" value={results.operationalStatus} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Algorithm performance vs configuration scale">
        <Curve name="system performance curve" x={results.wGrid} y={results.yGrid} slot={0} width={2.5} />
        <Points
          name="current scale configuration"
          x={[results.currentW]}
          y={[results.currentMetric]}
          slot={0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
