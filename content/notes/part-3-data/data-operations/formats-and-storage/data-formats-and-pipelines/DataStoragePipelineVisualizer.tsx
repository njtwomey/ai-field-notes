import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataStoragePipelineVisualizer() {
  const state = useFigureState({
    storageFormat: choice<'row' | 'columnar' | 'array' | 'sharded'>(
      [
        { value: 'sharded', label: 'Sharded Sequential Records (WebDataset / TFRecord: Linear streaming)' },
        { value: 'columnar', label: 'Columnar File (Apache Parquet: Projection pushdown, compressed chunks)' },
        { value: 'array', label: 'Chunked N-D Array (Zarr / HDF5: Multidimensional tensor slices)' },
        { value: 'row', label: 'Row Text Records (JSON Lines / CSV: Heavy text parsing overhead)' },
      ],
      'sharded',
      { label: 'data storage format' },
    ),
    ioWorkers: slider(1, 16, 4, {
      step: 1,
      label: 'asynchronous I/O prefetch worker processes',
    }),
  })

  const results = useMemo(() => {
    // Generate throughput curves across workers W in [1, 16]
    const wGrid: number[] = []
    const throughputGrid: number[] = []
    const gpuStallGrid: number[] = []

    // GPU consumption bandwidth requirement: e.g. 1200 MB/s to saturate compute
    const gpuRequirement = 1200.0

    for (let w = 1; w <= 16; w += 1) {
      wGrid.push(w)
      let baseThroughputPerWorker = 0

      if (state.storageFormat === 'sharded') {
        // High sequential streaming bandwidth, minimal random IOPS
        baseThroughputPerWorker = 180.0
      } else if (state.storageFormat === 'columnar') {
        // Efficient decompression, moderate decode overhead
        baseThroughputPerWorker = 120.0
      } else if (state.storageFormat === 'array') {
        // Direct buffer memcpy, good chunk alignment
        baseThroughputPerWorker = 140.0
      } else {
        // Severe text parsing & string allocation bottleneck (Python GIL / JSON decode)
        baseThroughputPerWorker = 28.0
      }

      // Diminishing returns from disk/network bus saturation and IPC overhead
      const totalThroughput = (baseThroughputPerWorker * w) / (1.0 + 0.04 * (w - 1))
      throughputGrid.push(totalThroughput)

      const stallPct = Math.max(0, 100.0 * (1.0 - totalThroughput / gpuRequirement))
      gpuStallGrid.push(stallPct)
    }

    const currentW = state.ioWorkers
    const currentIdx = currentW - 1
    const currentThroughput = throughputGrid[currentIdx]
    const currentGpuStall = gpuStallGrid[currentIdx]

    let compressionRatio = '1.0x (Baseline)'
    let accessPattern = 'Sequential streaming'

    if (state.storageFormat === 'sharded') {
      compressionRatio = '2.8x (Tar / Zstandard sharded)'
      accessPattern = 'High-throughput sequential linear streaming'
    } else if (state.storageFormat === 'columnar') {
      compressionRatio = '4.2x (Parquet Snappy / Dictionary encoding)'
      accessPattern = 'Selective column projection pushdown'
    } else if (state.storageFormat === 'array') {
      compressionRatio = '2.1x (Blosc / Zstandard chunked arrays)'
      accessPattern = 'Multidimensional chunk strided slices'
    } else {
      compressionRatio = '1.2x (Gzip JSONL line streams)'
      accessPattern = 'Full record deserialization on every row'
    }

    return {
      wGrid,
      throughputGrid,
      gpuStallGrid,
      currentW,
      currentThroughput,
      currentGpuStall,
      compressionRatio,
      accessPattern,
    }
  }, [state.storageFormat, state.ioWorkers])

  const axisX = useAxis({
    label: 'concurrent I/O prefetch workers',
    range: [1, 16],
  })
  const axisY = useAxis({
    label: 'pipeline ingest throughput (MB/s)',
    range: [0, 2200],
  })

  return (
    <Figure
      title="Storage format layout and data pipeline throughput dynamics"
      state={state}
      hoverReadout={false}
      caption="Evaluating storage layouts and data pipeline throughput. Sequential sharded formats (WebDataset / TFRecord) maximize linear streaming bandwidth from local NVMe or cloud object storage, bypassing random IOPS bottlenecks to prevent GPU starvation (Mohan et al. 2021). Columnar formats (Apache Parquet) excel at feature engineering via column projection and dictionary compression. Text formats (JSON Lines, CSV) incur severe CPU parsing overheads, bottlenecking modern accelerators even with multiple worker processes."
      readouts={
        <>
          <Readout label="ingest throughput" value={`${results.currentThroughput.toFixed(0)} MB/s`} />
          <Readout
            label="GPU data stall fraction"
            value={`${results.currentGpuStall.toFixed(1)}% idle waiting for I/O`}
          />
          <Readout label="compression efficiency" value={results.compressionRatio} />
          <Readout label="optimal access pattern" value={results.accessPattern} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Ingest throughput across parallel loader workers">
        <Curve name="pipeline throughput (MB/s)" x={results.wGrid} y={results.throughputGrid} slot={0} width={2.5} />
        <Points
          name="current worker configuration"
          x={[results.currentW]}
          y={[results.currentThroughput]}
          slot={0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
