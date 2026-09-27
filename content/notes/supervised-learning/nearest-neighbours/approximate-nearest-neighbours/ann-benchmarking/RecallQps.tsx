import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

/**
 * Measured with faiss-cpu 1.15.1 on one thread of an Apple M1 Pro: 10^5 vectors and 1000 queries in 64 dimensions from
 * a mixture of 256 Gaussian clusters (the data of the FAISS note), best of three timings. Each row: 10-recall@10, 1-recall@1, queries per
 * second, and the search parameter.
 */
const RUNS: { name: string; knob: string; rows: [number, number, number, number][] }[] = [
  {
    name: 'IVF1024,Flat',
    knob: 'nprobe',
    rows: [
      [0.621, 0.681, 136590, 1],
      [0.851, 0.891, 111458, 2],
      [0.977, 0.99, 88625, 4],
      [1.0, 1.0, 66083, 8],
      [1.0, 1.0, 40899, 16],
      [1.0, 1.0, 24736, 32],
      [1.0, 1.0, 13434, 64],
    ],
  },
  {
    name: 'HNSW32',
    knob: 'efSearch',
    rows: [
      [0.937, 0.945, 66646, 10],
      [0.958, 0.96, 45581, 16],
      [0.971, 0.971, 28462, 32],
      [0.983, 0.983, 18296, 64],
      [0.991, 0.991, 11739, 128],
      [0.995, 0.995, 6263, 256],
    ],
  },
  {
    name: 'IVF1024,PQ16,RFlat',
    knob: 'nprobe',
    rows: [
      [0.62, 0.681, 50247, 1],
      [0.85, 0.891, 43331, 2],
      [0.976, 0.99, 39187, 4],
      [0.998, 1.0, 33029, 8],
      [0.999, 1.0, 24691, 16],
      [0.999, 1.0, 16750, 32],
    ],
  },
  {
    name: 'IVF1024,PQ16',
    knob: 'nprobe',
    rows: [
      [0.555, 0.479, 58848, 1],
      [0.698, 0.581, 50648, 2],
      [0.759, 0.629, 44811, 4],
      [0.767, 0.631, 37025, 8],
      [0.767, 0.631, 26862, 16],
      [0.767, 0.631, 18593, 32],
      [0.767, 0.631, 10959, 64],
    ],
  },
]
const FLAT_QPS = 1828
const METRICS = [
  { value: 'r10', label: '10-recall@10' },
  { value: 'r1', label: '1-recall@1' },
] as const
type Metric = (typeof METRICS)[number]['value']
const X_RANGE: [number, number] = [0.45, 1.005]
const Y_RANGE: [number | undefined, number | undefined] = [1000, 200000]

/** Recall against throughput for four FAISS indexes, with a draggable recall target. */
export function RecallQps() {
  const [metric, setMetric] = useState<Metric>('r10')
  const target = useParam(0.95, { min: 0.5, max: 1, step: 0.005 })
  const col = metric === 'r10' ? 0 : 1

  const series = useMemo(
    (): XYSeries[] => [
      ...RUNS.map((r, i): XYSeries => ({
        name: r.name,
        type: 'line',
        x: r.rows.map((row) => row[col]),
        y: r.rows.map((row) => row[2]),
        slot: i,
      })),
      { name: 'exact (Flat)', type: 'scatter', x: [1], y: [FLAT_QPS], emphasis: true },
    ],
    [col],
  )
  const best = RUNS.map((r) => {
    const ok = r.rows.filter((row) => row[col] >= target.value)
    if (!ok.length) return { name: r.name, text: 'never reaches it' }
    const top = ok.reduce((a, b) => (b[2] > a[2] ? b : a))
    return { name: r.name, text: `${formatNumber(top[2])} QPS (${r.knob} ${top[3]})` }
  })
  const handles: Handle[] = [{ kind: 'x', at: target.value, label: 'recall target', onDrag: (x) => target.set(x) }]

  return (
    <Interactive
      title="Recall against queries per second"
      caption="Each curve sweeps one search-time parameter of one index; up and to the right is better, and the upper-right envelope is the Pareto frontier. Measured with FAISS on 100 000 synthetic vectors of dimension 64, one thread. Drag the recall target to read off the fastest setting of each index that reaches it. Plain IVF-PQ saturates below 0.77: its 16-byte codes cannot resolve the nearest neighbours however many lists it scans, and re-ranking with the exact vectors (RFlat) removes that ceiling."
      controls={
        <>
          <ParamChoice label="recall measure" value={metric} onChange={setMetric} options={[...METRICS]} />
          <ParamSlider label="recall target" param={target} />
        </>
      }
      readout={
        <>
          {best.map((b) => (
            <Readout key={b.name} label={b.name} value={b.text} />
          ))}
          <Readout label="exact scan" value={`${FLAT_QPS} QPS`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel={metric === 'r10' ? '10-recall@10' : '1-recall@1'}
        yLabel="queries per second"
        xRange={X_RANGE}
        yRange={Y_RANGE}
        yLog
        handles={handles}
        height={380}
      />
    </Interactive>
  )
}
