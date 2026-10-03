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
} from 'aifn-render'

type Preset = 'one' | 'two'

// Five items; item 1 is relevant and its score is the one the reader moves. The other scores stay fixed.
const OTHERS = [1.2, 0.5, -0.2, -0.9]
const LABELS: Record<Preset, number[]> = { one: [1, 0, 0, 0, 0], two: [1, 0, 1, 0, 0] }
const PRESET_LABEL: Record<Preset, string> = { one: 'one relevant item', two: 'two relevant items' }
const X_MIN = -3
const X_MAX = 3
const GRID = Array.from({ length: 301 }, (_, k) => X_MIN + (k * (X_MAX - X_MIN)) / 300)
const X_RANGE: [number, number] = [X_MIN, X_MAX]
const Y_RANGE: [number | undefined, number | undefined] = [0, 1.02]

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

/** Metrics and smooth surrogates of one binary-labelled list, as functions of the scores. */
function evaluate(s: number[], y: number[], temperature: number) {
  const n = s.length
  // Rank 1 is the top; ties count against the item, so the metrics never overstate the ranking.
  const rank = s.map((si, i) => 1 + s.reduce((a, sj, j) => a + (j !== i && sj >= si ? 1 : 0), 0))
  const relevant = y.flatMap((v, i) => (v > 0 ? [i] : []))
  const idcg = relevant.reduce((a, _, k) => a + 1 / Math.log2(k + 2), 0)
  const ndcg = relevant.reduce((a, i) => a + 1 / Math.log2(1 + rank[i]), 0) / idcg
  const rr = Math.max(...relevant.map((i) => 1 / rank[i]))
  const m = Math.max(...s)
  const logZ = m + Math.log(s.reduce((a, v) => a + Math.exp(v - m), 0))
  // Softmax cross-entropy with the target spread evenly over the relevant items.
  const loss = relevant.reduce((a, i) => a + (logZ - s[i]), 0) / relevant.length
  const approxRank = s.map((si, i) => s.reduce((a, sj, j) => a + (j !== i ? sigmoid((sj - si) / temperature) : 0), 1))
  const approxNdcg = relevant.reduce((a, i) => a + 1 / Math.log2(1 + approxRank[i]), 0) / idcg
  return { rank: rank[0], ndcg, rr, bound: Math.exp(-loss), loss, approxNdcg, n }
}

/**
 * NDCG and reciprocal rank step as one relevant item's score passes the others, while the softmax cross-entropy bound
 * exp(−L) and ApproxNDCG change smoothly. The bound stays below both metrics everywhere.
 */
export function MetricSteps() {
  const [preset, setPreset] = useState<Preset>('one')
  const score = useParam(-0.5, { min: X_MIN, max: X_MAX, step: 0.01 })
  const [temperature, setTemperature] = useState(0.5)
  const y = LABELS[preset]

  const series = useMemo((): XYSeries[] => {
    const rows = GRID.map((x) => evaluate([x, ...OTHERS], y, temperature))
    return [
      { name: 'NDCG', type: 'line', x: GRID, y: rows.map((r) => r.ndcg), slot: 0 },
      { name: 'reciprocal rank', type: 'line', x: GRID, y: rows.map((r) => r.rr), slot: 1 },
      { name: 'ApproxNDCG', type: 'line', x: GRID, y: rows.map((r) => r.approxNdcg), slot: 2 },
      {
        name: 'exp(−softmax CE), a lower bound',
        type: 'line',
        x: GRID,
        y: rows.map((r) => r.bound),
        slot: 3,
        dashed: true,
      },
      { name: 'other items’ scores', type: 'scatter', x: OTHERS, y: OTHERS.map(() => 0.02), muted: true },
    ]
  }, [y, temperature])

  const now = useMemo(() => evaluate([score.value, ...OTHERS], y, temperature), [score.value, y, temperature])

  const handles: Handle[] = [{ kind: 'x', at: score.value, label: 'score of item 1', onDrag: score.set }]

  return (
    <Interactive
      title="Metrics step, surrogates slide"
      caption="Five items; item 1 is relevant and the grey marks show the fixed scores of the other four. Drag the vertical line (or use the slider) to move item 1's score. NDCG and reciprocal rank change only when item 1 passes another item, so their gradient is zero between the jumps. The softmax cross-entropy bound exp(−L) rises smoothly and stays below both metrics. ApproxNDCG replaces each rank indicator by a sigmoid: a small temperature tracks NDCG closely but flattens between the jumps, a large one is smooth but biased."
      controls={
        <>
          <ParamSlider label="score of item 1" param={score} />
          <ParamSlider
            label="ApproxNDCG temperature T"
            value={temperature}
            onChange={setTemperature}
            min={0.05}
            max={2}
            step={0.05}
          />
          <ParamChoice
            label="labels"
            value={preset}
            onChange={setPreset}
            options={(Object.keys(LABELS) as Preset[]).map((p) => ({ value: p, label: PRESET_LABEL[p] }))}
          />
        </>
      }
      readout={
        <>
          <Readout label="rank of item 1" value={`${now.rank} of ${now.n}`} />
          <Readout label="NDCG" value={formatNumber(now.ndcg)} />
          <Readout label="reciprocal rank" value={formatNumber(now.rr)} />
          <Readout label="softmax CE L" value={formatNumber(now.loss)} />
          <Readout label="exp(−L)" value={formatNumber(now.bound)} />
          <Readout label="ApproxNDCG" value={formatNumber(now.approxNdcg)} />
        </>
      }
    >
      <XYChart
        height={320}
        series={series}
        handles={handles}
        xRange={X_RANGE}
        yRange={Y_RANGE}
        xLabel="score of item 1"
        yLabel="value"
      />
    </Interactive>
  )
}
