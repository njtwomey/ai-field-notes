import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

const KEYS = 16
const MAX_LOG_D = 10

/** One query and 16 keys with independent standard normal components, up to d = 1024. */
function draw(seed: number) {
  const r = rng(seed)
  const d = 2 ** MAX_LOG_D
  const q = Array.from({ length: d }, () => r.normal())
  const k = Array.from({ length: KEYS }, () => Array.from({ length: d }, () => r.normal()))
  return { q, k }
}

const INDEX = Array.from({ length: KEYS }, (_, j) => j + 1)

/** Softmax attention weights of one query over 16 random keys, with and without the 1/√d scaling. */
export function ScalingDemo() {
  const logD = useParam(6, { min: 1, max: MAX_LOG_D, step: 1 })
  const [scaled, setScaled] = useState(true)
  const [seed, setSeed] = useState(1)
  const sample = useMemo(() => draw(seed), [seed])
  const d = 2 ** logD.value

  const r = useMemo(() => {
    const scale = scaled ? 1 / Math.sqrt(d) : 1
    const scores = sample.k.map((kj) => {
      let s = 0
      for (let i = 0; i < d; i++) s += sample.q[i] * kj[i]
      return s * scale
    })
    const m = Math.max(...scores)
    const exps = scores.map((s) => Math.exp(s - m))
    const z = exps.reduce((a, b) => a + b, 0)
    const weights = exps.map((e) => e / z)
    const mean = scores.reduce((a, b) => a + b, 0) / KEYS
    const sd = Math.sqrt(scores.reduce((a, s) => a + (s - mean) ** 2, 0) / KEYS)
    const entropy = -weights.reduce((a, w) => a + (w > 0 ? w * Math.log2(w) : 0), 0)
    return { weights, sd, entropy, max: Math.max(...weights) }
  }, [sample, d, scaled])

  const series: XYSeries[] = [{ name: 'attention weight', type: 'bar', x: INDEX, y: r.weights, slot: 0 }]

  return (
    <Interactive
      title="Why divide by √d"
      caption="One query attends over 16 keys whose components are independent standard normals. Without scaling, the scores have standard deviation about √d, so as d grows the softmax puts almost all its weight on one key and its gradient vanishes. Dividing by √d keeps the score spread near 1 for every d, and the weights stay spread out."
      controls={
        <>
          <ParamSlider label="dimension d" param={logD} format={(v) => String(2 ** v)} withArrows />
          <ParamSwitch label="divide scores by √d" checked={scaled} onChange={setScaled} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New query and keys</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="score std" value={formatNumber(r.sd)} />
          <Readout label="largest weight" value={formatNumber(r.max)} />
          <Readout label="entropy (bits, max 4)" value={formatNumber(r.entropy)} />
        </>
      }
    >
      <XYChart series={series} xLabel="key" yLabel="weight" yRange={[0, 1]} xRange={[0.5, KEYS + 0.5]} height={280} />
    </Interactive>
  )
}
