import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type HeatmapOverlay,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

const N = 300
const T = Array.from({ length: N }, (_, n) => n)
// Two sources of different character: a smooth tone and a buzzy sawtooth.
const SOURCES = [T.map((n) => Math.sin((2 * Math.PI * 4 * n) / N)), T.map((n) => 2 * (((9 * n) / N) % 1) - 1)]
const NOISE = (() => {
  const g = rng(5)
  return [T.map(() => g.normal()), T.map(() => g.normal())]
})()

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)
function siSdr(ref: number[], est: number[]) {
  const alpha = dot(est, ref) / dot(ref, ref)
  const err = est.map((v, i) => v - alpha * ref[i])
  return 10 * Math.log10((alpha * alpha * dot(ref, ref)) / Math.max(dot(err, err), 1e-300))
}

/**
 * A separator's outputs are unordered: it may return the tone on channel 2 and the sawtooth on channel 1. Scoring in
 * a fixed order then measures the labelling, not the separation; PIT scores the best assignment of outputs to sources.
 */
export function OutputOrder() {
  const leak = useParam(0.15, { min: 0, max: 0.5, step: 0.01 })
  const noise = useParam(0.05, { min: 0, max: 0.5, step: 0.01 })
  const [swapped, setSwapped] = useState(true)

  const r = useMemo(() => {
    const order = swapped ? [1, 0] : [0, 1]
    // Output k contains mostly source order[k], some of the other source, and noise.
    const outputs = order.map((j, k) =>
      T.map((n) => (1 - leak.value) * SOURCES[j][n] + leak.value * SOURCES[1 - j][n] + noise.value * NOISE[k][n]),
    )
    // m[k][j]: SI-SDR of output k against source j.
    const m = outputs.map((o) => SOURCES.map((s) => siSdr(s, o)))
    const fixed = (m[0][0] + m[1][1]) / 2
    const other = (m[0][1] + m[1][0]) / 2
    const best = other > fixed ? [1, 0] : [0, 1]
    return { outputs, m, fixed, pit: Math.max(fixed, other), best }
  }, [leak.value, noise.value, swapped])

  const overlay: HeatmapOverlay[] = [
    { name: 'assignment chosen by PIT', type: 'scatter', x: r.best, y: [0, 1], emphasis: true },
  ]
  const waves: XYSeries[] = [
    { name: 'output 1', type: 'line', x: T, y: r.outputs[0], slot: 0 },
    { name: 'output 2', type: 'line', x: T, y: r.outputs[1], slot: 1 },
  ]

  return (
    <Interactive
      title="Which output is which source?"
      caption="A separator returns two outputs, each mostly one source with some leakage from the other and some noise. The heatmap is the SI-SDR of every output against every source. Scoring output 1 against source 1 and output 2 against source 2 gives a large negative number when the separator happened to return the sources in the other order, although the separation is good. PIT takes the better of the two assignments. Turn the swap off to see the fixed-order and PIT scores agree."
      controls={
        <>
          <ParamSwitch label="outputs in swapped order" checked={swapped} onChange={setSwapped} />
          <ParamSlider label="leakage from the other source" param={leak} />
          <ParamSlider label="noise level" param={noise} />
        </>
      }
      readout={
        <>
          <Readout label="fixed-order mean SI-SDR (dB)" value={formatNumber(r.fixed)} />
          <Readout label="PIT mean SI-SDR (dB)" value={formatNumber(r.pit)} />
          <Readout label="assignment" value={r.best[0] === 0 ? 'output 1 → source 1' : 'output 1 → source 2'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={waves} xLabel="sample" yLabel="amplitude" height={260} />
        <Heatmap
          x={[0, 1]}
          y={[0, 1]}
          z={r.m}
          range={[-30, 30]}
          scale="diverging"
          xLabel="source (0 = tone, 1 = sawtooth)"
          yLabel="output (0, 1)"
          valueLabel="SI-SDR (dB)"
          overlay={overlay}
          height={260}
        />
      </div>
    </Interactive>
  )
}
