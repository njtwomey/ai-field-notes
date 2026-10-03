import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { studentTCdf } from '@/lib/math/special'

type Shape = 'normal' | 'skewed'

const BINS = 41

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const variance = (xs: number[]) => {
  const m = mean(xs)
  return xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1)
}

/**
 * Two samples of size n; group B is shifted by δ. The permutation null is the distribution of the difference in means
 * over random relabellings of the pooled data; the p-value counts relabellings at least as extreme as the observed one.
 */
export function PermutationNull() {
  const [n, setN] = useState(8)
  const [delta, setDelta] = useState(1)
  const [perms, setPerms] = useState(2000)
  const [shape, setShape] = useState<Shape>('skewed')
  const [seed, setSeed] = useState(3)

  const r = useMemo(() => {
    const g = rng(seed)
    const draw = () => (shape === 'normal' ? g.normal() : -Math.log(Math.max(g.uniform(), 1e-12)) - 1)
    const a = Array.from({ length: n }, draw)
    const b = Array.from({ length: n }, () => draw() + delta)
    const obs = mean(b) - mean(a)
    const pooled = [...a, ...b]
    const total = pooled.reduce((s, x) => s + x, 0)
    const diffs: number[] = []
    const idx = pooled.map((_, i) => i)
    for (let p = 0; p < perms; p++) {
      // Partial Fisher–Yates shuffle: the first n positions become group B.
      for (let i = 0; i < n; i++) {
        const j = i + Math.floor(g.uniform() * (2 * n - i))
        ;[idx[i], idx[j]] = [idx[j], idx[i]]
      }
      let sb = 0
      for (let i = 0; i < n; i++) sb += pooled[idx[i]]
      diffs.push(sb / n - (total - sb) / n)
    }
    const extreme = diffs.filter((d) => Math.abs(d) >= Math.abs(obs) - 1e-12).length
    const pPerm = (extreme + 1) / (perms + 1)
    const va = variance(a) / n
    const vb = variance(b) / n
    const t = obs / Math.sqrt(va + vb)
    const df = (va + vb) ** 2 / (va ** 2 / (n - 1) + vb ** 2 / (n - 1))
    const pWelch = 2 * (1 - studentTCdf(Math.abs(t), df))

    const span = Math.max(Math.abs(obs), ...diffs.map(Math.abs)) * 1.05
    const width = (2 * span) / BINS
    const mids = Array.from({ length: BINS }, (_, i) => -span + (i + 0.5) * width)
    const inner = new Array<number>(BINS).fill(0)
    const tail = new Array<number>(BINS).fill(0)
    for (const d of diffs) {
      const k = Math.min(BINS - 1, Math.max(0, Math.floor((d + span) / width)))
      if (Math.abs(d) >= Math.abs(obs) - 1e-12) tail[k]++
      else inner[k]++
    }
    const scale = 1 / (perms * width)
    const top = Math.max(...mids.map((_, i) => (inner[i] + tail[i]) * scale))
    const series: XYSeries[] = [
      { name: 'permutation null', type: 'bar', x: mids, y: inner.map((c) => c * scale), slot: 0 },
      { name: 'at least as extreme', type: 'bar', x: mids, y: tail.map((c) => c * scale), slot: 1 },
      { name: 'observed difference', type: 'line', x: [obs, obs], y: [0, top * 1.1], emphasis: true },
      { name: 'mirror of observed', type: 'line', x: [-obs, -obs], y: [0, top * 1.1], emphasis: true, dashed: true },
    ]
    return { obs, pPerm, pWelch, series, span, top }
  }, [n, delta, perms, shape, seed])

  return (
    <Interactive
      title="The permutation null distribution"
      caption="Two groups of size n, the second shifted by δ. Each permutation reassigns the pooled observations to the two groups at random and recomputes the difference in means. Under the null hypothesis every assignment is equally likely, so the histogram is the exact reference distribution up to Monte Carlo error. The p-value is the fraction of permutations at least as extreme as the observed difference (in either direction), counting the observed labelling itself. With skewed data and small n it can differ from Welch's t-test, which relies on approximate normality."
      controls={
        <>
          <ParamSlider label="shift δ" value={delta} onChange={setDelta} min={0} max={2} step={0.1} />
          <ParamSlider label="size of each group n" value={n} onChange={setN} min={3} max={40} step={1} />
          <ParamSlider label="permutations" value={perms} onChange={setPerms} min={200} max={10000} step={200} />
          <ParamChoice
            label="data"
            value={shape}
            onChange={setShape}
            options={[
              { value: 'normal', label: 'normal' },
              { value: 'skewed', label: 'skewed (exponential)' },
            ]}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New samples</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="observed difference" value={formatNumber(r.obs)} />
          <Readout label="permutation p" value={formatNumber(r.pPerm)} />
          <Readout label="Welch t-test p" value={formatNumber(r.pWelch)} />
        </>
      }
    >
      <XYChart
        height={300}
        series={r.series}
        xRange={[-r.span, r.span]}
        yRange={[0, r.top * 1.15]}
        xLabel="difference in means (B − A)"
        yLabel="density"
      />
    </Interactive>
  )
}
