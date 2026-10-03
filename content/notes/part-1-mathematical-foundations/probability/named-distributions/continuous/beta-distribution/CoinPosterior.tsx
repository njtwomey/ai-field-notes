import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { distribution } from '@/lib/distributions'
import { linspace, rng } from '@/lib/math'
import { invertCdf } from '@/lib/math/special'

const beta = distribution('beta')
const MAX_FLIPS = 1000
// One fixed stream of uniforms: flip i lands heads when u[i] < p, so changing p re-reads the same coin tosses.
const UNIFORMS = (() => {
  const r = rng(7)
  return Array.from({ length: MAX_FLIPS }, () => r.uniform())
})()

/** A Beta prior on a coin's bias, updated flip by flip into a Beta posterior. */
export function CoinPosterior() {
  const [a, setA] = useState(2)
  const [b, setB] = useState(2)
  const [p, setP] = useState(0.7)
  const [n, setN] = useState(0)

  const heads = useMemo(() => UNIFORMS.slice(0, n).filter((u) => u < p).length, [n, p])
  const tails = n - heads
  const post = { a: a + heads, b: b + tails }

  const series = useMemo((): XYSeries[] => {
    const xs = linspace(0.002, 0.998, 400)
    const prior = xs.map((x) => beta.density(x, { a, b }))
    const posterior = xs.map((x) => beta.density(x, { a: a + heads, b: b + tails }))
    const top = Math.max(...posterior, ...prior.filter(Number.isFinite))
    return [
      { name: `prior Beta(${a}, ${b})`, type: 'line', x: xs, y: prior, slot: 0, dashed: true },
      { name: 'posterior', type: 'line', x: xs, y: posterior, slot: 1, area: true },
      { name: 'true bias p', type: 'line', x: [p, p], y: [0, top], slot: 2, dashed: true },
    ]
  }, [a, b, heads, tails, p])

  const [lo, hi] = [0.025, 0.975].map((u) => invertCdf((x) => beta.cdf(x, post), u, 0, 1))

  return (
    <Interactive
      title="Learning a coin's bias"
      caption="The prior (dashed) is Beta(α, β). Each flip adds 1 to α for heads or to β for tails, and the posterior (shaded) narrows around the true bias. A strong prior, with large α and β, needs more flips to move."
      controls={
        <>
          <ParamSlider label="prior α" value={a} onChange={setA} min={0.5} max={30} step={0.5} />
          <ParamSlider label="prior β" value={b} onChange={setB} min={0.5} max={30} step={0.5} />
          <ParamSlider label="true bias p" value={p} onChange={setP} min={0.05} max={0.95} step={0.05} />
          <ParamButton onClick={() => setN((v) => Math.min(v + 1, MAX_FLIPS))}>Flip 1</ParamButton>
          <ParamButton onClick={() => setN((v) => Math.min(v + 10, MAX_FLIPS))}>Flip 10</ParamButton>
          <ParamButton onClick={() => setN((v) => Math.min(v + 100, MAX_FLIPS))}>Flip 100</ParamButton>
          <ParamButton onClick={() => setN(0)} disabled={n === 0}>
            Reset
          </ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="flips" value={`${n} (${heads} heads, ${tails} tails)`} />
          <Readout label="posterior" value={`Beta(${formatNumber(post.a)}, ${formatNumber(post.b)})`} />
          <Readout label="posterior mean" value={formatNumber(post.a / (post.a + post.b))} />
          <Readout label="95% credible interval" value={`${formatNumber(lo)} to ${formatNumber(hi)}`} />
        </>
      }
    >
      <XYChart height={300} series={series} xLabel="bias p" yLabel="density" xRange={[0, 1]} yRange={[0, undefined]} />
    </Interactive>
  )
}
