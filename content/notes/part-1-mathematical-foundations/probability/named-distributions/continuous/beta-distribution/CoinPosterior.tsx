import { useMemo, useState } from 'react'
import {
  Area,
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { Beta } from 'aifn-compute/probability/distributions'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

const MAX_FLIPS = 1000
// One fixed stream of uniforms: flip i lands heads when u[i] < p, so changing p re-reads the same coin tosses.
const UNIFORMS = (() => {
  const r = stream(7)
  return Array.from({ length: MAX_FLIPS }, () => uniform(r))
})()

/** A Beta prior on a coin's bias, updated flip by flip into a Beta posterior. */
export function CoinPosterior() {
  const state = useFigureState({
    a: float(2, { min: 0.5, max: 30, step: 0.5, label: 'prior α' }),
    b: float(2, { min: 0.5, max: 30, step: 0.5, label: 'prior β' }),
    p: slider(0.05, 0.95, 0.7, { step: 0.05, label: 'true bias p' }),
  })
  const [n, setN] = useState(0)

  const heads = useMemo(() => UNIFORMS.slice(0, n).filter((u) => u < state.p).length, [n, state.p])
  const tails = n - heads
  const post = { a: state.a + heads, b: state.b + tails }

  const series = useMemo(() => {
    const xs = toFlat(linspace(0.002, 0.998, 400))
    const priorLaw = Beta(state.a, state.b)
    const postLaw = Beta(state.a + heads, state.b + tails)
    const prior = xs.map((x) => priorLaw.prob(x))
    const posterior = xs.map((x) => postLaw.prob(x))
    return [
      { name: `prior Beta(${state.a}, ${state.b})`, x: xs, y: prior, slot: 0, dashed: true },
      { name: 'posterior', x: xs, y: posterior, slot: 1 },
    ] as const
  }, [state.a, state.b, heads, tails])

  const postLaw = Beta(post.a, post.b)
  const [lo, hi] = [0.025, 0.975].map((u) => postLaw.quantile(u))

  const xAxis = useAxis({ label: 'bias p', range: [0, 1] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Learning a coin's bias"
      state={state}
      caption="The prior (dashed) is Beta(α, β). Each flip adds 1 to α for heads or to β for tails, and the posterior (shaded) narrows around the true bias. A strong prior, with large α and β, needs more flips to move. Drag the vertical line at the true bias p to change it."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => setN((v) => Math.min(v + 1, MAX_FLIPS))}>
            Flip 1
          </Button>
          <Button variant="outline" size="sm" onClick={() => setN((v) => Math.min(v + 10, MAX_FLIPS))}>
            Flip 10
          </Button>
          <Button variant="outline" size="sm" onClick={() => setN((v) => Math.min(v + 100, MAX_FLIPS))}>
            Flip 100
          </Button>
          <Button variant="outline" size="sm" onClick={() => setN(0)} disabled={n === 0}>
            Reset
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="flips" value={`${n} (${heads} heads, ${tails} tails)`} />
          <Readout label="posterior" value={`Beta(${formatNumber(post.a)}, ${formatNumber(post.b)})`} />
          <Readout label="posterior mean" value={formatNumber(post.a / (post.a + post.b))} />
          <Readout label="95% credible interval" value={`${formatNumber(lo)} to ${formatNumber(hi)}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Area {...series[1]} />
        <Handle {...state.handle('p', { label: 'true bias p' })} />
      </Plot>
    </Figure>
  )
}
