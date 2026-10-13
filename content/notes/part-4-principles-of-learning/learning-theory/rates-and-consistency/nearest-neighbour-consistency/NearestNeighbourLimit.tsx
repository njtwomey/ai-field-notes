import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { type ClassificationTruth } from 'aifn-methods/data'
import { gaussians } from 'aifn-methods/data/synthetic'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const X_RANGE: [number, number] = [-4, 4]
const GRID = toFlat(linspace(X_RANGE[0], X_RANGE[1], 241))
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/**
 * The mechanism behind the Cover–Hart limit: as m grows the nearest training point to a query x comes ever closer, so
 * its posterior η(x′) approaches η(x), but its label stays one random draw. The 1-NN rule's conditional error at x
 * tends to 2η(x)(1 − η(x)), not to the Bayes error min(η(x), 1 − η(x)).
 */
export function NearestNeighbourLimit() {
  const state = useFigureState({
    m: int(100, { ge: 1, le: 20000, suggestions: [10, 100, 1000, 10000], label: 'training points m' }),
    separation: slider(0.5, 4, 1.5, { step: 0.1, label: 'class separation d′' }),
    query: slider(X_RANGE[0], X_RANGE[1], 0.4, { step: 0.01, label: 'query x' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })
  const { m, separation, query, seed } = state

  const data = useMemo(() => {
    const d = gaussians(stream(`nn-limit/${seed}`), { n: m, means: [[-1], [1]], separation })
    const truth = d.meta.truth as ClassificationTruth
    const xs = toFlat(d.x)
    const ys = toFlat(d.y!)
    const posterior = toFlat(truth.posterior(tensor(Array.from(GRID, (v) => [v]))))
    const eta = Array.from(GRID, (_, i) => posterior[2 * i + 1])
    const model = kNearestNeighbours({ k: 1 }).fit(dataset(d.x, d.y!))
    return { xs, ys, eta, model, truth }
  }, [m, separation, seed])

  const at = useMemo(() => {
    const index = toFlat(data.model.neighbours(tensor([[query]])).index)[0]
    const xNear = data.xs[index]
    const yNear = data.ys[index]
    const post = toFlat(data.truth.posterior(tensor([[query], [xNear]])))
    return { xNear, yNear, eta: post[1], etaNear: post[3] }
  }, [data, query])

  const byClass = (c: number) => {
    const x: number[] = []
    for (let i = 0; i < data.xs.length; i++) if (data.ys[i] === c) x.push(data.xs[i])
    return x
  }
  const zeros = byClass(0)
  const ones = byClass(1)
  const dense = m > 2000
  const { eta, etaNear } = at
  const bayes = Math.min(eta, 1 - eta)
  const conditional = eta * (1 - etaNear) + (1 - eta) * etaNear
  const limit = 2 * eta * (1 - eta)

  const xAxis = useAxis({ label: 'input x', range: X_RANGE })
  const yAxis = useAxis({ label: 'η(x) = P(y = 1 | x), and labels', range: [-0.08, 1.08] })
  return (
    <Figure
      title="Why the nearest neighbour's error does not reach the Bayes error"
      purpose="The nearest training point converges to the query, so its posterior converges to the query's; its label remains one random draw from that posterior."
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="distance to nearest neighbour" value={fmt(Math.abs(at.xNear - query))} />
          <Readout label="η(x), η(x′)" value={`${fmt(eta)}, ${fmt(etaNear)}`} />
          <Readout label="Bayes error at x" value={fmt(bayes)} />
          <Readout label="1-NN error at x, this neighbour" value={fmt(conditional)} />
          <Readout label="limit 2η(1 − η)" value={fmt(limit)} />
        </>
      }
      caption="Two Gaussian classes on the line with equal priors; the ink curve is the posterior η(x) = P(y = 1 | x), and the training points sit at height 0 or 1 by label. Drag the vertical line to move the query x. The large ink point is its nearest training point x′, and the small ink point on the curve is η(x′). The 1-NN rule predicts the label of x′, so its error at x, averaged over the labels of x and x′, is η(x)(1 − η(x′)) + (1 − η(x))η(x′). As m grows the distance |x − x′| shrinks, η(x′) approaches η(x), and this error approaches 2η(x)(1 − η(x)), which exceeds the Bayes error min(η(x), 1 − η(x)) everywhere except where η is 0, ½ or 1."
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve name="posterior η(x)" x={GRID} y={data.eta} emphasis />
        {zeros.length > 0 && (
          <Points name="label 0" slot={0} x={zeros} y={zeros.map(() => 0)} thin={!dense} dense={dense} />
        )}
        {ones.length > 0 && (
          <Points name="label 1" slot={1} x={ones} y={ones.map(() => 1)} thin={!dense} dense={dense} />
        )}
        <Points name="nearest neighbour x′" x={[at.xNear]} y={[at.yNear]} emphasis size={12} />
        <Points name="η at x′" x={[at.xNear]} y={[etaNear]} emphasis size={7} />
        <Handle
          kind="x"
          at={query}
          onDrag={(v) => state.set('query', Math.max(X_RANGE[0], Math.min(X_RANGE[1], v)))}
          label="x"
        />
      </Plot>
    </Figure>
  )
}
