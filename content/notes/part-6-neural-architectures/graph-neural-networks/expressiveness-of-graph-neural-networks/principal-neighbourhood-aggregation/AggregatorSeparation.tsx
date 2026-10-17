import { tensor, toRows } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import {
  Bars,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { pnaAggregate, type PnaAggregator } from 'aifn-compute/nn/graph'

const MAX = 5
const AGGREGATORS: PnaAggregator[] = ['mean', 'max', 'min', 'std', 'sum']
const value = (initial: number) => slider(-3, 3, initial, { step: 0.1, onChart: true })

/** The aggregates of one node's neighbours: the centre of a star whose leaves carry the multiset. */
function aggregates(values: number[], scaled: boolean, delta: number): number[] {
  const g = fromEdges(
    values.length + 1,
    values.map((_, i) => [0, i + 1] as [number, number]),
    { directed: false },
  )
  const h = tensor([[0], ...values.map((v) => [v])])
  const out = pnaAggregate(g, h, {
    aggregators: AGGREGATORS,
    scalers: [scaled ? 'amplification' : 'identity'],
    delta,
    epsilon: 0,
  })
  return toRows(out as never)[0]
}

/** Two neighbourhoods of scalar features and the aggregators that tell them apart. */
export function AggregatorSeparation() {
  const state = useFigureState({
    na: int(2, { ge: 1, le: MAX, label: 'neighbours of node a' }),
    nb: int(2, { ge: 1, le: MAX, label: 'neighbours of node b' }),
    scaled: setting(false, 'multiply by the amplification scaler log(d + 1) / δ'),
    a0: value(-1),
    a1: value(1),
    a2: value(0.5),
    a3: value(-0.5),
    a4: value(2),
    b0: value(-2),
    b1: value(2),
    b2: value(0.5),
    b3: value(-0.5),
    b4: value(2),
  })
  const as = [state.a0, state.a1, state.a2, state.a3, state.a4].slice(0, state.na)
  const bs = [state.b0, state.b1, state.b2, state.b3, state.b4].slice(0, state.nb)
  // δ: the mean of log(d + 1) over the two receiving nodes, standing in for the training-set average.
  const delta = (Math.log(state.na + 1) + Math.log(state.nb + 1)) / 2
  // Two stars of at most five leaves: cheap enough to recompute on every drag.
  const r = { a: aggregates(as, state.scaled, delta), b: aggregates(bs, state.scaled, delta) }
  const separates = AGGREGATORS.filter((_, k) => Math.abs(r.a[k] - r.b[k]) > 1e-9)

  const vx = useAxis({ label: 'neighbour feature', range: [-3.2, 3.2] })
  const vy = useAxis({ range: [-0.6, 1.6], zoom: false })
  const cx = useAxis({ categories: AGGREGATORS })
  const cy = useAxis({ label: state.scaled ? 'scaled aggregate' : 'aggregate' })
  const handles = (prefix: 'a' | 'b', values: number[], y: number) =>
    values.map((v, i) => (
      <Handle
        key={`${prefix}${i}`}
        kind="point"
        at={[v, y]}
        onDrag={(p) => state.set(`${prefix}${i}`, p[0])}
        label={`${prefix}${i + 1}`}
      />
    ))
  return (
    <Figure
      title="Which aggregators separate two neighbourhoods"
      state={state}
      readouts={
        <>
          <Readout label="node a: neighbours" value={as.map((v) => formatNumber(v)).join(', ')} />
          <Readout label="node b: neighbours" value={bs.map((v) => formatNumber(v)).join(', ')} />
          <Readout
            label="aggregators that separate them"
            value={separates.length > 0 ? separates.join(', ') : 'none'}
          />
          <Readout label="δ" value={formatNumber(delta)} />
        </>
      }
      caption="Top: the neighbour features of two nodes a (upper row) and b (lower row). Drag the points to change the features; the neighbour counts are typed above. Bottom: the mean, maximum, minimum, standard deviation and sum of each multiset. The default pair, {−1, 1} against {−2, 2}, has the same mean and sum but different extremes and spread. Set both counts to one and two equal values to see every degree-blind aggregator fail while the sum, or the mean times the amplification scaler, still separates them."
    >
      <Plots rows={2} heights={[1, 2]}>
        <Plot x={vx} y={vy} legend={false}>
          <Segments
            name="rows"
            segments={[
              { from: [-3, 1], to: [3, 1] },
              { from: [-3, 0], to: [3, 0] },
            ]}
            muted
          />
          <Points name="node a" x={as} y={as.map(() => 1)} slot={0} size={12} />
          <Points name="node b" x={bs} y={bs.map(() => 0)} slot={1} size={12} />
          {handles('a', as, 1)}
          {handles('b', bs, 0)}
        </Plot>
        <Plot x={cx} y={cy}>
          <Bars name="node a" x={AGGREGATORS.map((_, k) => k - 0.18)} y={r.a} width={0.34} slot={0} />
          <Bars name="node b" x={AGGREGATORS.map((_, k) => k + 0.18)} y={r.b} width={0.34} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
