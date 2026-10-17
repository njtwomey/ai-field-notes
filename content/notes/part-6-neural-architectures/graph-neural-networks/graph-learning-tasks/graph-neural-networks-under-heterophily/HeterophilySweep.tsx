import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { propagate } from 'aifn-compute/graph/propagation'
import { FixedHeight } from '../_shared/figure-helpers'
import { edgeHomophily } from 'aifn-compute/graph/measures'
import {
  drawBlockModel,
  homophilySweep,
  idealAccuracy,
  labelledSplit,
  viewAccuracies,
  VIEW_NAMES,
  VIEWS,
} from '../_shared/linear-node-models'

const PANEL = 280
const BLOCK = 100
const GRID = Array.from({ length: 11 }, (_, i) => i / 10)
const FINE = Array.from({ length: 101 }, (_, i) => i / 100)

/** Three linear node classifiers on a two-block contextual stochastic block model, against edge homophily. */
export function HeterophilySweep() {
  const state = useFigureState({
    homophily: slider(0, 1, 0.2, { step: 0.01, label: 'edge homophily h' }),
    degree: int(6, { ge: 1, le: 30, suggestions: [2, 6, 15], label: 'expected degree d' }),
    signal: float(1, { ge: 0, le: 5, suggestions: [0.5, 1, 2], label: 'feature signal' }),
    perClass: int(20, { ge: 2, le: 80, suggestions: [5, 20, 50], label: 'labelled nodes per class' }),
  })
  const { homophily, degree, signal, perClass } = state

  const sweep = useComputed(
    () => homophilySweep({ homophilies: GRID, setting: { blockSize: BLOCK, degree, signal }, perClass, seeds: 3 }),
    [degree, signal, perClass],
    { mode: 'release' },
  )

  const current = useMemo(() => {
    const s = stream('heterophily/figure')
    const d = drawBlockModel(child(s, 'graph'), { blockSize: BLOCK, degree, signal, homophily })
    const y = toFlat(d.y)
    const split = labelledSplit(child(s, 'split'), y, perClass)
    // Along the class-difference direction: own value u_v and the mean of the neighbours' values.
    const xs = toRows(d.x) as number[][]
    const u = xs.map((r) => (r[0] - r[1]) / Math.SQRT2)
    const m = (toRows(propagate(d.graph, d.x, { aggregate: 'mean' }) as never) as number[][]).map(
      (r) => (r[0] - r[1]) / Math.SQRT2,
    )
    return { y, u, m, accuracy: viewAccuracies(d, split), measured: edgeHomophily(d.graph, y) }
  }, [homophily, degree, signal, perClass])

  const ux = useAxis({ label: 'own feature u_v', range: [-4, 4], nice: false })
  const my = useAxis({ label: 'neighbour mean m_v', range: [-3, 3], nice: false })
  const hx = useAxis({ label: 'edge homophily h', range: [0, 1], nice: false })
  const ay = useAxis({ label: 'test accuracy', range: [0.4, 1], nice: false })
  // The GCN's mean mixes u and m as (u + d m) / (d + 1): its decision boundary is the line u + d m = 0.
  const boundary = { x: [-4, 4], y: [4 / degree, -4 / degree] }
  const percent = (x: number) => `${Math.round(100 * x)}%`

  return (
    <Figure
      title="Where averaging the neighbours stops helping"
      state={state}
      defaultSize="L"
      caption={`Two classes of ${BLOCK} nodes in a contextual stochastic block model: each node expects d neighbours, a share h of them in its own class, and carries two features whose class means differ by the feature signal. Three logistic regressions are fitted on the labelled nodes: on the node's own features (no edges), on the mean over the node and its neighbours (a linear GCN layer), and on the node's own features next to its neighbours' mean (H2GCN's separation of ego and neighbours). Left: every node at the chosen h, along the class-difference direction, its own value u against its neighbours' mean m, coloured by class; the dashed line u + d·m = 0 is the only boundary the GCN mean can draw. Right: test accuracy against h, averaged over three graphs (solid), and the accuracy the formula of the note predicts when every node has degree exactly d (dashed). The GCN mean is worst near h = (d − 1)/(2d), where a node's own value and its neighbours' cancel; the separated model never falls below the features-only model. Drag the vertical line or use the slider to set h.`}
      readouts={
        <>
          <Readout label="measured edge homophily" value={formatNumber(current.measured)} />
          {VIEWS.map((v) => (
            <Readout key={v} label={VIEW_NAMES[v]} value={percent(current.accuracy[v])} />
          ))}
          <Readout label="GCN worst at h" value={formatNumber((degree - 1) / (2 * degree))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={ux} y={my} title={`nodes at h = ${formatNumber(homophily)}`}>
            <Points
              name="nodes"
              x={current.u}
              y={current.m}
              group={current.y}
              groupNames={['class 0', 'class 1']}
              thin
            />
            <Curve name="GCN boundary u + d·m = 0" x={boundary.x} y={boundary.y} dashed emphasis />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={hx} y={ay} title="accuracy against homophily">
            {VIEWS.map((v, k) => (
              <Curve key={v} name={VIEW_NAMES[v]} slot={k} x={GRID} y={sweep.value[v]} stale={sweep.stale} showPoints />
            ))}
            {VIEWS.map((v, k) => (
              <Curve
                key={`ideal-${v}`}
                name={`${VIEW_NAMES[v]}, formula`}
                slot={k}
                x={FINE}
                y={FINE.map((h) => idealAccuracy(v, { degree, homophily: h, signal }))}
                dashed
                thin
              />
            ))}
            <Handle {...state.handle('homophily', { label: 'h' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
