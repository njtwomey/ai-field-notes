import { edmondsKarpSteps, type EdmondsKarpState } from 'aifn/graph/flows'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { formatValue, GraphView } from '@lab/views'
import { Readout } from '@lab/viz'
import { Framed, Sequence } from '../_shared/common'
import { lettered, placed, stateAt } from '../_shared/data'

// ── Edmonds–Karp ────────────────────────────────────────────────────────────────────────────────────────────────────

/** CLRS Figure 26.1: s, v₁ … v₄, t with capacities; the maximum flow is 23. */
const NET = lettered(
  'sabcdt',
  [
    ['sa', 16],
    ['sb', 13],
    ['ba', 4],
    ['ac', 12],
    ['cb', 9],
    ['bd', 14],
    ['dc', 7],
    ['ct', 20],
    ['dt', 4],
  ],
  true,
)
const NET_LABELS = ['s', 'v_1', 'v_2', 'v_3', 'v_4', 't'].map((x) => `$${x}$`)
const NET_AT = placed('sabcdt', { s: [0, 1.5], a: [2.5, 0], b: [2.5, 3], c: [5, 0], d: [5, 3], t: [7.5, 1.5] })

function describeFlow(s: EdmondsKarpState): string {
  if (s.t === 0 && !s.done) return 'no flow yet'
  if (s.done) return 'no augmenting path: the flow is maximum'
  const path = Array.from(s.pathNodes.data, (v) => NET_LABELS[v].replace(/\$/g, '')).join(' → ')
  return `augment ${formatValue(s.bottleneck)} along ${path}`
}

export function MaxFlowSpecimen() {
  const t = useMemo(() => trace(edmondsKarpSteps(NET, { source: 0, sink: 5 }), undefined, 100), [])
  const [step, setStep] = useState(0)
  const s = stateAt(t, step)
  const onPath = new Map(Array.from(s.path.data, (m) => [m >= 0 ? m : -m - 1, m >= 0]))
  const side = s.reached.data
  const cut = (k: number) => s.done && side[NET.edges[k].from] === 1 && side[NET.edges[k].to] === 0
  let cutCapacity = 0
  NET.edges.forEach((e, k) => cut(k) && (cutCapacity += e.weight!))

  return (
    <Figure
      title="Edmonds–Karp: augmenting paths and the minimum cut"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="augmentation" value={step} onChange={setStep} count={t.steps.length} defaultSpeed={1} />
        </div>
      }
      readouts={
        <>
          <Readout label="step" value={describeFlow(s)} />
          <Readout label="flow value" value={formatValue(s.value)} />
          {s.done && <Readout label="cut capacity" value={formatValue(cutCapacity)} />}
        </>
      }
      caption="CLRS Figure 26.1; edges are labelled flow/capacity. Each step finds a shortest augmenting path by breadth-first search in the residual network (active edges; a path may cancel flow on an edge by using it backwards) and pushes its bottleneck. When the sink is out of reach, the nodes the last search reached (slot 0) and the rest (slot 1) form a minimum cut: the highlighted saturated edges, whose capacities sum to the flow value."
    >
      <Framed
        footer={
          <Sequence label="path" ends="source first" items={Array.from(s.pathNodes.data, (v) => NET.labels![v])} />
        }
      >
        <GraphView
          graph={NET}
          layout={NET_AT}
          height="fill"
          ariaLabel="Flow network (CLRS Figure 26.1)"
          nodeLabels={NET_LABELS}
          nodeState={(v) => (s.done ? 'done' : side[v] ? 'active' : 'idle')}
          nodeTone={(v) => (s.done ? (side[v] ? 0 : 1) : undefined)}
          edgeLabels={(k) => `${formatValue(s.flow.data[k])}/${NET.edges[k].weight}`}
          edgeState={(k) => (onPath.has(k) ? 'active' : s.flow.data[k] > 0 ? 'done' : 'idle')}
          edgeHighlight={cut}
          edgeDashed={(k) => onPath.get(k) === false}
        />
      </Framed>
    </Figure>
  )
}
