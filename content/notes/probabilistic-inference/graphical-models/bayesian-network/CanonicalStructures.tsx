import { useState } from 'react'
import { GraphDiagram, Interactive, ParamSwitch, Readout, type GraphEdge, type GraphNode } from '@/components/viz'

type Structure = { name: string; edges: GraphEdge[]; blockedWhenObserved: boolean }

const STRUCTURES: Structure[] = [
  {
    name: 'Chain  a → c → b',
    edges: [
      { source: 'a', target: 'c' },
      { source: 'c', target: 'b' },
    ],
    blockedWhenObserved: true,
  },
  {
    name: 'Fork  a ← c → b',
    edges: [
      { source: 'c', target: 'a' },
      { source: 'c', target: 'b' },
    ],
    blockedWhenObserved: true,
  },
  {
    name: 'Collider  a → c ← b',
    edges: [
      { source: 'a', target: 'c' },
      { source: 'b', target: 'c' },
    ],
    blockedWhenObserved: false,
  },
]

/** The three canonical structures, with c observed or not, and whether a path between a and b is open. */
export function CanonicalStructures() {
  const [observed, setObserved] = useState(false)
  const nodes = (s: Structure): GraphNode[] => {
    // Chains read left to right; forks and colliders put c between a and b, above or below them.
    const collider = !s.blockedWhenObserved
    const chain = s.name.startsWith('Chain')
    return [
      { id: 'a', x: 0, y: chain ? 0 : collider ? 0 : 1 },
      { id: 'c', x: 1, y: chain ? 0 : collider ? 1 : 0, kind: observed ? 'observed' : 'variable' },
      { id: 'b', x: 2, y: chain ? 0 : collider ? 0 : 1 },
    ]
  }
  return (
    <Interactive
      title="Chain, fork and collider"
      caption="Each structure links a and b through c. A shaded node is observed. Observing c blocks the chain and the fork, so a and b become independent given c; observing c opens the collider, so a and b, independent a priori, become dependent given c."
      controls={<ParamSwitch label="observe c" checked={observed} onChange={setObserved} />}
      readout={STRUCTURES.map((s) => {
        const open = observed ? !s.blockedWhenObserved : s.blockedWhenObserved
        return <Readout key={s.name} label={s.name.split(' ')[0]} value={open ? 'a and b dependent' : 'a ⫫ b'} />
      })}
    >
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {STRUCTURES.map((s) => {
          const open = observed ? !s.blockedWhenObserved : s.blockedWhenObserved
          return (
            <div key={s.name} className="text-center">
              <GraphDiagram
                nodes={nodes(s)}
                edges={s.edges}
                highlight={open ? ['a', 'b'] : []}
                height={150}
                ariaLabel={s.name}
              />
              <p className="text-xs text-muted-foreground">{s.name}</p>
            </div>
          )
        })}
      </div>
    </Interactive>
  )
}
