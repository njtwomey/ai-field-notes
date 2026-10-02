import { useState } from 'react'
import { Diagram } from 'aifn-render'
import { link, variable } from 'aifn-render'
import type { DiagramEdge, DiagramSpec } from 'aifn-render'
import { Interactive, ParamSwitch, Readout } from 'aifn-render'

type Structure = { name: string; edges: DiagramEdge[]; blockedWhenObserved: boolean }

const STRUCTURES: Structure[] = [
  {
    name: 'Chain  a → c → b',
    edges: [link('a', 'c'), link('c', 'b')],
    blockedWhenObserved: true,
  },
  {
    name: 'Fork  a ← c → b',
    edges: [link('c', 'a'), link('c', 'b')],
    blockedWhenObserved: true,
  },
  {
    name: 'Collider  a → c ← b',
    edges: [link('a', 'c'), link('b', 'c')],
    blockedWhenObserved: false,
  },
]

/** The three canonical structures, with c observed or not, and whether a path between a and b is open. */
export function CanonicalStructures() {
  const [observed, setObserved] = useState(false)
  const spec = (s: Structure, open: boolean): DiagramSpec => {
    // Chains read left to right; forks and colliders put c between a and b, above or below them.
    const collider = !s.blockedWhenObserved
    const chain = s.name.startsWith('Chain')
    const ends = chain ? 0 : collider ? 0 : 1.3
    const size = { w: 0.75, h: 0.75 }
    return {
      unit: 56,
      nodes: [
        variable('a', 0, ends, '$a$', { ...size, highlight: open }),
        variable('c', 1.5, chain ? 0 : collider ? 1.3 : 0, '$c$', { ...size, filled: observed }),
        variable('b', 3, ends, '$b$', { ...size, highlight: open }),
      ],
      edges: s.edges,
    }
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
            <div key={s.name} className="flex flex-col gap-1 text-center">
              <div className="flex flex-1 items-center">
                <Diagram spec={spec(s, open)} ariaLabel={s.name} />
              </div>
              <p className="text-xs text-muted-foreground">{s.name}</p>
            </div>
          )
        })}
      </div>
    </Interactive>
  )
}
