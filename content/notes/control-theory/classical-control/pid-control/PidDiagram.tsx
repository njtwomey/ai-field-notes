import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const term = (id: string, y: number, label: string) => ({ id, x: 4.8, y, w: 1.7, h: 0.8, label, tone: 0 })

const spec: DiagramSpec = {
  nodes: [
    { id: 'r', x: 0, y: 2, shape: 'text', w: 0.6, label: '$r$' },
    { id: 's1', x: 1.3, y: 2, shape: 'op', label: '$\\Sigma$' },
    { id: 's1m', x: 1.65, y: 2.5, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$-$' },
    { id: 'fan', x: 2.9, y: 2, shape: 'dot' },
    term('kp', 0.6, '$K_p$'),
    term('ki', 2, '$K_i / s$'),
    term('kd', 3.4, '$K_d\\, s$'),
    { id: 'pl', x: 6.55, y: 0.35, shape: 'text', small: true, w: 1.3, label: 'present' },
    { id: 'il', x: 6.55, y: 1.75, shape: 'text', small: true, w: 1.3, label: 'past' },
    { id: 'dl', x: 6.55, y: 3.15, shape: 'text', small: true, w: 1.3, label: 'future' },
    { id: 's2', x: 7.6, y: 2, shape: 'op', label: '$\\Sigma$' },
    { id: 'P', x: 9.8, y: 2, w: 1.6, h: 0.9, label: '$P(s)$', tone: 1 },
    { id: 'tap', x: 11.4, y: 2, shape: 'dot' },
    { id: 'y', x: 12.4, y: 2, shape: 'text', w: 0.6, label: '$y$' },
  ],
  edges: [
    { from: 'r', to: 's1' },
    { from: 's1', to: 'fan', arrow: 'none', label: '$e$' },
    { from: 'fan', to: 'kp:w', via: [[2.9, 0.6]] },
    { from: 'fan', to: 'ki' },
    { from: 'fan', to: 'kd:w', via: [[2.9, 3.4]] },
    { from: 'kp:e', to: 's2:n', via: [[7.6, 0.6]] },
    { from: 'ki', to: 's2' },
    { from: 'kd:e', to: 's2:s', via: [[7.6, 3.4]] },
    { from: 's2', to: 'P', label: '$u$' },
    { from: 'P', to: 'tap', arrow: 'none' },
    { from: 'tap', to: 'y' },
    {
      from: 'tap:s',
      to: 's1:s',
      via: [
        [11.4, 4.7],
        [1.3, 4.7],
      ],
    },
  ],
  groups: [{ id: 'C', label: 'PID controller', tone: 0, around: ['fan', 'kp', 'kd', 's2', 'pl'], pad: 0.35 }],
}

/** The parallel form of the PID controller inside a unity-feedback loop. */
export function PidDiagram() {
  return (
    <Interactive
      title="The parallel PID controller"
      caption="The error e feeds three branches in parallel: a gain on the present error, an integrator that accumulates past error, and a differentiator that extrapolates the error forward. Their sum is the plant input u. The measured output y is subtracted from the reference r to close the loop."
    >
      <Diagram
        spec={spec}
        ariaLabel="Error e splits into proportional, integral and derivative branches, which are summed into the plant input u; the plant output y is fed back and subtracted from the reference r"
      />
    </Interactive>
  )
}
