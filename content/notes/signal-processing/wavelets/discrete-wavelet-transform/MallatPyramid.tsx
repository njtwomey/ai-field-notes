import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const STEP = 3.6

/** One level of the pyramid: split the approximation at `x0` into a lowpass and a highpass half. */
function level(j: number): { nodes: DiagramNode[]; edges: DiagramEdge[] } {
  const x0 = 1 + (j - 1) * STEP
  const s = `s${j}`
  const down = (id: string, y: number): DiagramNode => ({
    id,
    x: x0 + 2.1,
    y,
    shape: 'circle',
    w: 0.75,
    h: 0.75,
    label: '$\\downarrow 2$',
    tone: 'neutral',
  })
  return {
    nodes: [
      { id: s, x: x0, y: 0, shape: 'dot' },
      { id: `h${j}`, x: x0 + 0.9, y: 0, w: 0.8, h: 0.7, label: '$h$', tone: 0 },
      { id: `g${j}`, x: x0 + 0.9, y: 1.3, w: 0.8, h: 0.7, label: '$g$', tone: 1 },
      down(`ha${j}`, 0),
      down(`gd${j}`, 1.3),
      { id: `d${j}`, x: x0 + 2.1, y: 2.6, shape: 'text', w: 1.6, label: `$d_${j}$ ($N/${2 ** j}$)` },
    ],
    edges: [
      { from: s, to: `h${j}` },
      { from: s, to: `g${j}:w`, via: [[x0, 1.3]] },
      { from: `h${j}`, to: `ha${j}` },
      { from: `g${j}`, to: `gd${j}` },
      { from: `gd${j}`, to: `d${j}` },
    ],
  }
}

const levels = [1, 2, 3].map(level)

const spec: DiagramSpec = {
  unit: 38,
  spread: [1.2, 1],
  nodes: [
    { id: 'x', x: -0.4, y: 0, shape: 'text', w: 1.4, label: '$x = a_0$' },
    ...levels.flatMap((l) => l.nodes),
    { id: 'a3', x: 1 + 3 * STEP, y: 0, shape: 'text', w: 1.6, label: '$a_3$ ($N/8$)' },
  ],
  edges: [
    { from: 'x', to: 's1', arrow: 'none' },
    ...levels.flatMap((l) => l.edges),
    { from: 'ha1', to: 's2', arrow: 'none', label: '$a_1$' },
    { from: 'ha2', to: 's3', arrow: 'none', label: '$a_2$' },
    { from: 'ha3', to: 'a3' },
  ],
}

/** Mallat's algorithm: a cascade of two-channel splits applied to the lowpass branch. */
export function MallatPyramid() {
  return (
    <Interactive
      title="Mallat's pyramid, three levels"
      caption="Each level filters the current approximation with the lowpass filter h and the highpass filter g and keeps every second sample. The highpass output is kept as the detail coefficients d_j; the lowpass output a_j is split again. A signal of N samples gives N/2 + N/4 + N/8 detail coefficients and N/8 approximation coefficients: N in total."
    >
      <Diagram
        spec={spec}
        ariaLabel="Three levels of the discrete wavelet transform: at each level the approximation is filtered by h and g and downsampled by two; the g branch gives the detail coefficients and the h branch feeds the next level"
      />
    </Interactive>
  )
}
