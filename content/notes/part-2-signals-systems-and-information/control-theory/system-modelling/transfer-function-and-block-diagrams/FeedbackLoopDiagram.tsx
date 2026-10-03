import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const sign = (id: string, x: number, y: number, s: string) => ({
  id,
  x,
  y,
  shape: 'text' as const,
  small: true,
  w: 0.3,
  h: 0.3,
  label: s,
})

const spec: DiagramSpec = {
  nodes: [
    { id: 'r', x: 0, y: 0, shape: 'text', w: 0.6, label: '$r$' },
    { id: 's1', x: 1.6, y: 0, shape: 'op', label: '$\\Sigma$' },
    sign('s1p', 1.2, -0.35, '$+$'),
    sign('s1m', 1.95, 0.5, '$-$'),
    { id: 'C', x: 3.8, y: 0, w: 1.6, h: 0.9, label: '$C(s)$', tone: 0 },
    { id: 'P', x: 6.8, y: 0, w: 1.6, h: 0.9, label: '$P(s)$', tone: 1 },
    { id: 's2', x: 8.8, y: 0, shape: 'op', label: '$\\Sigma$' },
    { id: 'd', x: 8.8, y: -1.4, shape: 'text', w: 1.6, label: 'disturbance $d$' },
    { id: 'tap', x: 10.2, y: 0, shape: 'dot' },
    { id: 'y', x: 11.4, y: 0, shape: 'text', w: 0.6, label: '$y$' },
    { id: 's3', x: 10.2, y: 2.2, shape: 'op', label: '$\\Sigma$' },
    { id: 'n', x: 12.6, y: 2.2, shape: 'text', w: 1.8, label: 'sensor noise $n$' },
  ],
  edges: [
    { from: 'r', to: 's1' },
    { from: 's1', to: 'C', label: '$e$' },
    { from: 'C', to: 'P', label: '$u$' },
    { from: 'P', to: 's2' },
    { from: 'd', to: 's2' },
    { from: 's2', to: 'tap', arrow: 'none' },
    { from: 'tap', to: 'y' },
    { from: 'tap:s', to: 's3:n' },
    { from: 'n', to: 's3' },
    { from: 's3:w', to: 's1:s', via: [[1.6, 2.2]], label: 'measured output $y_m = y + n$', labelSide: 'right' },
  ],
}

/** The standard single-loop feedback system. */
export function FeedbackLoopDiagram() {
  return (
    <Interactive
      title="The standard feedback loop"
      caption="The controller C acts on the error e between the reference r and the measured output. The plant P turns the input u into the output y, to which the disturbance d adds. The sensor adds noise n. Each signal reaches y through S = 1/(1 + PC) or T = PC/(1 + PC)."
    >
      <Diagram
        spec={spec}
        ariaLabel="Reference r minus measured output gives error e, which passes through controller C and plant P; a disturbance adds to the plant output y, and sensor noise adds to the fed-back measurement"
      />
    </Interactive>
  )
}
