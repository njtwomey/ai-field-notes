import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: -0.6, y: 1.5, shape: 'text', w: 0.9, label: '$x[n]$' },
    { id: 'xtap', x: 0.6, y: 1.5, shape: 'dot' },
    { id: 'fir', x: 3.6, y: 1.5, w: 2.6, h: 1.1, label: 'FIR filter\n$y[n] = \\wvec[n]^\\top\\xvec[n]$', tone: 0 },
    { id: 'sum', x: 6.6, y: 1.5, shape: 'op', label: '$\\Sigma$' },
    { id: 'minus', x: 6.2, y: 1.9, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$-$' },
    { id: 'd', x: 6.6, y: 0, shape: 'text', w: 2.4, label: 'desired $d[n]$' },
    { id: 'etap', x: 8, y: 1.5, shape: 'dot' },
    { id: 'e', x: 9.2, y: 1.5, shape: 'text', w: 0.9, label: '$e[n]$' },
    {
      id: 'upd',
      x: 3.6,
      y: 3.6,
      w: 3.8,
      h: 1.1,
      label: 'LMS update\n$\\wvec[n+1] = \\wvec[n] + \\mu\\, e[n]\\, \\xvec[n]$',
      tone: 2,
    },
  ],
  edges: [
    { from: 'x', to: 'xtap', arrow: 'none' },
    { from: 'xtap', to: 'fir' },
    { from: 'fir', to: 'sum', label: '$y[n]$' },
    { from: 'd', to: 'sum' },
    { from: 'sum', to: 'etap', arrow: 'none' },
    { from: 'etap', to: 'e' },
    { from: 'etap:s', to: 'upd:e', via: [[8, 3.6]], label: 'error' },
    { from: 'xtap:s', to: 'upd:w', via: [[0.6, 3.6]] },
    { from: 'upd:n', to: 'fir:s', dashed: true, tone: 2, label: 'new weights', labelSide: 'right' },
  ],
}

/** The adaptive-filter loop that LMS runs once per sample. */
export function LmsDiagram() {
  return (
    <Figure
      title="The LMS adaptive filter"
      caption="An FIR filter with weights w[n] maps the last M input samples x[n] to the output y[n]. The error e[n] = d[n] − y[n] against the desired signal, multiplied by the input vector and the step size μ, updates the weights before the next sample arrives."
    >
      <Diagram
        spec={spec}
        ariaLabel="The input x passes through an FIR filter to give y; y is subtracted from the desired signal d to give the error e; the error and the input drive the LMS update, which sets the filter's weights"
      />
    </Figure>
  )
}
