import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const PAST = 4.4
const FUT = 8.8
const MID = (PAST + FUT) / 2

const spec: DiagramSpec = {
  nodes: [
    { id: 'sin', x: 0, y: 8.2, w: 2.4, h: 0.8, label: 'static $\\svec_i$', tone: 'neutral' },
    { id: 'pin', x: PAST, y: 8.2, w: 3.2, h: 0.8, label: 'past $y$, $\\zvec$, $\\xvec$', tone: 'neutral' },
    { id: 'fin', x: FUT, y: 8.2, w: 3.2, h: 0.8, label: 'known future $\\xvec$', tone: 'neutral' },
    { id: 'svs', x: 0, y: 6.8, w: 2.4, h: 0.8, label: 'selection', tone: 2 },
    { id: 'pvs', x: PAST, y: 6.8, w: 3.2, h: 0.8, label: 'variable selection', tone: 2 },
    { id: 'fvs', x: FUT, y: 6.8, w: 3.2, h: 0.8, label: 'variable selection', tone: 2 },
    { id: 'senc', x: 0, y: 5.2, w: 2.4, h: 1, label: 'static encoders\n4 GRNs', tone: 2 },
    { id: 'j', x: 2, y: 5.2, shape: 'dot' },
    { id: 'lenc', x: PAST, y: 5.2, w: 3.2, h: 0.8, label: 'LSTM encoder', tone: 0 },
    { id: 'ldec', x: FUT, y: 5.2, w: 3.2, h: 0.8, label: 'LSTM decoder', tone: 0 },
    { id: 'enr', x: MID, y: 3.4, w: 7.6, h: 0.8, label: 'static enrichment, GRN per position', tone: 'neutral' },
    { id: 'att', x: MID, y: 2.1, w: 7.6, h: 0.8, label: 'masked interpretable multi-head attention', tone: 1 },
    { id: 'ff', x: MID, y: 0.8, w: 7.6, h: 0.8, label: 'gated skip, position-wise GRN', tone: 'neutral' },
    { id: 'q', x: FUT, y: -0.6, w: 3.8, h: 0.8, label: 'quantile heads $\\hat y(q, t, \\tau)$', tone: 'neutral' },
  ],
  edges: [
    { from: 'sin', to: 'svs' },
    { from: 'pin', to: 'pvs' },
    { from: 'fin', to: 'fvs' },
    { from: 'svs', to: 'senc' },
    { from: 'pvs', to: 'lenc' },
    { from: 'fvs', to: 'ldec' },
    { from: 'senc', to: 'j', arrow: 'none' },
    { from: 'j', to: 'lenc', label: '$\\cvec_c, \\cvec_h$', labelOffset: 0.15 },
    { from: 'j', to: 'pvs:w', via: [[2, 6.8]], label: '$\\cvec_s$', labelRotate: false, labelSide: 'right' },
    { from: 'senc:n', to: 'enr:w', via: [[0, 3.4]], label: '$\\cvec_e$', labelRotate: false, labelSide: 'right' },
    { from: 'lenc', to: 'ldec' },
    {
      from: 'lenc:n',
      to: 'enr:s',
      via: [
        [PAST, 4.3],
        [MID, 4.3],
      ],
    },
    {
      from: 'ldec:n',
      to: 'enr:s',
      via: [
        [FUT, 4.3],
        [MID, 4.3],
      ],
    },
    { from: 'enr', to: 'att' },
    { from: 'att', to: 'ff' },
    { from: 'ff:n', to: 'q:w', via: [[MID, -0.6]] },
    {
      from: 'enr:e',
      to: 'ff:e',
      via: [
        [MID + 4.4, 3.4],
        [MID + 4.4, 0.8],
      ],
      dashed: true,
      label: 'gated skip',
    },
  ],
}

/** Where each input enters the temporal fusion transformer, from the bottom up. */
export function TftDiagram() {
  return (
    <Figure
      title="The temporal fusion transformer"
      caption="Read from the bottom. Static, past and known-future inputs each pass through their own variable selection network. Static context vectors condition the selection, initialise the LSTM encoder and enrich the temporal features. The LSTM encoder–decoder handles local patterns, a single masked attention layer handles long-range ones, and gated skips let the model bypass any block. Linear quantile heads read the future positions."
    >
      <Diagram
        spec={spec}
        ariaLabel="Temporal fusion transformer: static, past and future inputs through variable selection; static encoders produce context vectors; LSTM encoder and decoder; static enrichment; masked attention; gated position-wise GRN; quantile heads"
      />
    </Figure>
  )
}
