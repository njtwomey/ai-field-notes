import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 34,
  nodes: [
    {
      id: 'seq',
      x: 1.8,
      y: 10.6,
      w: 3.4,
      h: 0.8,
      label: 'last 100 actions:\ntype, PinSage, time',
      tone: 1,
      small: true,
    },
    { id: 'cand', x: 5.6, y: 10.6, w: 2.6, h: 0.8, label: 'candidate pin\nPinSage', tone: 2, small: true },
    {
      id: 'fuse',
      x: 3.6,
      y: 9,
      w: 5.4,
      h: 0.7,
      label: 'early fusion: concatenate candidate to every action',
      tone: 'neutral',
      small: true,
    },
    {
      id: 'mask',
      x: 3.6,
      y: 7.8,
      w: 5.4,
      h: 0.7,
      label: 'random time-window mask (training only)',
      tone: 'neutral',
      small: true,
      dashed: true,
    },
    {
      id: 'tf',
      x: 3.6,
      y: 6.3,
      w: 5.4,
      h: 1.1,
      shape: 'stack',
      label: 'transformer encoder: 2 layers, 1 head',
      tone: 0,
    },
    { id: 'topk', x: 2.2, y: 4.8, w: 2.4, h: 0.7, label: 'first $K$ outputs', tone: 0, small: true },
    { id: 'max', x: 5.0, y: 4.8, w: 2.4, h: 0.7, label: 'max pool', tone: 0, small: true },
    { id: 'z', x: 3.6, y: 3.6, w: 2.6, h: 0.7, label: 'flatten to $\\zvec$', tone: 0, small: true },
    {
      id: 'static',
      x: 9.4,
      y: 10.6,
      w: 3.2,
      h: 0.8,
      label: 'user, pin, context\nfeatures',
      tone: 'neutral',
      small: true,
    },
    { id: 'pf', x: 12.6, y: 10.6, w: 2.6, h: 0.8, label: 'PinnerFormer\n(daily batch)', tone: 1, small: true },
    { id: 'emb', x: 9.4, y: 8.4, w: 3.2, h: 0.7, label: 'embedding layers', tone: 'neutral', small: true },
    { id: 'cat', x: 10.6, y: 3.6, shape: 'op', label: '$\\Vert$' },
    { id: 'dcn', x: 10.6, y: 2.4, w: 3, h: 0.7, label: 'DCN V2 feature cross', tone: 'neutral', small: true },
    { id: 'mlp', x: 10.6, y: 1.3, w: 3, h: 0.7, label: 'MLP', tone: 'neutral', small: true },
    { id: 'heads', x: 10.6, y: 0.1, w: 4.6, h: 0.7, label: 'heads: click, repin, hide, …', tone: 'ink', small: true },
  ],
  edges: [
    { from: 'seq', to: 'fuse:s', via: [[1.8, 9.6]] },
    { from: 'cand', to: 'fuse:s', via: [[5.6, 9.6]] },
    { from: 'fuse', to: 'mask' },
    { from: 'mask', to: 'tf' },
    { from: 'tf:n', to: 'topk:s', via: [[2.2, 5.5]] },
    { from: 'tf:n', to: 'max:s', via: [[5.0, 5.5]] },
    { from: 'topk:n', to: 'z:s', via: [[2.2, 4.2]] },
    { from: 'max:n', to: 'z:s', via: [[5.0, 4.2]] },
    { from: 'z', to: 'cat:w' },
    { from: 'static', to: 'emb' },
    {
      from: 'emb:n',
      to: 'cat:s',
      via: [
        [9.4, 4.6],
        [10.6, 4.6],
      ],
    },
    { from: 'pf:n', to: 'cat:e', via: [[12.6, 3.6]] },
    { from: 'cat', to: 'dcn' },
    { from: 'dcn', to: 'mlp' },
    { from: 'mlp', to: 'heads' },
  ],
  groups: [
    { id: 'transact', label: 'TransAct (real time)', around: ['fuse', 'mask', 'tf', 'topk', 'max', 'z'], pad: 0.45 },
    {
      id: 'pinnability',
      label: 'Pinnability ranker',
      around: ['emb', 'cat', 'dcn', 'mlp', 'heads'],
      pad: 0.45,
      labelAt: 'top-right',
    },
  ],
}

/** TransAct inside Pinterest's homefeed ranking model, next to the daily PinnerFormer embedding. */
export function TransActDiagram() {
  return (
    <Interactive
      title="TransAct inside the homefeed ranker"
      caption="For each candidate pin, the candidate's PinSage embedding is concatenated to every action in the user's real-time sequence before a small transformer encodes it. The first K output positions and an element-wise max over all positions are flattened and joined with the static features and the daily PinnerFormer user embedding, crossed by DCN V2 and mapped to one probability per action type."
    >
      <Diagram
        spec={spec}
        ariaLabel="TransAct: real-time actions fused with the candidate pin, masked, encoded by a two-layer transformer, compressed by first-K outputs and max pooling, then concatenated with static features and PinnerFormer in the Pinnability ranker with DCN V2, MLP and action heads"
      />
    </Interactive>
  )
}
