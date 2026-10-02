import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramNode, DiagramSpec } from '@/components/diagram/types'

const COLS: { k: string; x: number; inp: string; h: string; out: string; th: string }[] = [
  {
    k: 'a',
    x: 0,
    inp: '$z_{i,t_0-2}$, $\\xvec_{i,t_0-1}$',
    h: '$\\hvec_{i,t_0-1}$',
    out: '$z_{i,t_0-1}$ observed',
    th: 't_0-1',
  },
  {
    k: 'b',
    x: 3.8,
    inp: '$z_{i,t_0-1}$, $\\xvec_{i,t_0}$',
    h: '$\\tilde\\hvec_{i,t_0}$',
    out: '$\\tilde z_{i,t_0}$ sampled',
    th: 't_0',
  },
  {
    k: 'c',
    x: 7.6,
    inp: '$\\tilde z_{i,t_0}$, $\\xvec_{i,t_0+1}$',
    h: '$\\tilde\\hvec_{i,t_0+1}$',
    out: '$\\tilde z_{i,t_0+1}$ sampled',
    th: 't_0+1',
  },
]

const spec: DiagramSpec = {
  nodes: [
    ...COLS.flatMap(({ k, x, inp, h, out, th }): DiagramNode[] => [
      { id: `in${k}`, x, y: 4.4, shape: 'text', w: 2.6, label: inp },
      { id: `h${k}`, x, y: 3, w: 1.8, h: 0.8, label: h, tone: 0 },
      { id: `l${k}`, x, y: 1.5, w: 2.6, h: 0.8, label: `$\\ell(\\cdot \\mid \\thetavec_{i,${th}})$`, tone: 1 },
      { id: `o${k}`, x, y: 0.2, shape: 'text', w: 2.6, label: out },
    ]),
    { id: 'hin', x: -2.2, y: 3, shape: 'text', w: 0.6, label: '$\\cdots$' },
    { id: 'hout', x: 9.8, y: 3, shape: 'text', w: 0.6, label: '$\\cdots$' },
  ],
  edges: [
    ...COLS.flatMap(({ k }) => [
      { from: `in${k}`, to: `h${k}` },
      { from: `h${k}`, to: `l${k}` },
      { from: `l${k}`, to: `o${k}` },
    ]),
    { from: 'hin', to: 'ha' },
    { from: 'ha', to: 'hb' },
    { from: 'hb', to: 'hc' },
    { from: 'hc', to: 'hout' },
    {
      from: 'ob:e',
      to: 'inc:w',
      via: [
        [5.7, 0.2],
        [5.7, 4.4],
      ],
      dashed: true,
    },
  ],
  groups: [
    { id: 'cond', label: 'conditioning range', tone: 'neutral', dashed: true, around: ['ina', 'oa'], pad: 0.25 },
    { id: 'pred', label: 'prediction range', tone: 1, dashed: true, around: ['inb', 'oc'], pad: 0.25 },
  ],
}

/** DeepAR unrolled across the forecast origin: observed values feed the recurrence before it, samples after it. */
export function DeepArDiagram() {
  return (
    <Interactive
      title="DeepAR at the forecast origin"
      caption="One recurrent network with the same weights runs over both ranges. Each step reads the previous value and the current covariates, and an affine layer maps the hidden state to the parameters of the likelihood. In the conditioning range the previous value is observed. In the prediction range it is the previous step's sample (dashed), so one pass draws one joint sample path."
    >
      <Diagram
        spec={spec}
        ariaLabel="DeepAR unrolled: at each step the previous value and covariates enter the recurrent state, which gives the likelihood parameters; after the forecast origin each sampled value is fed to the next step"
      />
    </Interactive>
  )
}
