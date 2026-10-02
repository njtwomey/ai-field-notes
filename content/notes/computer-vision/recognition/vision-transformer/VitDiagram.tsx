import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramNode, DiagramSpec } from '@/components/diagram/types'

/** Columns of the patch tokens; the class token sits at x = 0.4. */
const PATCH: [string, number, string][] = [
  ['1', 2, '1'],
  ['2', 3.2, '2'],
  ['3', 4.4, '3'],
  ['N', 6.4, 'N'],
]
const CLS = 0.4

const spec: DiagramSpec = {
  nodes: [
    {
      id: 'imgl',
      x: 4.2,
      y: 7.6,
      shape: 'text',
      w: 5,
      small: true,
      label: 'image cut into $N$ patches of $P \\times P$ pixels',
    },
    ...PATCH.flatMap(([k, x, j]): DiagramNode[] => [
      { id: `p${k}`, x, y: 6.7, w: 0.8, h: 0.7, label: `$\\xvec_${j}$`, tone: 'neutral', filled: true },
      { id: `t${k}`, x, y: 3.9, w: 0.8, h: 0.6, label: `$\\zvec_${j}$`, tone: 0 },
      { id: `b${k}`, x, y: 2.75, shape: 'dot' },
    ]),
    { id: 'pd', x: 5.4, y: 6.7, shape: 'text', w: 0.6, label: '$\\cdots$' },
    { id: 'td', x: 5.4, y: 3.9, shape: 'text', w: 0.6, label: '$\\cdots$' },
    { id: 'E', x: 4.2, y: 5.3, w: 5.4, h: 0.7, label: 'shared linear projection $\\Emat$', tone: 'neutral' },
    { id: 'cls', x: CLS, y: 3.9, w: 1.4, h: 0.6, label: '$\\xvec_{\\text{class}}$', tone: 2 },
    { id: 'bcls', x: CLS, y: 2.75, shape: 'dot' },
    {
      id: 'pos',
      x: 9,
      y: 3.9,
      shape: 'text',
      w: 2.8,
      small: true,
      label: '+ position embedding\n$\\evec^{\\text{pos}}_n$ on every token',
    },
    {
      id: 'enc',
      x: 3.4,
      y: 2.1,
      shape: 'stack',
      w: 7.6,
      h: 1.3,
      label: 'transformer encoder, $L$ pre-norm blocks',
      tone: 0,
    },
    { id: 'tcls', x: CLS, y: 1.45, shape: 'dot' },
    { id: 'out', x: CLS, y: 0.2, w: 1.4, h: 0.6, label: '$\\zvec_0^{(L)}$', tone: 2 },
    { id: 'head', x: 3.4, y: 0.2, w: 2.4, h: 0.7, label: 'LN, linear head', tone: 1 },
    { id: 'y', x: 5.6, y: 0.2, shape: 'text', w: 1.2, label: 'class $\\hat\\yvec$' },
  ],
  edges: [
    ...PATCH.flatMap(([k]) => [
      { from: `p${k}`, to: `t${k}` },
      { from: `t${k}`, to: `b${k}` },
    ]),
    { from: 'cls', to: 'bcls' },
    { from: 'tcls', to: 'out' },
    { from: 'out', to: 'head' },
    { from: 'head', to: 'y' },
  ],
}

/** From pixels to a class: patches become tokens, a class token gathers them, and its final state is classified. */
export function VitDiagram() {
  return (
    <Interactive
      title="The vision transformer"
      caption="Each patch is flattened and mapped by the same linear layer to a D-dimensional token, and a learned position embedding is added. A learned class token, tied to no patch, is prepended. After L encoder blocks the class token's state, which has attended to every patch, goes to the classification head; the patch outputs are unused for classification."
    >
      <Diagram
        spec={spec}
        ariaLabel="Vision transformer: image patches projected by a shared linear layer into tokens, a class token prepended, all passed through a transformer encoder; the class token output goes to a linear head that predicts the class"
      />
    </Interactive>
  )
}
