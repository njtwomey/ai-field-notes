import { Diagram, Figure, merge, op } from 'aifn-render'
import type { DiagramEdge, DiagramNode } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: number): DiagramNode => ({
  id,
  x,
  y,
  label,
  tone,
  w: 2.2,
  h: 0.6,
})

/** One residual block: `stages` are the boxes on the branch from bottom to top, `after` a norm after the addition. */
function column(p: string, ox: number, title: string, stages: DiagramNode[], after: boolean) {
  const top = 1.6 + (after ? 0.9 : 0)
  const nodes: DiagramNode[] = [
    { id: `${p}t`, x: ox, y: 7.7, shape: 'text', label: title },
    { id: `${p}in`, x: ox, y: 6.9, shape: 'text', label: '$\\xvec_\\ell$' },
    { id: `${p}d`, x: ox, y: 6.2, shape: 'dot' },
    ...stages,
    op(`${p}add`, ox, top, '$+$'),
    ...(after ? [box(`${p}ln`, ox, 1.6, 'norm', 2)] : []),
    { id: `${p}out`, x: ox, y: 0.8, shape: 'text', label: '$\\xvec_{\\ell+1}$' },
  ]
  const chain = [`${p}d`, ...stages.map((s) => s.id), `${p}add`, ...(after ? [`${p}ln`] : []), `${p}out`]
  const edges: DiagramEdge[] = [
    { from: `${p}in`, to: `${p}d`, arrow: 'none' },
    ...chain.slice(1).map((id, i) => ({ from: chain[i], to: id })),
    {
      from: `${p}d:w`,
      to: `${p}add:w`,
      via: [
        [ox - 1.6, 6.2],
        [ox - 1.6, top],
      ],
    },
  ]
  return { nodes, edges }
}

const spec = {
  unit: 34,
  ...merge(
    column('a', 1.8, 'post-norm', [box('aF', 1.8, 4.2, 'sublayer $F$', 0)], true),
    column('b', 6.2, 'pre-norm', [box('bn', 6.2, 5.0, 'norm', 2), box('bF', 6.2, 3.7, 'sublayer $F$', 0)], false),
    column(
      'c',
      10.6,
      'peri-norm',
      [box('cn', 10.6, 5.2, 'norm', 2), box('cF', 10.6, 4.1, 'sublayer $F$', 0), box('cm', 10.6, 3.0, 'norm', 2)],
      false,
    ),
  ),
}

/** The three placements of normalisation in a residual block. */
export function NormPlacements() {
  return (
    <Figure
      title="Where the normalisation sits"
      caption="One residual block in each arrangement; the sublayer F is attention or the feed-forward network. Post-norm normalises the sum, so the identity path from input to output passes through a normalisation. Pre-norm normalises only the branch input, so the identity path is exact and the residual stream is a plain sum of branch outputs. Peri-norm (sandwich) also normalises the branch output before it is added."
    >
      <Diagram spec={spec} ariaLabel="Post-norm, pre-norm and peri-norm residual blocks" />
    </Figure>
  )
}
