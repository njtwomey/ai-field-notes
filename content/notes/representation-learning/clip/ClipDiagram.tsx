import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramNode, DiagramSpec } from '@/components/diagram/types'

const IDX = ['1', '2', '3', 'N']
const COL = [4, 5, 6, 7]
const ROW = [2.6, 3.5, 4.4, 5.3]

const cells: DiagramNode[] = ROW.flatMap((y, i) =>
  COL.map((x, j) => ({
    id: `c${i}${j}`,
    x,
    y,
    w: 0.8,
    h: 0.7,
    label: i === j ? `$\\ell_{${IDX[i]}${IDX[j]}}$` : '',
    tone: i === j ? 2 : ('neutral' as const),
    small: true,
  })),
)

const spec: DiagramSpec = {
  nodes: [
    { id: 'cap', x: 5.5, y: -1.1, shape: 'text', w: 1.4, label: 'captions' },
    { id: 'g', x: 5.5, y: 0, w: 4, h: 0.8, label: 'text encoder $g$, project, normalise', tone: 1 },
    ...COL.map((x, j) => ({ id: `v${j}`, x, y: 1.4, w: 0.8, h: 0.6, label: `$\\vvec_${IDX[j]}$`, tone: 1 })),
    { id: 'img', x: -3.6, y: 3.95, shape: 'text', w: 1.2, label: 'images' },
    { id: 'f', x: 0, y: 3.95, w: 2.8, h: 1.1, label: 'image encoder $f$\nproject, normalise', tone: 0 },
    ...ROW.map((y, i) => ({ id: `u${i}`, x: 2.6, y, w: 0.8, h: 0.6, label: `$\\uvec_${IDX[i]}$`, tone: 0 })),
    ...cells,
    {
      id: 'note',
      x: 10.2,
      y: 3.95,
      w: 3.6,
      h: 1.4,
      shape: 'text',
      small: true,
      label:
        '$\\ell_{ij} = \\uvec_i\\transpose \\vvec_j / \\tau$\ncross-entropy along each row\nand down each column;\nthe diagonal is the target',
    },
  ],
  edges: [
    { from: 'cap', to: 'g' },
    ...COL.map((_, j) => ({ from: 'g', to: `v${j}`, route: 'straight' as const })),
    { from: 'img', to: 'f' },
    ...ROW.map((_, i) => ({ from: 'f', to: `u${i}`, route: 'straight' as const })),
  ],
}

/** Two encoders into one space, trained so that the matching pairs sit on the diagonal of the similarity matrix. */
export function ClipDiagram() {
  return (
    <Interactive
      title="CLIP's contrastive objective"
      caption="A batch of N image–caption pairs is embedded by the two encoders into unit vectors in one shared space. Every image is scored against every caption. The N matching pairs lie on the diagonal of the N × N logit matrix, and the loss is a softmax classification of the diagonal entry in each row and each column."
    >
      <Diagram
        spec={spec}
        ariaLabel="CLIP: text encoder produces caption vectors v_1 to v_N, image encoder produces image vectors u_1 to u_N, and their N by N matrix of scaled dot products has the matching pairs on the diagonal"
      />
    </Interactive>
  )
}
