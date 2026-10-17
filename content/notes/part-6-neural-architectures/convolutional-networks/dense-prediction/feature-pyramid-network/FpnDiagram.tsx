import { Diagram, Figure } from 'aifn-render'
import type { DiagramEdge, DiagramNode } from 'aifn-render'

/** The traced shapes of an 800 × 1024 input on a ResNet with FPN (strides 4 to 32, then P6 at 64). */
const LEVELS = [
  { k: 5, stride: 32, h: 25, w: 32, c: 2048, width: 1.4 },
  { k: 4, stride: 16, h: 50, w: 64, c: 1024, width: 2.0 },
  { k: 3, stride: 8, h: 100, w: 128, c: 512, width: 2.6 },
  { k: 2, stride: 4, h: 200, w: 256, c: 256, width: 3.2 },
]
const ROW = 1.7
const C_X = 0
const OP_X = 4.6
const P_X = 7.4
const y = (i: number) => i * ROW

const nodes: DiagramNode[] = [
  { id: 'img', x: C_X, y: y(4) + 0.4, shape: 'text', w: 3.4, label: 'image, $800 \\times 1024$' },
  ...LEVELS.map((l, i): DiagramNode => ({
    id: `c${l.k}`,
    x: C_X,
    y: y(i),
    w: l.width,
    h: 0.8,
    label: `$C_${l.k}$`,
    tone: 0,
    notes: { w: `$${l.h} \\times ${l.w}$, ${l.c} ch, stride ${l.stride}` },
  })),
  ...LEVELS.map((l, i): DiagramNode => ({
    id: `p${l.k}`,
    x: P_X,
    y: y(i),
    w: l.width,
    h: 0.8,
    label: `$P_${l.k}$`,
    tone: 1,
    notes: { e: `$${l.h} \\times ${l.w}$, 256 ch` },
  })),
  // The adders; at the top a junction, since the 1 × 1 projection of C5 has nothing to add to.
  ...LEVELS.map((l, i): DiagramNode =>
    i === 0
      ? { id: `op${l.k}`, x: OP_X, y: y(i), shape: 'dot' }
      : { id: `op${l.k}`, x: OP_X, y: y(i), shape: 'op', label: '$+$' },
  ),
  { id: 'p6', x: P_X, y: y(-1), w: 1.0, h: 0.8, label: '$P_6$', tone: 1, notes: { e: '$13 \\times 16$, 256 ch' } },
]

const edges: DiagramEdge[] = [
  { from: 'img', to: 'c2' },
  // Bottom-up: each stage halves the resolution.
  ...LEVELS.slice(1).map((l, i): DiagramEdge => ({
    from: `c${l.k}`,
    to: `c${l.k + 1}`,
    label: i === 2 ? 'bottom-up, stride 2' : undefined,
    labelSide: 'right',
    labelRotate: false,
  })),
  // Lateral 1 × 1 convolutions into the adders, and the adders' 3 × 3 convolutions out.
  ...LEVELS.map((l, i): DiagramEdge => ({
    from: `c${l.k}`,
    to: `op${l.k}`,
    label: i === 1 ? 'lateral $1 \\times 1$' : undefined,
  })),
  ...LEVELS.map((l, i): DiagramEdge => ({
    from: `op${l.k}`,
    to: `p${l.k}`,
    label: i === 1 ? '$3 \\times 3$' : undefined,
  })),
  // Top-down: the merged map of the level above, upsampled by 2 (nearest), into the adder below.
  ...LEVELS.slice(1).map((l, i): DiagramEdge => ({
    from: `op${l.k + 1}`,
    to: `op${l.k}`,
    label: i === 0 ? 'top-down, $2\\times$ up' : undefined,
    labelSide: 'left',
    labelRotate: false,
    tone: 1,
  })),
  { from: 'p5', to: 'p6', label: 'subsample, stride 2', labelRotate: false, labelSide: 'right' },
]

/** FPN on a ResNet: bottom-up stages, lateral 1 × 1 convolutions, a top-down path of upsampling and addition. */
export function FpnDiagram() {
  return (
    <Figure
      title="Feature pyramid network on an 800 × 1024 image"
      caption="Left: the bottom-up stages of a ResNet, C2 to C5, at strides 4 to 32, with their map sizes and channels for an 800 × 1024 input. Right: the pyramid P2 to P6, all with 256 channels. Each adder takes the lateral 1 × 1 projection of the backbone map and the merged map of the level above upsampled by 2, so semantics from C5 reach every level; a 3 × 3 convolution then smooths the sum. At the top, the 1 × 1 projection of C5 has no level above it and passes straight on. P6 is a stride-2 subsampling of P5, used only for the largest anchors."
    >
      <Diagram
        spec={{ nodes, edges }}
        ariaLabel="Feature pyramid network: backbone stages C2 to C5 on the left, pyramid levels P2 to P6 on the right, lateral connections and a top-down path of upsampling and addition between them"
      />
    </Figure>
  )
}
