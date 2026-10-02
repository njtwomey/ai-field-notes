import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode } from '@/components/diagram/types'

const LEVELS: [string, string][] = [
  ['$H \\times W$', '64'],
  ['$H/2$', '128'],
  ['$H/4$', '256'],
  ['$H/8$', '512'],
]
const Y = (i: number) => i * 1.5
const ENC_X = 0
const DEC_X = 7.6

const nodes: DiagramNode[] = [
  { id: 'in', x: ENC_X, y: -1.3, shape: 'text', w: 1.6, label: 'image' },
  { id: 'out', x: DEC_X, y: -1.3, shape: 'text', w: 3.6, label: 'class scores, $H \\times W \\times K$' },
  ...LEVELS.flatMap(([res, ch], i): DiagramNode[] => [
    { id: `e${i}`, x: ENC_X, y: Y(i), w: 2.6, h: 0.8, label: `${res}, ${ch} ch`, tone: 0 },
    { id: `d${i}`, x: DEC_X, y: Y(i), w: 2.6, h: 0.8, label: `${res}, ${ch} ch`, tone: 1 },
  ]),
  { id: 'bot', x: (ENC_X + DEC_X) / 2, y: Y(4), w: 2.8, h: 0.8, label: '$H/16$, 1024 ch', tone: 2 },
]

const edges: DiagramEdge[] = [
  { from: 'in', to: 'e0' },
  { from: 'd0', to: 'out' },
  ...[0, 1, 2].flatMap((i): DiagramEdge[] => [
    { from: `e${i}`, to: `e${i + 1}`, label: i === 0 ? 'max pool' : undefined, labelRotate: false, labelSide: 'right' },
    { from: `d${i + 1}`, to: `d${i}`, label: i === 0 ? 'up-conv' : undefined, labelRotate: false, labelSide: 'right' },
  ]),
  { from: 'e3:s', to: 'bot:w', via: [[ENC_X, Y(4)]] },
  { from: 'bot:e', to: 'd3:s', via: [[DEC_X, Y(4)]] },
  ...LEVELS.map((_, i): DiagramEdge => ({
    from: `e${i}`,
    to: `d${i}`,
    dashed: true,
    label: i === 0 ? 'copy and concatenate' : undefined,
  })),
]

/** U-Net: a contracting path, an expanding path, and a skip at every resolution. */
export function UNetDiagram() {
  return (
    <Interactive
      title="U-Net"
      caption="Each box is two 3 × 3 convolutions at the resolution and channel count shown. The encoder halves the resolution and doubles the channels at each stage; the decoder reverses both. The dashed skips copy each encoder map across and concatenate it with the upsampled decoder features at the same resolution, which restores the precise locations lost by pooling."
    >
      <Diagram
        spec={{ nodes, edges }}
        ariaLabel="U-Net: four encoder stages with max pooling down to a bottleneck, four decoder stages with up-convolution back to full resolution, and skip connections concatenating encoder maps into the decoder at each resolution"
      />
    </Interactive>
  )
}
