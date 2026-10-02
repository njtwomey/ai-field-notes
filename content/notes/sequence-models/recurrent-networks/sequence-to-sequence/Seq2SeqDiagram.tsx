import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const state = (id: string, x: number, label: string, tone: number) => ({ id, x, y: 2, w: 1.1, h: 0.75, label, tone })
const token = (id: string, x: number, y: number, label: string) => ({ id, x, y, shape: 'text' as const, label })

const spec: DiagramSpec = {
  nodes: [
    state('h1', 0, '$\\hvec_1$', 0),
    state('h2', 1.9, '$\\hvec_2$', 0),
    { id: 'hd', x: 3.35, y: 2, shape: 'text', w: 0.5, label: '$\\cdots$' },
    state('hS', 4.8, '$\\hvec_S$', 0),
    token('x1', 0, 3.5, '$x_1$'),
    token('x2', 1.9, 3.5, '$x_2$'),
    token('xS', 4.8, 3.5, '$x_S$'),
    state('s1', 8, '$\\svec_1$', 1),
    state('s2', 9.9, '$\\svec_2$', 1),
    { id: 'sd', x: 11.35, y: 2, shape: 'text', w: 0.5, label: '$\\cdots$' },
    state('sT', 12.8, '$\\svec_T$', 1),
    token('i1', 8, 3.5, '$y_0$ (start)'),
    token('i2', 9.9, 3.5, '$y_1$'),
    token('iT', 12.8, 3.5, '$y_{T-1}$'),
    token('o1', 8, 0.5, '$y_1$'),
    token('o2', 9.9, 0.5, '$y_2$'),
    token('oT', 12.8, 0.5, '$y_T$ (end)'),
  ],
  edges: [
    { from: 'h1', to: 'h2' },
    { from: 'h2', to: 'hd', arrow: 'none' },
    { from: 'hd', to: 'hS' },
    { from: 'hS', to: 's1', label: '$\\cvec = \\hvec_S$' },
    { from: 's1', to: 's2' },
    { from: 's2', to: 'sd', arrow: 'none' },
    { from: 'sd', to: 'sT' },
    { from: 'x1', to: 'h1' },
    { from: 'x2', to: 'h2' },
    { from: 'xS', to: 'hS' },
    { from: 'i1', to: 's1' },
    { from: 'i2', to: 's2' },
    { from: 'iT', to: 'sT' },
    { from: 's1', to: 'o1', label: 'softmax', labelSide: 'right', labelRotate: false },
    { from: 's2', to: 'o2' },
    { from: 'sT', to: 'oT' },
  ],
  groups: [
    { id: 'enc', label: 'encoder', tone: 0, around: ['h1', 'hS', 'x1'], pad: 0.4 },
    { id: 'dec', label: 'decoder', tone: 1, around: ['s1', 'sT', 'i1', 'o1'], pad: 0.4, labelAt: 'bottom-right' },
  ],
}

/** The encoder reads the whole input into one vector; the decoder writes the output from it, one token at a time. */
export function Seq2SeqDiagram() {
  return (
    <Interactive
      title="Encoder and decoder, unrolled"
      caption="The encoder RNN reads the input and passes only its final state to the decoder. The decoder RNN starts from that context vector and, at each step, reads the token it emitted at the previous step and outputs a distribution over the next one."
    >
      <Diagram
        spec={spec}
        ariaLabel="Sequence-to-sequence model: encoder states h_1 to h_S over the input tokens, final state c passed to decoder states s_1 to s_T, each emitting an output token"
      />
    </Interactive>
  )
}
