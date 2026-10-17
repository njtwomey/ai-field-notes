import { useMemo, useState } from 'react'
import {
  Curve,
  Diagram,
  Figure,
  formatNumber,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  type DiagramSpec,
} from 'aifn-render'
import { networkCost } from 'aifn-compute/nn/functional'
import { modernisationLadder } from '../_shared/architectures.ts'

/** The ladder, counted once: each step is a whole network. */
const STEPS = modernisationLadder().map((step) => {
  const c = networkCost(step.model.stages, step.model.input)
  return {
    ...step,
    params: c.params / 1e6,
    macs: c.multiplies / 1e9,
    depth: c.depth,
    field: c.layers[1].receptiveField.size[0],
  }
})
const LABELS = STEPS.map((s) => s.label)
const AT = STEPS.map((_, i) => i)

/** The three layers of a stage-1 residual block at each step, top to bottom, and the operation after the sum. */
const BLOCKS: [string, string, string, string][] = [
  ['1×1, 64\nBN, ReLU', '3×3, 64\nBN, ReLU', '1×1, 256\nBN', 'ReLU'],
  ['1×1, 64\nBN, ReLU', '3×3, 64\nBN, ReLU', '1×1, 256\nBN', 'ReLU'],
  ['1×1, 64\nBN, ReLU', '3×3, 64\nBN, ReLU', '1×1, 256\nBN', 'ReLU'],
  ['1×1, 64\nBN, ReLU', 'd3×3, 64\nBN, ReLU', '1×1, 256\nBN', 'ReLU'],
  ['1×1, 96\nBN, ReLU', 'd3×3, 96\nBN, ReLU', '1×1, 384\nBN', 'ReLU'],
  ['1×1, 384\nBN, ReLU', 'd3×3, 384\nBN, ReLU', '1×1, 96\nBN', 'ReLU'],
  ['d3×3, 96\nBN, ReLU', '1×1, 384\nBN, ReLU', '1×1, 96\nBN', 'ReLU'],
  ['d7×7, 96\nBN, ReLU', '1×1, 384\nBN, ReLU', '1×1, 96\nBN', 'ReLU'],
  ['d7×7, 96\nLN', '1×1, 384\nGELU', '1×1, 96', 'none'],
  ['d7×7, 96\nLN', '1×1, 384\nGELU', '1×1, 96\nlayer scale', 'none'],
]

function blockSpec(step: number): DiagramSpec {
  const [a, b, c, after] = BLOCKS[step]
  const prev = step > 0 ? BLOCKS[step - 1] : BLOCKS[0]
  const changed = (i: number) => step > 0 && prev[i] !== BLOCKS[step][i]
  return {
    nodes: [
      { id: 'x', x: 0, y: 1.5, shape: 'text', w: 0.6, label: '$\\xvec$' },
      { id: 'split', x: 1, y: 1.5, shape: 'dot' },
      { id: 'a', x: 2.9, y: 1.5, w: 2.1, h: 1, label: a, tone: 0, highlight: changed(0) },
      { id: 'b', x: 5.4, y: 1.5, w: 2.1, h: 1, label: b, tone: 0, highlight: changed(1) },
      { id: 'c', x: 7.9, y: 1.5, w: 2.1, h: 1, label: c, tone: 0, highlight: changed(2) },
      { id: 'add', x: 9.6, y: 1.5, shape: 'op', label: '$\\oplus$' },
      {
        id: 'after',
        x: 10.9,
        y: 1.5,
        shape: 'pill',
        w: 1.1,
        h: 0.6,
        label: after,
        tone: 'neutral',
        small: true,
        highlight: changed(3),
      },
    ],
    edges: [
      { from: 'x', to: 'split', arrow: 'none' },
      { from: 'split', to: 'a' },
      { from: 'a', to: 'b' },
      { from: 'b', to: 'c' },
      { from: 'c', to: 'add' },
      { from: 'add', to: 'after' },
      {
        from: 'split:n',
        to: 'add:n',
        via: [
          [1, 0.2],
          [9.6, 0.2],
        ],
        label: 'shortcut',
      },
    ],
    unit: 36,
  }
}

/** ResNet-50 to ConvNeXt-T one design change at a time, every step counted as a whole network. */
export function ConvNeXtLadder() {
  const [step, setStep] = useState(0)
  const s = STEPS[step]
  const spec = useMemo(() => blockSpec(step), [step])

  const x = useAxis({ label: 'step', categories: LABELS })
  const yMacs = useAxis({ label: 'multiply-adds (billions)', range: [0, 6] })
  const yParams = useAxis({ label: 'parameters (millions)', range: [0, 40] })
  return (
    <Figure
      title="From ResNet-50 to ConvNeXt-T, one change at a time"
      defaultSize="L"
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={STEPS.length}
          label="step"
          format={(i) => `${i}: ${STEPS[i].label}`}
        />
      }
      caption="Step through the modernisation of Liu et al. (2022). Top: one residual block of stage 1 at this step, with the layers changed by the step highlighted (d marks a depthwise convolution). Bottom: multiply-adds per 224 × 224 image and parameters of the whole network after each step, counted from its layer specification; the counts agree with the paper's reported GFLOPs within 3%. The depthwise step halves the cost, widening to 96 channels buys it back, and the inverted bottleneck and the move of the depthwise layer bring it to about ResNet-50's 4.1 billion. The receptive field readout is that of stage 1's output, which the 7 × 7 kernels nearly triple."
      readouts={
        <>
          <Readout label="change" value={s.change} />
          <Readout label="multiply-adds" value={`${formatNumber(s.macs)} G`} />
          <Readout label="parameters" value={`${formatNumber(s.params)} M`} />
          <Readout label="weight layers" value={s.depth} />
          <Readout label="receptive field after stage 1" value={`${s.field} px`} />
        </>
      }
    >
      <Diagram spec={spec} ariaLabel={`Stage-1 block at step ${step}: ${BLOCKS[step].slice(0, 3).join(', ')}`} />
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={x} y={yMacs} height={220} title="multiply-adds">
          <Curve name="multiply-adds" x={AT} y={STEPS.map((t) => t.macs)} slot={0} showPoints />
          <Points name="this step" x={[step]} y={[s.macs]} emphasis size={10} />
        </Plot>
        <Plot x={x} y={yParams} height={220} title="parameters">
          <Curve name="parameters" x={AT} y={STEPS.map((t) => t.params)} slot={1} showPoints />
          <Points name="this step" x={[step]} y={[s.params]} emphasis size={10} />
        </Plot>
      </div>
    </Figure>
  )
}
