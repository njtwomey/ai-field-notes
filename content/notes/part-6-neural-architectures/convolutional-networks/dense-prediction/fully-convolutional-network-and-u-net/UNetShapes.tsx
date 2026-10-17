import { useMemo } from 'react'
import { choice, Diagram, Figure, formatNumber, int, Readout, slider, useFigureState } from 'aifn-render'
import type { DiagramEdge, DiagramNode } from 'aifn-render'
import { convolutionCost } from 'aifn-compute/nn/functional'

/** One input channel and two classes, as in the original U-Net. */
const IN_CHANNELS = 1
const CLASSES = 2

type Stage = { from: number; to: number; channels: number }
type Plan =
  | {
      ok: true
      encoder: Stage[]
      bottleneck: Stage
      decoder: (Stage & { crop: number })[]
      output: number
      params: number
      multiplies: number
    }
  | { ok: false; reason: string }

/**
 * Trace a U-Net of `depth` poolings: two 3 × 3 convolutions per stage (valid or same padding), 2 × 2 max pooling, a
 * 2 × 2 stride-2 transposed convolution up, crop and concatenate the skip, two 3 × 3 convolutions, a final 1 × 1.
 */
function plan(depth: number, base: number, padding: number, input: number): Plan {
  let size = input
  let channels = IN_CHANNELS
  let params = 0
  let multiplies = 0
  const layer = (spec: Parameters<typeof convolutionCost>[0]) => {
    const c = convolutionCost(spec)
    params += c.params
    multiplies += c.multiplies
    size = c.outputShape[0]
    channels = c.outChannels
  }
  const twoConvs = (out: number) => {
    layer({ inChannels: channels, outChannels: out, kernel: 3, padding, input: [size, size] })
    layer({ inChannels: channels, outChannels: out, kernel: 3, padding, input: [size, size] })
  }
  try {
    const encoder: Stage[] = []
    for (let i = 0; i < depth; i++) {
      const from = size
      twoConvs(base * 2 ** i)
      encoder.push({ from, to: size, channels })
      if (size % 2 !== 0)
        return { ok: false, reason: `stage ${i + 1} ends at ${size} pixels, which 2 × 2 pooling cannot halve` }
      size /= 2
    }
    const from = size
    twoConvs(base * 2 ** depth)
    const bottleneck = { from, to: size, channels }
    const decoder: (Stage & { crop: number })[] = []
    for (let i = depth - 1; i >= 0; i--) {
      layer({
        kind: 'transposed',
        inChannels: channels,
        outChannels: base * 2 ** i,
        kernel: 2,
        stride: 2,
        input: [size, size],
      })
      const up = size
      const crop = (encoder[i].to - up) / 2
      channels *= 2 // the cropped encoder map is concatenated
      twoConvs(base * 2 ** i)
      decoder.unshift({ from: up, to: size, channels, crop })
    }
    layer({ inChannels: channels, outChannels: CLASSES, kernel: 1, input: [size, size] })
    return { ok: true, encoder, bottleneck, decoder, output: size, params, multiplies }
  } catch {
    return { ok: false, reason: 'the input is too small: a valid convolution has nothing left to read' }
  }
}

/** The nearest input sizes below and above `n` that trace without an odd pooling. */
function nearestValid(depth: number, base: number, padding: number, n: number): [number | null, number | null] {
  const fits = (m: number) => m >= 1 && plan(depth, base, padding, m).ok
  let below: number | null = null
  let above: number | null = null
  for (let d = 1; d <= 64 && (below === null || above === null); d++) {
    if (below === null && fits(n - d)) below = n - d
    if (above === null && fits(n + d)) above = n + d
  }
  return [below, above]
}

const sq = (n: number) => `${n}^2`
const shape = (s: Stage) => (s.from === s.to ? `$${sq(s.to)}$` : `$${sq(s.from)} \\to ${sq(s.to)}$`)
const Y = (i: number) => i * 1.4
const ENC_X = 0
const DEC_X = 8.4

/** U-Net with its tensor shape at every stage, traced for any depth, width, padding and input size. */
export function UNetShapes() {
  const state = useFigureState({
    depth: slider(2, 5, 4, { step: 1, label: 'poolings (depth)' }),
    base: choice([16, 32, 64], 64, { label: 'channels at stage 1' }),
    padding: choice(['valid', 'same'] as const, 'valid', { label: 'padding of the 3 × 3 convolutions' }),
    input: int(572, { ge: 16, le: 2048, suggestions: [572, 512, 256, 188], label: 'input side (pixels)' }),
  })
  const pad = state.padding === 'same' ? 1 : 0
  const p = useMemo(() => plan(state.depth, state.base, pad, state.input), [state.depth, state.base, pad, state.input])
  const valid = useMemo(
    () => (p.ok ? null : nearestValid(state.depth, state.base, pad, state.input)),
    [p.ok, state.depth, state.base, pad, state.input],
  )

  const spec = useMemo(() => {
    if (!p.ok) return null
    const depth = p.encoder.length
    const nodes: DiagramNode[] = [
      { id: 'in', x: ENC_X, y: -1.2, shape: 'text', w: 3, label: `image, $${sq(state.input)}$` },
      {
        id: 'out',
        x: DEC_X,
        y: -1.2,
        shape: 'text',
        w: 4.4,
        label: `$1 \\times 1$ conv: ${CLASSES} class scores, $${sq(p.output)}$`,
      },
      ...p.encoder.map((s, i): DiagramNode => ({
        id: `e${i}`,
        x: ENC_X,
        y: Y(i),
        w: 3.6,
        h: 0.8,
        label: `${shape(s)}, ${s.channels} ch`,
        tone: 0,
      })),
      ...p.decoder.map((s, i): DiagramNode => ({
        id: `d${i}`,
        x: DEC_X,
        y: Y(i),
        w: 3.6,
        h: 0.8,
        label: `${shape(s)}, ${s.channels} ch`,
        tone: 1,
      })),
      {
        id: 'bot',
        x: (ENC_X + DEC_X) / 2,
        y: Y(depth),
        w: 3.6,
        h: 0.8,
        label: `${shape(p.bottleneck)}, ${p.bottleneck.channels} ch`,
        tone: 2,
      },
    ]
    const edges: DiagramEdge[] = [
      { from: 'in', to: 'e0' },
      { from: 'd0', to: 'out' },
      ...Array.from({ length: depth - 1 }, (_, i): DiagramEdge[] => [
        {
          from: `e${i}`,
          to: `e${i + 1}`,
          label: i === 0 ? 'max pool' : undefined,
          labelRotate: false,
          labelSide: 'right',
        },
        {
          from: `d${i + 1}`,
          to: `d${i}`,
          label: i === 0 ? 'up-conv' : undefined,
          labelRotate: false,
          labelSide: 'right',
        },
      ]).flat(),
      { from: `e${depth - 1}:s`, to: 'bot:w', via: [[ENC_X, Y(depth)]] },
      { from: 'bot:e', to: `d${depth - 1}:s`, via: [[DEC_X, Y(depth)]] },
      ...p.decoder.map((s, i): DiagramEdge => ({
        from: `e${i}`,
        to: `d${i}`,
        dashed: true,
        label: i === 0 ? 'copy and concatenate' : undefined,
        note: s.crop > 0 ? { text: `crop ${s.crop}` } : undefined,
      })),
    ]
    return { nodes, edges }
  }, [p, state.input])

  return (
    <Figure
      title="U-Net shapes by depth"
      state={state}
      caption="Each box is two 3 × 3 convolutions, labelled with the side of its input and output maps and its channel count. Set the depth, the width of the first stage, the padding and the input side. With valid padding every convolution trims 2 pixels, so the encoder maps are larger than the decoder maps they join and must be cropped (the chips on the skips), the output is smaller than the input, and only some input sides avoid an odd size before a pooling. Same padding keeps sizes, and any side divisible by 2 to the power of the depth works. Doubling the first-stage width multiplies the parameter count by about 4."
      readouts={
        p.ok ? (
          <>
            <Readout label="output side" value={`${p.output} (input ${state.input})`} />
            <Readout label="parameters" value={`${formatNumber(p.params / 1e6)} million`} />
            <Readout label="multiply-accumulates" value={`${formatNumber(p.multiplies / 1e9)} billion`} />
          </>
        ) : (
          <>
            <Readout label="does not trace" value={p.reason} />
            <Readout
              label="nearest input sides that work"
              value={valid ? valid.filter((v) => v !== null).join(' or ') || 'none within 64 pixels' : ''}
            />
          </>
        )
      }
    >
      {spec ? (
        <Diagram
          spec={spec}
          ariaLabel="U-Net: encoder stages with max pooling down to a bottleneck, decoder stages with up-convolution back up, and skip connections that crop and concatenate encoder maps into the decoder at each resolution"
        />
      ) : (
        <p className="p-6 text-sm text-muted-foreground">
          No network: {p.ok ? '' : p.reason}. Try one of the input sides listed below.
        </p>
      )}
    </Figure>
  )
}
