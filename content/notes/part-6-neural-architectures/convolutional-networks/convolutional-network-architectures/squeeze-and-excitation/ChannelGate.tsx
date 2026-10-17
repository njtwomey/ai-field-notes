import { useMemo } from 'react'
import {
  Bars,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const N = 12
const CELLS = Array.from({ length: N }, (_, i) => i)

/** Four fixed channel patterns whose means at strength 1 are 1, 4, 0.5 and 0 (the worked example's squeeze). */
const PATTERNS: number[][][] = (() => {
  const stripes = CELLS.map(() => CELLS.map((j) => (j % 2 === 0 ? 2 : 0)))
  const bumpRaw = CELLS.map((i) => CELLS.map((j) => Math.exp(-((i - 5.5) ** 2 + (j - 5.5) ** 2) / 18)))
  const bumpMean = bumpRaw.flat().reduce((a, b) => a + b, 0) / (N * N)
  const bump = bumpRaw.map((row) => row.map((v) => (4 * v) / bumpMean))
  const diagonal = CELLS.map((i) => CELLS.map((j) => ((i + j) % 4 < 2 ? 1 : 0)))
  const checker = CELLS.map((i) => CELLS.map((j) => ((i + j) % 2 === 0 ? 1 : -1)))
  return [stripes, bump, diagonal, checker]
})()

/** Excitation weights of the worked example: W₁ is 2 × 4 (reduction r = 2), W₂ is 4 × 2, no biases. */
const W1 = [
  [1, 0.5, -1, 0],
  [-1, 0, 2, 1],
]
const W2 = [
  [0.8, 0],
  [-0.4, 1],
  [0.6, -1],
  [0, 0.5],
]
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))
const NAMES = ['channel 1', 'channel 2', 'channel 3', 'channel 4']

/** A squeeze-and-excitation gate on four channels: average-pool, bottleneck MLP, sigmoid, rescale. */
export function ChannelGate() {
  const state = useFigureState({
    a1: slider(0, 3, 1, { step: 0.05, label: 'strength of channel 1' }),
    a2: slider(0, 3, 1, { step: 0.05, label: 'strength of channel 2' }),
    a3: slider(0, 3, 1, { step: 0.05, label: 'strength of channel 3' }),
    a4: slider(0, 3, 1, { step: 0.05, label: 'strength of channel 4' }),
    channels: int(256, { min: 4, max: 4096, label: 'channels C (for the cost)', suggestions: [64, 256, 1024, 2048] }),
    reduction: int(16, { min: 1, max: 64, label: 'reduction ratio r', suggestions: [4, 8, 16, 32] }),
  })
  const { a1, a2, a3, a4 } = state

  const gate = useMemo(() => {
    const strengths = [a1, a2, a3, a4]
    const maps = PATTERNS.map((p, c) => p.map((row) => row.map((v) => strengths[c] * v)))
    const z = maps.map((m) => m.flat().reduce((a, b) => a + b, 0) / (N * N))
    const hidden = W1.map((row) =>
      Math.max(
        0,
        row.reduce((s, w, c) => s + w * z[c], 0),
      ),
    )
    const s = W2.map((row) => sigmoid(row.reduce((acc, w, h) => acc + w * hidden[h], 0)))
    const gated = maps.map((m, c) => m.map((row) => row.map((v) => s[c] * v)))
    const peak = Math.max(1e-9, ...maps.map((m) => Math.max(...m.flat().map(Math.abs))))
    return { z, hidden, s, gated, range: [-peak, peak] as [number, number] }
  }, [a1, a2, a3, a4])

  const C = state.channels
  const hiddenWidth = Math.max(1, Math.floor(C / state.reduction))
  const seParams = 2 * C * hiddenWidth + hiddenWidth + C
  const convParams = 9 * C * C

  const mx = useAxis({ label: 'column', range: [-0.5, N - 0.5], nice: false })
  const my = useAxis({ label: 'row', range: [-0.5, N - 0.5], nice: false, inverse: true, equal: mx })
  const bx = useAxis({ label: 'channel', categories: NAMES })
  const zy = useAxis({ label: 'squeeze z_c', range: [0, undefined] })
  const sy = useAxis({ label: 'gate s_c', range: [0, 1] })

  return (
    <Figure
      title="A squeeze-and-excitation gate on four channels"
      state={state}
      defaultSize="L"
      caption="Top: four channels of a feature map after the gate, s_c · u_c, on one signed colour scale whose ends are the largest input value, so a channel fades as its gate falls. Bottom: the squeeze z_c (the channel's mean over its 12 × 12 grid) and the gate s_c = σ((W₂ ReLU(W₁z))_c), with the worked example's fixed weights and r = 2. The defaults reproduce the worked example: channel 2 has the largest mean but gets the smallest gate, 0.27, because the gate of a channel depends on the whole vector z. Raise the strength of channel 3 and watch channel 2 recover and channel 4 change although its own input did not move. The readouts count the weights an SE block adds at C channels and reduction r, beside a 3 × 3 convolution at the same width."
      readouts={{
        'this gate': (
          <>
            {gate.s.map((v, c) => (
              <Readout key={c} label={`s${c + 1}`} value={formatNumber(v)} />
            ))}
          </>
        ),
        'cost at C, r': (
          <>
            <Readout label="hidden width C/r" value={hiddenWidth} />
            <Readout label="SE parameters" value={formatNumber(seParams)} />
            <Readout label="3 × 3 convolution weights, 9C²" value={formatNumber(convParams)} />
            <Readout label="ratio" value={`${formatNumber((100 * seParams) / convParams)}%`} />
          </>
        ),
      }}
    >
      <Plots cols={4} scale={0.55}>
        {gate.gated.map((m, c) => (
          <Plot key={c} x={mx} y={my} title={`${NAMES[c]} × ${formatNumber(gate.s[c])}`}>
            <Raster
              x={CELLS}
              y={CELLS}
              z={m}
              scale="diverging"
              range={gate.range}
              colorBar={c === 3}
              valueLabel="gated value"
            />
          </Plot>
        ))}
      </Plots>
      <Plots cols={2} scale={0.45}>
        <Plot x={bx} y={zy} title="squeeze: channel means">
          <Bars name="squeeze z" x={[0, 1, 2, 3]} y={gate.z} width={0.6} slot={0} />
        </Plot>
        <Plot x={bx} y={sy} title="excitation: gates">
          <Bars name="gate s" x={[0, 1, 2, 3]} y={gate.s} width={0.6} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
