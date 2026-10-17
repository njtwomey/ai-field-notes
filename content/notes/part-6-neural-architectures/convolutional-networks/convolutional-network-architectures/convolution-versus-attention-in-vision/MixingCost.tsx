import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { layerCost } from 'aifn-compute/nn/functional'

const SIDES = Array.from({ length: 63 }, (_, i) => i + 2)
const LAYERS = Array.from({ length: 12 }, (_, i) => i + 1)

/** Multiply-adds of one spatial-mixing layer on an n × n grid of d-dimensional tokens. */
function costs(n: number, d: number, k: number, w: number) {
  const N = n * n
  const conv = layerCost({ kind: 'conv', out: d, kernel: k, padding: 'same', bias: false }, [d, n, n]).multiplies
  // Query, key, value and output projections are 1×1 convolutions: 4d² per token.
  const projections = 4 * layerCost({ kind: 'conv', out: d, kernel: 1, bias: false }, [d, n, n]).multiplies
  const window = Math.min(w, n) ** 2
  return {
    conv,
    global: projections + 2 * N * N * d,
    windowed: projections + 2 * N * window * d,
  }
}

/** Cost and reach of a k × k convolution against global and windowed self-attention on a grid of tokens. */
export function MixingCost() {
  const state = useFigureState({
    dim: int(384, { min: 32, max: 2048, step: 32, label: 'channels d', suggestions: [96, 384, 768, 1024] }),
    kernel: choice([3, 5, 7], 3, { label: 'convolution kernel k' }),
    window: choice([7, 12, 14], 7, { label: 'attention window w' }),
    side: int(14, { min: 2, max: 64, label: 'grid side n', onChart: true }),
  })
  const { dim: d, kernel: k, window: w } = state
  const n = Math.min(64, Math.max(2, Math.round(state.side)))

  const curves = useMemo(() => {
    const rows = SIDES.map((s) => costs(s, d, k, w))
    return {
      conv: rows.map((r) => r.conv / 1e6),
      global: rows.map((r) => r.global / 1e6),
      windowed: rows.map((r) => r.windowed / 1e6),
    }
  }, [d, k, w])
  const here = costs(n, d, k, w)
  const reach = useMemo(
    () => ({
      conv: LAYERS.map((l) => Math.min(n, 1 + (k - 1) * l)),
      global: LAYERS.map(() => n),
    }),
    [n, k],
  )
  const crossover = Math.sqrt(((k * k - 4) * d) / 2)

  const cx = useAxis({ label: 'grid side n (tokens per side)', range: [2, 64] })
  const cy = useAxis({ label: 'multiply-adds per layer (millions)', log: true })
  const rx = useAxis({ label: 'layer', range: [1, 12], integer: true })
  const ry = useAxis({ label: 'receptive field (tokens per side)', range: [0, undefined] })
  return (
    <Figure
      title="Convolution and attention as spatial mixing"
      state={state}
      defaultSize="L"
      caption={`Left: multiply-adds of one mixing layer on an n × n grid of d-channel tokens, for one image, on a log scale: a k × k convolution (k²d²n²), global self-attention (4d²n² for the query, key, value and output projections plus 2n⁴d for the scores and the weighted sum) and attention restricted to w × w windows. Drag the vertical line to set n. Global attention is cheaper than a ${k} × ${k} convolution below n ≈ ${formatNumber(crossover)}, where its n⁴ term takes over; windowed attention stays linear in n². Right: how many tokens per side can influence one output after each layer at this n. A ${k} × ${k} stack grows by ${k - 1} per layer; global attention covers the grid from the first layer.`}
      readouts={{
        [`at n = ${n} (N = ${n * n} tokens)`]: (
          <>
            <Readout label={`${k}×${k} convolution`} value={`${formatNumber(here.conv / 1e6)} M`} />
            <Readout label="global attention" value={`${formatNumber(here.global / 1e6)} M`} />
            <Readout label={`${w}×${w} windows`} value={`${formatNumber(here.windowed / 1e6)} M`} />
          </>
        ),
        parameters: (
          <>
            <Readout label={`${k}×${k} convolution, k²d²`} value={formatNumber(k * k * d * d)} />
            <Readout label="attention projections, 4d²" value={formatNumber(4 * d * d)} />
          </>
        ),
      }}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={cx} y={cy} height={300} title="cost of one layer">
          <Curve name={`${k}×${k} convolution`} x={SIDES} y={curves.conv} slot={0} />
          <Curve name="global attention" x={SIDES} y={curves.global} slot={1} />
          <Curve name={`${w}×${w} window attention`} x={SIDES} y={curves.windowed} slot={2} dashed />
          <Handle
            kind="x"
            at={n}
            label={`n = ${n}`}
            onDrag={(v) => state.set('side', Math.min(64, Math.max(2, Math.round(v))))}
          />
        </Plot>
        <Plot x={rx} y={ry} height={300} title="reach after each layer">
          <Curve name={`${k}×${k} convolution`} x={LAYERS} y={reach.conv} slot={0} showPoints />
          <Curve name="global attention" x={LAYERS} y={reach.global} slot={1} />
        </Plot>
      </div>
    </Figure>
  )
}
