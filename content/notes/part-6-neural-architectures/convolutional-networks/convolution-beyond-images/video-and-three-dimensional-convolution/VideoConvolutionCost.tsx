import { useMemo } from 'react'
import { Bars, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { convolutionCost } from 'aifn-compute/nn/functional'
import { FixedHeight } from '../_shared/FixedHeight'

const LABELS = ['2-D, each frame', '3-D', '(2+1)-D', 'channel-separated']
const AT = [0, 1, 2, 3]

/**
 * Parameters and multiply-accumulates of four ways to build one spatiotemporal convolution layer on a clip of T
 * frames of S × S pixels, from C to C' channels, with a k × k spatial and k_t temporal kernel and same padding.
 */
export function VideoConvolutionCost() {
  const state = useFigureState({
    frames: int(16, { ge: 1, le: 64, suggestions: [8, 16, 32, 64], label: 'frames T' }),
    size: int(56, { ge: 7, le: 224, suggestions: [28, 56, 112, 224], label: 'frame size S (pixels)' }),
    kernel: int(3, { ge: 1, le: 7, suggestions: [1, 3, 5, 7], label: 'spatial kernel k' }),
    temporal: int(3, { ge: 1, le: 7, suggestions: [1, 3, 5], label: 'temporal kernel k_t' }),
    cin: int(64, { ge: 1, le: 1024, suggestions: [3, 64, 128, 256], label: 'input channels C' }),
    cout: int(128, { ge: 1, le: 1024, suggestions: [64, 128, 256, 512], label: "output channels C'" }),
  })
  const { frames, size, kernel: k, temporal: kt, cin, cout } = state

  const costs = useMemo(() => {
    const input = [frames, size, size]
    const sp = Math.floor((k - 1) / 2)
    const tp = Math.floor((kt - 1) / 2)
    const perFrame = convolutionCost({
      inChannels: cin,
      outChannels: cout,
      kernel: [1, k, k],
      padding: [0, sp, sp],
      input,
    })
    const full = convolutionCost({
      inChannels: cin,
      outChannels: cout,
      kernel: [kt, k, k],
      padding: [tp, sp, sp],
      input,
    })
    // Tran et al. (2018): M intermediate channels give the (2+1)-D block about the 3-D layer's parameter count.
    const M = Math.max(1, Math.floor((kt * k * k * cin * cout) / (k * k * cin + kt * cout)))
    const spatial = convolutionCost({ inChannels: cin, outChannels: M, kernel: [1, k, k], padding: [0, sp, sp], input })
    const temporal = convolutionCost({
      inChannels: M,
      outChannels: cout,
      kernel: [kt, 1, 1],
      padding: [tp, 0, 0],
      input,
    })
    const depthwise = convolutionCost({
      kind: 'depthwise',
      inChannels: cin,
      kernel: [kt, k, k],
      padding: [tp, sp, sp],
      input,
    })
    const pointwise = convolutionCost({ inChannels: cin, outChannels: cout, kernel: [1, 1, 1], input })
    const rows = [
      { params: perFrame.params, macs: perFrame.multiplies },
      { params: full.params, macs: full.multiplies },
      { params: spatial.params + temporal.params, macs: spatial.multiplies + temporal.multiplies },
      { params: depthwise.params + pointwise.params, macs: depthwise.multiplies + pointwise.multiplies },
    ]
    return { rows, M }
  }, [frames, size, k, kt, cin, cout])

  const x = useAxis({ label: 'layer', categories: LABELS })
  const yParams = useAxis({ label: 'parameters (thousands)', range: [0, undefined], hold: 'union' })
  const yMacs = useAxis({ label: 'multiply-accumulates (billions)', range: [0, undefined], hold: 'union' })
  const [perFrame, full, factored, separated] = costs.rows
  return (
    <Figure
      title="What a spatiotemporal layer costs"
      state={state}
      caption="Parameters and multiply-accumulates per clip of one convolution layer from C to C' channels on T frames of S × S pixels, with same padding, counted from each layer's geometry. 2-D, each frame: a 1 × k × k kernel applied to every frame. 3-D: a k_t × k × k kernel. (2+1)-D: a 1 × k × k convolution to M channels, then a k_t × 1 × 1 convolution, with M chosen so the pair has about as many parameters as the 3-D layer. Channel-separated: a depthwise k_t × k × k convolution, one kernel per channel, then a 1 × 1 × 1 convolution that mixes channels. At the defaults (a C3D-sized layer from 64 to 128 channels on 16 frames of 56 × 56) the 3-D layer costs k_t = 3 times the per-frame layer; the channel-separated layer costs about a seventh of the per-frame layer. Every cost except the parameter counts grows in proportion to T S²."
      readouts={
        <>
          <Readout label="3-D ÷ 2-D per frame" value={`${formatNumber(full.macs / perFrame.macs)} × the compute`} />
          <Readout label="intermediate channels M" value={String(costs.M)} />
          <Readout
            label="(2+1)-D"
            value={`${formatNumber(factored.params / 1e3)} k parameters, ${formatNumber(factored.macs / 1e9)} G multiply-accumulates`}
          />
          <Readout
            label="3-D"
            value={`${formatNumber(full.params / 1e3)} k parameters, ${formatNumber(full.macs / 1e9)} G multiply-accumulates`}
          />
          <Readout
            label="channel-separated"
            value={`${formatNumber(separated.params / 1e3)} k parameters, ${formatNumber(separated.macs / 1e9)} G multiply-accumulates`}
          />
        </>
      }
    >
      <FixedHeight height={240}>
        <div className="grid gap-4 md:grid-cols-2">
          <Plot x={x} y={yParams} title="parameters">
            <Bars name="parameters" x={AT} y={costs.rows.map((r) => r.params / 1e3)} slot={0} />
          </Plot>
          <Plot x={x} y={yMacs} title="multiply-accumulates per clip">
            <Bars name="multiply-accumulates" x={AT} y={costs.rows.map((r) => r.macs / 1e9)} slot={1} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
