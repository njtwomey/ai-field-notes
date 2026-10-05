import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Plots,
  Plot,
  Bars,
  Curve,
  Points,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { child, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { quantisationError, quantisationParams, type QuantisationParams } from 'aifn-compute/nn/quantise'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '—')
const BITS = [2, 3, 4, 5, 6, 7, 8]

function weightMatrix(kind: string, seed: number): Tensor {
  const s = stream(`quantise-weights-${seed}`)
  const z = toFlat(normal(child(s, 'z'), 0, 1, { shape: [16, 64] }) as Tensor)
  const chi = toFlat(normal(child(s, 'chi'), 0, 1, { shape: [16, 64, 3] }) as Tensor)
  const scale = toFlat(uniform(child(s, 'rows'), -1.2, 1.2, { shape: [16] }) as Tensor).map((v) => 10 ** v)
  const out = Float64Array.from(z, (v, i) => {
    const t = kind === 'heavy' ? v / Math.sqrt((chi[3 * i] ** 2 + chi[3 * i + 1] ** 2 + chi[3 * i + 2] ** 2) / 3) : v
    const r = Math.floor(i / 64)
    return 0.1 * t * (kind === 'rows' ? scale[r] : 1)
  })
  return fromData(out, [16, 64])
}

function histogram(values: ArrayLike<number>, lo: number, hi: number, bins: number) {
  const counts = new Array<number>(bins).fill(0)
  const w = (hi - lo) / bins
  for (let i = 0; i < values.length; i++) {
    const k = Math.floor((values[i] - lo) / w)
    if (k >= 0 && k < bins) counts[k]++
  }
  return { x: counts.map((_, k) => lo + (k + 0.5) * w), y: counts, width: w }
}

const VARIANTS = [
  { key: 'tensor', label: 'per tensor, nearest', axis: undefined, rounding: 'nearest' },
  { key: 'channel', label: 'per channel, nearest', axis: 0, rounding: 'nearest' },
  { key: 'stochastic', label: 'per tensor, stochastic', axis: undefined, rounding: 'stochastic' },
] as const

export function BitsErrorExplorer() {
  const [kind, setKind] = useState('rows')
  const [seed, setSeed] = useState(0)
  const [bits, setBits] = useState(4)
  const [scheme, setScheme] = useState<'symmetric' | 'affine'>('symmetric')
  const [granularity, setGranularity] = useState<'tensor' | 'channel'>('tensor')

  const W = useMemo(() => weightMatrix(kind, seed), [kind, seed])
  const w = useMemo(() => toFlat(W), [W])

  const params: QuantisationParams = useMemo(
    () =>
      quantisationParams(W, {
        bits,
        scheme,
        ...(granularity === 'channel' ? { axis: 0 } : {}),
      }),
    [W, bits, scheme, granularity],
  )

  const err = useMemo(() => quantisationError(W, params), [W, params])

  const curves = useMemo(
    () =>
      VARIANTS.map((v) =>
        BITS.map((b) => {
          const p = quantisationParams(W, {
            bits: b,
            scheme,
            ...(v.axis === undefined ? {} : { axis: v.axis }),
          })
          return quantisationError(
            W,
            p,
            v.rounding === 'stochastic' ? { rounding: 'stochastic', stream: stream('quantise-sr') } : {},
          ).sqnr
        }),
      ),
    [W, scheme],
  )

  const span = Math.max(...w.map(Math.abs)) * 1.05
  const histAll = useMemo(() => histogram(w, -span, span, 80), [w, span])
  const histRow0 = useMemo(() => histogram(w.slice(0, 64), -span, span, 80), [w, span])

  const levels = useMemo(() => {
    const s = params.scale[0]
    const z = params.zeroPoint[0]
    const nLevels = params.qmax - params.qmin + 1
    if (nLevels > 64) return []
    return Array.from({ length: nLevels }, (_, k) => s * (params.qmin + k - z))
  }, [params])

  const xv = useAxis({ label: 'weight value', range: [-span, span], key: `${kind}-${seed}` })
  const yc = useAxis({ label: 'count', hold: 'union', key: `${kind}-${seed}` })
  const xb = useAxis({ label: 'bits b', integer: true, range: [1.5, 8.5] })
  const ys = useAxis({ label: 'SQNR (dB)', hold: 'union', key: `${kind}-${seed}-${scheme}` })

  return (
    <Figure
      title="Quantisation error and SQNR scaling across bit widths"
      purpose="Every additional bit roughly halves the quantisation step size s and adds ~6 dB of SQNR; when weight variances differ across channels, per-channel scaling drastically outperforms single-tensor scaling."
      defaultSize="XL"
      controls={
        <>
          <ControlGroup title="1 · Weight distribution">
            <Select
              label="weight structure"
              value={kind}
              onChange={setKind}
              options={[
                { value: 'gaussian', label: 'Gaussian' },
                { value: 'heavy', label: 'heavy-tailed (Student t₃)' },
                { value: 'rows', label: 'channels on different scales' },
              ]}
            />
            <NumberSelector
              label="seed"
              value={seed}
              onChange={setSeed}
              min={0}
              max={20}
              step={1}
              suggestions={[0, 1, 2, 5]}
            />
          </ControlGroup>
          <ControlGroup title="2 · Quantiser settings">
            <NumberSelector
              label="bit width b"
              value={bits}
              onChange={setBits}
              min={2}
              max={8}
              step={1}
              suggestions={[2, 3, 4, 6, 8]}
            />
            <Select
              label="scheme"
              value={scheme}
              onChange={(v) => setScheme(v as 'symmetric' | 'affine')}
              options={[
                { value: 'symmetric', label: 'symmetric (zero-point = 0)' },
                { value: 'affine', label: 'affine (asymmetric min/max)' },
              ]}
            />
            <Select
              label="granularity"
              value={granularity}
              onChange={(v) => setGranularity(v as 'tensor' | 'channel')}
              options={[
                { value: 'tensor', label: 'per tensor (single scale)' },
                { value: 'channel', label: 'per output channel (row)' },
              ]}
            />
          </ControlGroup>
        </>
      }
      readouts={
        <>
          <Readout label="integer levels 2ᵇ" value={2 ** bits} />
          <Readout label="scale s (channel 0)" value={fmt(params.scale[0])} />
          <Readout label="zero point z (channel 0)" value={params.zeroPoint[0]} />
          <Readout label="mean squared error" value={fmt(err.mse)} />
          <Readout label="SQNR" value={`${fmt(err.sqnr)} dB`} />
          <Readout label="clipped fraction" value={`${(100 * err.clipped).toFixed(2)}%`} />
        </>
      }
      caption="Left: Histogram of all weights (blue) and output channel 0 (red). Discrete vertical tick marks depict the active quantisation levels for channel 0. Notice that when channels operate on different scales, a per-tensor scale compresses small-channel weights into just 1 or 2 levels, causing severe SQNR degradation. Right: SQNR (in decibels) versus bit width b showing the theoretical ~6.02 dB/bit slope across per-tensor, per-channel, and stochastic rounding schemes."
    >
      <Plots cols={2}>
        <Plot x={xv} y={yc} title="Weight histogram and grid levels">
          <Bars name="all weights" x={histAll.x} y={histAll.y} width={histAll.width} slot={0} />
          <Bars name="channel 0" x={histRow0.x} y={histRow0.y} width={histRow0.width} slot={1} />
          {levels.map((lvl, idx) => (
            <Curve
              key={idx}
              name="quantisation levels"
              x={[lvl, lvl]}
              y={[0, Math.max(...histAll.y) * 0.9]}
              slot={2}
              width={1}
              silent
            />
          ))}
        </Plot>
        <Plot x={xb} y={ys} title="SQNR (dB) vs bit width b">
          {VARIANTS.map((v, idx) => (
            <Curve key={v.key} name={v.label} x={BITS} y={curves[idx]} slot={idx} showPoints />
          ))}
          <Points name="selected bit width" x={[bits]} y={[err.sqnr]} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
