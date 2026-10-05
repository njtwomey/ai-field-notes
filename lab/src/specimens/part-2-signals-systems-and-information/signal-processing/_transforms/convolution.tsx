/**
 * "Convolution and the convolution theorem": convolution in time against multiplication of DFTs, with a kernel whose
 * taps are dragged; circular against linear convolution as the zero-padding grows.
 */
import { useMemo, useState } from 'react'
import { convolve } from 'aifn-compute/foundation/convolution'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import { fromData, toComplexFlat, toFlat, type ComplexNumber, type Tensor } from 'aifn-compute/foundation/tensor'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, useFigureState } from 'aifn-render/state'
import { Area, Curve, Handle, Plot, Plots, Points, Readout, Segments, useAxis } from 'aifn-render/viz'
import { fmt, range, stems } from './common'

const N = 48
const L = 9

const INPUTS = [
  { value: 'pulse', label: 'rectangular pulse' },
  { value: 'step', label: 'noisy step' },
  { value: 'impulses', label: 'three impulses' },
  { value: 'chirp', label: 'rising chirp' },
] as const

const KERNELS: Record<string, { label: string; taps: number[] }> = {
  box: { label: 'moving average', taps: [0, 0, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 0, 0] },
  gauss: {
    label: 'Gaussian',
    taps: range(L).map((i) => Math.exp(-0.5 * ((i - 4) / 1.4) ** 2) / (1.4 * Math.sqrt(2 * Math.PI))),
  },
  decay: {
    label: 'exponential decay (causal)',
    taps: range(L).map((i) => (i >= 4 ? (0.4 * 0.6 ** (i - 4)) / (1 - 0.6 ** 5) : 0)),
  },
  diff: { label: 'difference (edge detector)', taps: [0, 0, 0, 1, 0, -1, 0, 0, 0] },
}

function input(kind: string): number[] {
  return range(N).map((i) => {
    if (kind === 'pulse') return i >= 14 && i < 30 ? 1 : 0
    if (kind === 'step') return (i >= 20 ? 1 : 0) + 0.15 * Math.sin(i * 2.3) * Math.cos(i * 0.7)
    if (kind === 'impulses') return i === 8 ? 1 : i === 22 ? -0.6 : i === 30 ? 0.8 : 0
    return Math.cos(0.004 * Math.PI * i * i)
  })
}

const abs = (z: ComplexNumber) => Math.hypot(z.re, z.im)

export function ConvolutionTheoremFigure() {
  const state = useFigureState({
    signals: row('1 · signals', {
      x: choice(INPUTS, 'pulse', { label: 'input x' }),
      kernel: choice(
        Object.entries(KERNELS).map(([value, k]) => ({ value, label: k.label })),
        'decay',
        { label: 'kernel h (then drag its taps)' },
      ),
    }),
  })
  const { x: kind, kernel } = state.signals
  const [edit, setEdit] = useState<{ kernel: string; taps: number[] } | null>(null)
  const taps = edit?.kernel === kernel ? edit.taps : KERNELS[kernel].taps
  const x = useMemo(() => input(kind), [kind])
  const M = N + L - 1
  const out = useMemo(() => {
    const direct = toFlat(convolve(x, taps) as Tensor)
    // The theorem: the DFT of the linear convolution is the product of the DFTs, at M = N + L − 1 points.
    const X = toComplexFlat(fft(x, { n: M }) as Tensor)
    const H = toComplexFlat(fft(taps, { n: M }) as Tensor)
    const Y = new Float64Array(2 * M)
    for (let k = 0; k < M; k++) {
      Y[2 * k] = X[k].re * H[k].re - X[k].im * H[k].im
      Y[2 * k + 1] = X[k].re * H[k].im + X[k].im * H[k].re
    }
    const viaDft = toComplexFlat(ifft(fromData(Y, [M], 'complex128')) as Tensor).map((z) => z.re)
    const half = range(Math.floor(M / 2) + 1)
    return {
      direct,
      viaDft,
      gap: Math.max(...direct.map((v, i) => Math.abs(v - viaDft[i]))),
      f: half.map((k) => k / M),
      X: half.map((k) => abs(X[k])),
      H: half.map((k) => abs(H[k])),
      Y: half.map((k) => abs(X[k]) * abs(H[k])),
    }
  }, [x, taps, M])
  const n = useAxis({ label: 'sample n', range: [-1, M] })
  const v = useAxis({ label: 'value', hold: 'union', key: kind })
  const k = useAxis({ label: 'tap', range: [-0.5, L - 0.5] })
  const hv = useAxis({ label: 'h[k]', range: [-1.2, 1.2] })
  const f = useAxis({ label: 'frequency (cycles/sample)', range: [0, 0.5] })
  const mag = useAxis({ label: '|DFT|', hold: 'union', key: `${kind}${kernel}` })
  const hmag = useAxis({ label: '|H|', hold: 'union', key: kernel })
  return (
    <Figure
      title="Convolution in time is multiplication in frequency"
      purpose="Sliding the flipped kernel along the input and summing products gives the same output as multiplying the two DFTs bin by bin and inverting, once both are zero-padded to the full output length."
      defaultSize="L"
      state={state}
      readouts={
        <>
          <Readout label="output length N + L − 1" value={M} />
          <Readout label="max |direct − via DFT|" value={out.gap.toExponential(1)} />
          <Readout
            label="Σ h"
            value={fmt(
              taps.reduce((a, b) => a + b, 0),
              3,
            )}
          />
        </>
      }
      caption="Drag any tap of h up or down (its lower-left panel): the output and the spectra update together. The output computed by aifn convolve (stems) and by fft · fft → ifft (second markers) agree to rounding. On the right, |Y| = |X|·|H|: a smoothing kernel is a low-pass, the difference kernel a high-pass that zeroes DC."
    >
      <Plots rows={2} cols={2} widths={[2, 1]}>
        <Plot x={n} y={v}>
          <Curve name="x" x={range(N)} y={x} muted />
          <Segments name="y = x ∗ h (direct)" segments={stems(range(M), out.direct)} slot={0} />
          <Points name="y = x ∗ h (direct)" x={range(M)} y={out.direct} slot={0} size={4} />
          <Points name="IDFT(X·H)" x={range(M)} y={out.viaDft} slot={1} size={7} shape={2} />
        </Plot>
        <Plot x={f} y={mag}>
          <Curve name="|X|" x={out.f} y={out.X} muted />
          <Curve name="|Y| = |X||H|" x={out.f} y={out.Y} slot={0} />
        </Plot>
        <Plot x={k} y={hv}>
          <Segments name="h" segments={stems(range(L), taps)} slot={2} />
          {taps.map((t, i) => (
            <Handle
              key={i}
              kind="point"
              at={[i, t]}
              onDrag={([, y]) => {
                const next = [...taps]
                next[i] = Math.max(-1.2, Math.min(1.2, y))
                setEdit({ kernel, taps: next })
              }}
            />
          ))}
        </Plot>
        <Plot x={f} y={hmag}>
          <Curve name="|H|" x={out.f} y={out.H} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}

export function CircularConvolutionFigure() {
  const state = useFigureState({
    pad: slider(0, 12, 0, { label: 'zero-padding P (samples added to each input)', step: 1 }),
  })
  const P = state.pad
  const n0 = 24
  const len = 10
  const x = useMemo(() => range(n0).map((i) => (i < 16 ? 1 - i / 20 : 0.2)), [])
  const h = useMemo(() => range(len).map((i) => 0.35 * 0.75 ** i), [])
  const linear = useMemo(() => toFlat(convolve(x, h) as Tensor), [x, h])
  const size = n0 + P
  const circular = useMemo(() => {
    const X = toComplexFlat(fft(x, { n: size }) as Tensor)
    const H = toComplexFlat(fft(h, { n: size }) as Tensor)
    const Y = new Float64Array(2 * size)
    for (let k = 0; k < size; k++) {
      Y[2 * k] = X[k].re * H[k].re - X[k].im * H[k].im
      Y[2 * k + 1] = X[k].re * H[k].im + X[k].im * H[k].re
    }
    return toComplexFlat(ifft(fromData(Y, [size], 'complex128')) as Tensor).map((z) => z.re)
  }, [x, h, size])
  const wrapped = Math.max(0, len - 1 - P)
  const err = Math.max(...circular.map((v, i) => Math.abs(v - (linear[i] ?? 0))))
  const n = useAxis({ label: 'sample n', range: [-1, n0 + len] })
  const v = useAxis({ label: 'output', range: [-0.1, 1.6] })
  return (
    <Figure
      title="Circular against linear convolution"
      purpose="Multiplying two M-point DFTs convolves circularly: the tail beyond M wraps onto the start. Padding both inputs to M ≥ N + L − 1 leaves no tail to wrap, and the result equals the linear convolution."
      defaultSize="M"
      state={state}
      readouts={
        <>
          <Readout label="DFT length M = N + P" value={`${size} (need ≥ ${n0 + len - 1})`} />
          <Readout label="samples corrupted by wrap-around" value={wrapped} />
          <Readout label="max |circular − linear|" value={err.toExponential(1)} />
        </>
      }
      caption="x has N = 24 samples and h has L = 10 (a decaying exponential). Step P up: the shaded wrap-around region at the start shrinks by one sample per step, and at P = 9 (M = 33) the circular result (stems) lies on the linear one (curve) everywhere."
    >
      <Plot x={n} y={v}>
        {wrapped > 0 && (
          <Area
            name="wrapped tail"
            x={[-0.5, wrapped - 0.5]}
            y={[1.6, 1.6]}
            base={-0.1}
            tone="destructive"
            opacity={0.12}
            line={false}
          />
        )}
        <Curve name="linear x ∗ h" x={range(linear.length)} y={linear} emphasis />
        <Segments name={`circular, M = ${size}`} segments={stems(range(size), circular)} slot={0} />
        <Points name={`circular, M = ${size}`} x={range(size)} y={circular} slot={0} size={5} />
      </Plot>
    </Figure>
  )
}
