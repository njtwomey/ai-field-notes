import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { hippoOnline, hippoReconstruct, hippoWindow, type HippoMeasure } from 'aifn-compute/nn/sequence'

const T = 1000
const DT = 1 / T
const TIMES = Array.from({ length: T }, (_, k) => (k + 1) * DT)

type Signal = 'steps' | 'smooth' | 'walk'

const SIGNALS: Record<Signal, number[]> = {
  steps: TIMES.map((x) => (x >= 0.2 && x < 0.4 ? 1 : x >= 0.55 && x < 0.7 ? -0.5 : x >= 0.85 ? 0.8 : 0)),
  smooth: TIMES.map((x) => Math.sin(2 * Math.PI * 3 * x) + 0.5 * Math.sin(2 * Math.PI * 7 * x)),
  walk: (() => {
    const r = stream(11)
    let s = 0
    return TIMES.map(() => (s += 0.06 * normal(r)))
  })(),
}

const SIGNAL_OPTIONS = [
  { value: 'steps', label: 'steps' },
  { value: 'smooth', label: 'two sinusoids' },
  { value: 'walk', label: 'random walk' },
] as const
const MEASURES = [
  { value: 'legs', label: 'LegS (whole history)' },
  { value: 'legt', label: 'LegT (sliding window)' },
] as const

/**
 * The HiPPO state as a memory. An input is fed to the LegS or LegT system of order N, discretised by the bilinear
 * rule, and the state at the chosen time t is decoded back into a polynomial approximation of the history it
 * summarises: [0, t] for LegS, [t − θ, t] for LegT.
 */
export function HippoMemory() {
  const state = useFigureState({
    signal: choice<Signal>(SIGNAL_OPTIONS, 'steps', { label: 'input' }),
    measure: choice<HippoMeasure>(MEASURES, 'legs', { label: 'measure' }),
    order: int(16, { min: 1, max: 64, label: 'state size N', suggestions: [4, 8, 16, 32, 64] }),
    theta: float(0.3, {
      min: 0.05,
      max: 1,
      step: 0.05,
      label: 'window θ',
      when: (v) => v.measure === 'legt',
    }),
    t: float(0.9, { min: 0.05, max: 1, step: 0.001, label: 'time t', onChart: true }),
  })
  const { signal, measure, order, theta, t } = state
  const u = SIGNALS[signal]

  const states = useMemo(() => toRows(hippoOnline(u, DT, { measure, order, theta })), [u, measure, order, theta])

  const k = Math.min(T - 1, Math.max(0, Math.round(t / DT) - 1))
  const tk = (k + 1) * DT
  const h = states[k]
  const [a, b] = hippoWindow(measure, tk, theta)
  const view = useMemo(() => {
    const xs = Array.from({ length: 241 }, (_, i) => a + ((b - a) * i) / 240).filter((x) => x >= 0)
    const recon = toFlat(hippoReconstruct(h, measure, tk, theta, xs))
    // Error over the summarised window, against the input it saw (zero before time 0).
    const seen: number[] = []
    const inputs: number[] = []
    for (let j = 0; j <= k; j++) {
      const x = (j + 1) * DT
      if (x < a) continue
      seen.push(x)
      inputs.push(u[j])
    }
    const decoded = seen.length > 0 ? toFlat(hippoReconstruct(h, measure, tk, theta, seen)) : []
    const se = decoded.reduce((acc, r, i) => acc + (r - inputs[i]) ** 2, 0)
    const count = seen.length
    return { xs, recon, rmse: Math.sqrt(se / Math.max(count, 1)) }
  }, [h, measure, tk, theta, a, b, k, u])

  const coeffs = useMemo(() => Array.from(h), [h])
  const index = useMemo(() => coeffs.map((_, n) => n), [coeffs])

  const xAxis = useAxis({ label: 'time s', range: [0, 1] })
  const yAxis = useAxis({ label: 'u(s)', hold: 'union', key: signal })
  const nAxis = useAxis({ label: 'coefficient n', range: [-0.5, 63.5], integer: true })
  const cAxis = useAxis({ label: 'h_n(t)', hold: 'union', key: `${signal}-${measure}` })

  return (
    <Figure
      title="Reading the history back from a HiPPO state"
      state={state}
      caption="The input is fed one sample at a time (1000 samples on [0, 1]) to the HiPPO system of order N, discretised by the bilinear rule. Drag the line labelled t to choose when to stop. The top panel decodes the N numbers of the state at t into the polynomial Σ h_n g_n(s) over the history the state summarises: all of [0, t] for LegS, the last θ for LegT. The bottom panel shows the N coefficients. More coefficients resolve finer detail; with LegS the same N must cover a longer history as t grows, so detail far in the past blurs."
      readouts={
        <>
          <Readout label="summarised window" value={`[${formatNumber(Math.max(a, 0))}, ${formatNumber(b)}]`} />
          <Readout label="RMS reconstruction error" value={formatNumber(view.rmse)} />
          <Readout label="numbers stored" value={`${order}`} />
        </>
      }
    >
      <Plots rows={2} heights={[2, 1]}>
        <Plot x={xAxis} y={yAxis}>
          <Curve name="input u" x={TIMES} y={u} muted />
          <Curve name="decoded from h(t)" x={view.xs} y={view.recon} slot={0} />
          <Handle {...state.handle('t', { label: 't' })} />
        </Plot>
        <Plot x={nAxis} y={cAxis}>
          <Bars name="state h_n(t)" x={index} y={coeffs} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
