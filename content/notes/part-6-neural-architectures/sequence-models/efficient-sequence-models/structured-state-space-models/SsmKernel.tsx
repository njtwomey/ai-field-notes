import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

const L = 96
const FREQS = 160

type Cx = [number, number]
const add = (a: Cx, b: Cx): Cx => [a[0] + b[0], a[1] + b[1]]
const mul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
const div = (a: Cx, b: Cx): Cx => {
  const d = b[0] * b[0] + b[1] * b[1]
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}
const cexp = (a: Cx): Cx => [Math.exp(a[0]) * Math.cos(a[1]), Math.exp(a[0]) * Math.sin(a[1])]
const abs = (a: Cx) => Math.hypot(a[0], a[1])

type Disc = 'zoh' | 'bilinear'
type Mode = 'real' | 'complex'

/** Discretise dx/dt = λx + u (B = 1) with step Δ. Returns (Ā, B̄). */
function discretise(lambda: Cx, delta: number, disc: Disc): [Cx, Cx] {
  const dl: Cx = [delta * lambda[0], delta * lambda[1]]
  if (disc === 'zoh') {
    const a = cexp(dl)
    return [a, div(add(a, [-1, 0]), lambda)]
  }
  const den: Cx = [1 - dl[0] / 2, -dl[1] / 2]
  return [div([1 + dl[0] / 2, dl[1] / 2], den), div([delta, 0], den)]
}

// Test input: a square wave with seeded noise.
const U = (() => {
  const r = rng(7)
  return Array.from({ length: L }, (_, k) => (k % 40 < 20 ? 1 : -1) + 0.3 * r.normal())
})()
const STEPS = Array.from({ length: L }, (_, k) => k)

/**
 * A diagonal SSM with one real eigenvalue, or one complex-conjugate pair, discretised with step Δ. Its output is
 * computed twice, by the recurrence and by convolving with the kernel K, and the frequency response is |H(e^{iθ})|.
 */
function simulate(mode: Mode, alpha: number, omega: number, delta: number, disc: Disc) {
  const lambda: Cx = [-alpha, mode === 'complex' ? omega : 0]
  const [a, b] = discretise(lambda, delta, disc)
  // A conjugate pair with C = (1, 1) contributes twice the real part of one mode; a real mode contributes itself.
  const scale = mode === 'complex' ? 2 : 1
  const kernel: number[] = []
  let p: Cx = b
  for (let k = 0; k < L; k++) {
    kernel.push(scale * p[0])
    p = mul(p, a)
  }
  const recurrent: number[] = []
  let h: Cx = [0, 0]
  for (let k = 0; k < L; k++) {
    h = add(mul(a, h), [b[0] * U[k], b[1] * U[k]])
    recurrent.push(scale * h[0])
  }
  const conv = STEPS.map((k) => {
    let s = 0
    for (let j = 0; j <= k; j++) s += kernel[j] * U[k - j]
    return s
  })
  const theta = Array.from({ length: FREQS }, (_, i) => (Math.PI * (i + 0.5)) / FREQS)
  const response = theta.map((t) => {
    // H(z) = C B̄ / (1 - Ā z^{-1}) at z = e^{iθ}, summed over the conjugate pair.
    const one = div(b, add([1, 0], mul([-1, 0], mul(a, [Math.cos(t), -Math.sin(t)]))))
    if (mode === 'real') return abs(one)
    const aBar: Cx = [a[0], -a[1]]
    const other = div([b[0], -b[1]], add([1, 0], mul([-1, 0], mul(aBar, [Math.cos(t), -Math.sin(t)]))))
    return abs(add(one, other))
  })
  const error = Math.max(...recurrent.map((v, k) => Math.abs(v - conv[k])))
  return { a, kernel, recurrent, conv, theta, response, error }
}

const MODES = [
  { value: 'real', label: 'real λ' },
  { value: 'complex', label: 'complex pair' },
] as const
const DISCS = [
  { value: 'zoh', label: 'zero-order hold' },
  { value: 'bilinear', label: 'bilinear' },
] as const

export function SsmKernel() {
  const [mode, setMode] = useState<Mode>('complex')
  const [disc, setDisc] = useState<Disc>('zoh')
  const alpha = useParam(0.5, { min: 0.05, max: 2, step: 0.05 })
  const omega = useParam(2, { min: 0, max: 4, step: 0.05 })
  const [logDelta, setLogDelta] = useState(-1)
  const delta = 10 ** logDelta

  const sim = useMemo(
    () => simulate(mode, alpha.value, omega.value, delta, disc),
    [mode, alpha.value, omega.value, delta, disc],
  )
  const radius = abs(sim.a)
  const memory = radius < 1 ? -1 / Math.log(radius) : Infinity

  const eigSeries = useMemo<XYSeries[]>(
    () => [
      {
        name: 'eigenvalue λ',
        type: 'scatter',
        x: mode === 'complex' ? [-alpha.value, -alpha.value] : [-alpha.value],
        y: mode === 'complex' ? [omega.value, -omega.value] : [0],
        slot: 0,
      },
    ],
    [mode, alpha.value, omega.value],
  )
  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: [-alpha.value, mode === 'complex' ? omega.value : 0],
        label: 'λ',
        onDrag: ([x, y]) => {
          alpha.set(-x)
          if (mode === 'complex') omega.set(Math.abs(y))
        },
      },
    ],
    [alpha, omega, mode],
  )
  const kernelSeries = useMemo<XYSeries[]>(
    () => [{ name: 'kernel K_k', type: 'line', x: STEPS, y: sim.kernel, slot: 0 }],
    [sim.kernel],
  )
  const responseSeries = useMemo<XYSeries[]>(
    () => [{ name: '|H|', type: 'line', x: sim.theta, y: sim.response, slot: 0 }],
    [sim.theta, sim.response],
  )
  const outputSeries = useMemo<XYSeries[]>(
    () => [
      { name: 'input u', type: 'line', x: STEPS, y: U, muted: true },
      { name: 'recurrence', type: 'line', x: STEPS, y: sim.recurrent, slot: 0 },
      { name: 'convolution', type: 'scatter', x: STEPS, y: sim.conv, slot: 1 },
    ],
    [sim.recurrent, sim.conv],
  )

  return (
    <Interactive
      title="The SSM convolution kernel"
      caption="A diagonal state-space layer with one real eigenvalue λ = −α, or a conjugate pair λ = −α ± iω, discretised with step Δ. Drag λ in the complex plane or use the sliders. The kernel K_k = C Ā^k B̄ decays at the rate |Ā| per step and oscillates at ωΔ radians per step. A larger Δ shortens the memory and moves the resonance to a higher frequency. The output computed by the recurrence (line) and by convolution with K (dots) agree to rounding error."
      controls={
        <>
          <ParamChoice label="eigenvalues" value={mode} onChange={setMode} options={MODES} />
          <ParamChoice label="discretisation" value={disc} onChange={setDisc} options={DISCS} />
          <ParamSlider label="decay α" param={alpha} />
          {mode === 'complex' && <ParamSlider label="frequency ω" param={omega} />}
          <ParamSlider
            label="step Δ"
            value={logDelta}
            onChange={setLogDelta}
            min={-2}
            max={0.5}
            step={0.05}
            format={(v) => formatNumber(10 ** v)}
          />
        </>
      }
      readout={
        <>
          <Readout label="|Ā| (pole radius)" value={formatNumber(radius)} />
          <Readout label="memory −1 / ln|Ā|" value={Number.isFinite(memory) ? `${formatNumber(memory)} steps` : '∞'} />
          <Readout label="max |recurrent − conv|" value={sim.error.toExponential(1)} />
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <XYChart
          series={eigSeries}
          handles={handles}
          xLabel="Re λ"
          yLabel="Im λ"
          xRange={[-2.2, 0.2]}
          yRange={[-4.4, 4.4]}
          height={260}
        />
        <XYChart series={kernelSeries} xLabel="lag k" yLabel="K_k" xRange={[0, L - 1]} height={260} />
        <XYChart
          series={responseSeries}
          xLabel="frequency θ (radians per step)"
          yLabel="|H(e^{iθ})|"
          xRange={[0, Math.PI]}
          yLog
          height={260}
        />
        <XYChart series={outputSeries} xLabel="step k" yLabel="y_k" xRange={[0, L - 1]} height={260} />
      </div>
    </Interactive>
  )
}
