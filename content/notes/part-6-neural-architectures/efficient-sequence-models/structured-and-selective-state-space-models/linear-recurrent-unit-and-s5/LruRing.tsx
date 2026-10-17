import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { complex, realPart, type Tensor, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { lruConvolve, lruEigenvalues, lruKernel, lruRingInit, lruScan, type LruReadout } from 'aifn-compute/nn/sequence'

const T = 256
const STEPS = Array.from({ length: T }, (_, k) => k)
const CIRCLE = Array.from({ length: 181 }, (_, i) => (2 * Math.PI * i) / 180)

/** A burst of seeded noise for 32 steps, then silence, so the output shows how long the layer remembers. */
const U = (() => {
  const r = stream(5)
  return STEPS.map((k) => (k < 32 ? normal(r) : 0))
})()
const U_TENSOR = tensor(U)
const IMPULSE = tensor(STEPS.map((k) => (k === 0 ? 1 : 0)))

const ring = (radius: number) => ({
  x: CIRCLE.map((a) => radius * Math.cos(a)),
  y: CIRCLE.map((a) => radius * Math.sin(a)),
})

/**
 * The linear recurrent unit's eigenvalues, drawn uniformly on the ring r_min ≤ |λ| ≤ r_max with phases up to a
 * maximum, and what they do: impulse responses of each state, and the output on a burst of noise computed both by
 * the parallel scan and by convolution with the layer's kernel.
 */
export function LruRing() {
  const state = useFigureState({
    rMin: float(0.9, { min: 0, max: 0.999, step: 0.001, label: 'r_min', onChart: true }),
    rMax: float(0.99, { min: 0.01, max: 0.9999, step: 0.0001, label: 'r_max', onChart: true }),
    phase: float(2, { min: 0.02, max: 2, step: 0.01, label: 'max phase (× π)', suggestions: [0.1, 1, 2] }),
    n: int(16, { min: 1, max: 64, label: 'states n', suggestions: [4, 16, 64] }),
    normalise: setting(true, 'normalise inputs by γ = √(1 − |λ|²)'),
    seed: int(0, { min: 0, max: 99, label: 'seed', suggestions: [0, 1, 2] }),
  })
  const { rMin, rMax, phase, n, normalise, seed } = state
  const lo = Math.min(rMin, rMax)
  const hi = Math.max(rMin, rMax)

  const model = useMemo(() => {
    const root = stream(seed)
    const params = lruRingInit(n, { rMin: lo, rMax: hi, maxPhase: phase * Math.PI }, child(root, 0))
    const rc = child(root, 1)
    const scale = 1 / Math.sqrt(n)
    const cRe = Array.from({ length: n }, () => scale * normal(rc))
    const cIm = Array.from({ length: n }, () => scale * normal(rc))
    const ones = tensor(Array(n).fill(1))
    const zeros = tensor(Array(n).fill(0))
    const readout: LruReadout = { B: complex(ones, zeros), C: complex(tensor(cRe), tensor(cIm)), normalise }
    const eigen = lruEigenvalues(params)
    const ev = {
      re: toFlat(eigen.re as Tensor),
      im: toFlat(eigen.im as Tensor),
      modulus: toFlat(eigen.modulus as Tensor),
    }
    const scanY = toFlat(lruScan(U_TENSOR, params, readout).y as Tensor)
    const conv = toFlat(lruConvolve(U_TENSOR, params, readout) as Tensor)
    const kernel = toFlat(lruKernel(params, readout, T) as Tensor)
    // Impulse response of each state, Re(γ_j λ_j^k), from the scan of a unit impulse.
    const impulse = lruScan(IMPULSE, params, { ...readout, C: complex(ones, zeros) })
    const stateRows = toRows(realPart(impulse.states as Tensor))
    const responses = Array.from({ length: n }, (_, j) => stateRows.map((row) => row[j]))
    const gammaLog = toFlat(params.gammaLog)
    const halfLives = ev.modulus.map((r) => Math.log(0.5) / Math.log(r)).sort((a, b) => a - b)
    const gains = ev.modulus.map((r, j) => (normalise ? Math.exp(2 * gammaLog[j]) : 1) / (1 - r * r))
    const error = Math.max(...scanY.map((v, k) => Math.abs(v - conv[k])))
    return { ev, scan: scanY, conv, kernel, responses, halfLives, gain: gains.reduce((a, g) => a + g, 0) / n, error }
  }, [seed, n, lo, hi, phase, normalise])

  const unit = useMemo(() => ring(1), [])
  const inner = useMemo(() => ring(lo), [lo])
  const outer = useMemo(() => ring(hi), [hi])
  const median = model.halfLives[Math.floor(model.halfLives.length / 2)]

  const reAxis = useAxis({ label: 'Re λ', range: [-1.1, 1.1] })
  const imAxis = useAxis({ label: 'Im λ', range: [-1.1, 1.1], equal: reAxis })
  const kAxis = useAxis({ label: 'step k', range: [0, T - 1] })
  const respAxis = useAxis({ label: 'Re(γ λ^k)', hold: 'union', key: `${normalise}` })
  const yAxis = useAxis({ label: 'y_k', hold: 'union', key: `${normalise}-${seed}` })
  const kerAxis = useAxis({ label: 'kernel K_k', hold: 'union', key: `${normalise}-${seed}` })

  return (
    <Figure
      title="Eigenvalues on a ring set the memory of a linear recurrence"
      state={state}
      caption="The linear recurrent unit draws n eigenvalues uniformly on the ring r_min ≤ |λ| ≤ r_max of the complex plane, with phases from 0 to the maximum. Drag the lines labelled r_min and r_max along the real axis. Each state's impulse response Re(γλ^k) decays by |λ| per step and rotates by the phase. The input is 32 steps of noise followed by silence: the output keeps ringing for as long as the slowest eigenvalue remembers. The kernel K_k sums the responses with the output weights C. The scan (line) and the convolution with that kernel (dots) give the same output. Switch off the normalisation to see the state gain 1/(1 − |λ|²) blow up as r_max approaches 1."
      readouts={
        <>
          <Readout label="median half-life ln ½ / ln|λ|" value={`${formatNumber(median)} steps`} />
          <Readout label="mean white-noise state gain" value={formatNumber(model.gain)} />
          <Readout label="max |scan − convolution|" value={model.error.toExponential(1)} />
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[1, 1]}>
        <Plot x={reAxis} y={imAxis}>
          <Curve name="unit circle" x={unit.x} y={unit.y} muted />
          <Curve name="ring" x={inner.x} y={inner.y} muted dashed />
          <Curve name="ring" x={outer.x} y={outer.y} muted dashed />
          <Points name="eigenvalue λ" x={model.ev.re} y={model.ev.im} slot={0} />
          <Handle {...state.handle('rMin', { label: 'r_min' })} />
          <Handle {...state.handle('rMax', { label: 'r_max' })} />
        </Plot>
        <Plot x={kAxis} y={respAxis}>
          {model.responses.map((r, j) => (
            <Curve key={j} id={`resp-${j}`} name="state impulse response" x={STEPS} y={r} slot={0} thin silent />
          ))}
        </Plot>
        <Plot x={kAxis} y={kerAxis}>
          <Curve name="kernel K_k = Re Σ C γ λ^k B" x={STEPS} y={model.kernel} slot={3} />
        </Plot>
        <Plot x={kAxis} y={yAxis}>
          <Curve name="input u" x={STEPS} y={U} muted />
          <Curve name="scan" x={STEPS} y={model.scan} slot={1} />
          <Points name="convolution" x={STEPS} y={model.conv} slot={2} size={3} />
        </Plot>
      </Plots>
    </Figure>
  )
}
