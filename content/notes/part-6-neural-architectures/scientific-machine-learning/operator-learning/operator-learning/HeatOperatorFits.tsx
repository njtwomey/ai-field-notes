import { useMemo } from 'react'
import { Curve, Figure, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromData, matmul, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import { relativeL2Error } from 'aifn-compute/learning/metrics'
import { heatOperatorData } from 'aifn-methods/data/synthetic'
import { applyFourierMultiplier, fitDenseOperator, fitFourierMultiplier } from 'aifn-methods/neural/operator'
import { FixedHeight } from '../_shared/figure-helpers'

/** Grid points of the periodic grid on [0, 1). */
const n = 32
/** Diffusivity and output time of u_t = D u_xx. */
const D = 0.1
const T = 0.1
const TRAIN = 128
const TEST = 50
const PANEL = 250
/** The training-set sizes on the error curve. */
const sizes = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
const data = (t: Tensor) => readonlyData(t) as Float64Array

/** The pairs, drawn once: 128 for training and 50 for testing, with one fixed draw of output noise. */
const pairs = (() => {
  const d = heatOperatorData(stream('operator-learning/heat'), {
    count: TRAIN + TEST,
    grid: n,
    diffusivity: D,
    time: T,
    prior: { modes: 15 },
  })
  return {
    x: Array.from(d.x),
    a: data(d.inputs),
    u: data(d.outputs),
    z: data(normals(stream('operator-learning/noise'), [TRAIN * n])),
  }
})()
const rows = (v: Float64Array, from: number, to: number) => fromData(v.slice(from * n, to * n), [to - from, n])
const testInputs = rows(pairs.a, TRAIN, TRAIN + TEST)
const testOutputs = rows(pairs.u, TRAIN, TRAIN + TEST)

/** Both models fitted to the first m noisy pairs, with their test predictions and errors. */
function fitBoth(noisy: Float64Array, m: number) {
  const G = fitDenseOperator(rows(pairs.a, 0, m), rows(noisy, 0, m), 1e-3)
  const mult = fitFourierMultiplier(rows(pairs.a, 0, m), rows(noisy, 0, m))
  const dense = matmul(testInputs, G) as Tensor
  const fourier = applyFourierMultiplier(mult, testInputs)
  return {
    dense: data(dense),
    fourier: data(fourier),
    denseError: relativeL2Error(testOutputs, dense),
    fourierError: relativeL2Error(testOutputs, fourier),
  }
}

/**
 * Test error of two linear operators fitted by least squares to noisy solved pairs of the heat equation: a dense map on
 * grid values, and one multiplier per Fourier mode.
 */
export function HeatOperatorFits() {
  const state = useFigureState({
    noise: slider(0, 0.05, 0.01, { step: 0.005, label: 'noise on the solutions σ' }),
    N: int(8, { ge: 1, le: TRAIN, suggestions: [1, 4, 16, 64, 128], label: 'training pairs N' }),
    instance: int(1, { ge: 1, le: TEST, label: 'test function' }),
  })
  const { noise, N, instance } = state

  const noisy = useMemo(
    () => Float64Array.from(pairs.u.subarray(0, TRAIN * n), (v, i) => v + noise * pairs.z[i]),
    [noise],
  )
  const fits = useMemo(() => sizes.map((m) => fitBoth(noisy, m)), [noisy])
  const fit = useMemo(() => fits[sizes.indexOf(N)] ?? fitBoth(noisy, N), [fits, noisy, N])
  const row = (v: Float64Array) => Array.from(v.subarray((instance - 1) * n, instance * n))
  // Close each periodic curve by repeating its first value at x = 1.
  const closed = (v: number[]) => [...v, v[0]]
  const xs = [...pairs.x, 1]

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const uAxis = useAxis({ label: 'u', range: [-2.5, 2.5] })
  const nAxis = useAxis({ label: 'solved training pairs N', log: true, range: [1, TRAIN] })
  const eAxis = useAxis({ label: 'relative test error', log: true, range: [1e-3, 2] })
  const clamp = (e: number) => Math.min(Math.max(e, 1e-3), 2)
  const percent = (e: number) => `${(100 * e).toFixed(1)}%`

  return (
    <Figure
      title="The same data, two operator families"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption={`Solved pairs of the heat equation u_t = ${D} u_xx on the periodic interval [0, 1): initial profiles u₀ drawn from a Gaussian prior and the exact solutions at t = ${T}, observed with Gaussian noise of standard deviation σ on a grid of ${n} points. Two linear models are fitted by least squares to the first N pairs. The dense map is an unrestricted ${n} × ${n} matrix on grid values (${n * n} numbers, ridge-regularised). The Fourier multiplier multiplies each Fourier mode of the input by one learned complex number (${n / 2 + 1} numbers), so it can only represent operators that commute with shifts. Bottom: the relative L2 error on ${TEST} test pairs against N, at ${sizes.length} training-set sizes from 1 to ${TRAIN}. Drag the vertical line, or type N, to choose the training-set size; the top panel shows both models on one test function at that N, with its initial profile (grey) and true solution (dashed).`}
      readouts={
        <>
          <Readout label="dense map, test error" value={percent(fit.denseError)} />
          <Readout label="Fourier multiplier, test error" value={percent(fit.fourierError)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4">
        <FixedHeight height={PANEL}>
          <Plot x={xAxis} y={uAxis} title="one test function">
            <Curve name="initial profile u₀" x={xs} y={closed(row(data(testInputs)))} muted />
            <Curve name="true u(·, 0.1)" x={xs} y={closed(row(data(testOutputs)))} emphasis dashed />
            <Curve name="dense map" x={xs} y={closed(row(fit.dense))} slot={1} />
            <Curve name="Fourier multiplier" x={xs} y={closed(row(fit.fourier))} slot={0} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={nAxis} y={eAxis} title="test error against training-set size">
            <Curve name="dense map" x={sizes} y={fits.map((f) => clamp(f.denseError))} slot={1} />
            <Curve name="Fourier multiplier" x={sizes} y={fits.map((f) => clamp(f.fourierError))} slot={0} />
            <Handle
              kind="x"
              at={N}
              label="N"
              onDrag={(v) => state.set('N', Math.min(TRAIN, Math.max(1, Math.round(v))))}
            />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
