import { useMemo } from 'react'
import { Curve, Figure, float, Handle, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { readonlyData } from 'aifn-compute/foundation/tensor'
import { periodicGaussianField } from 'aifn-methods/data/synthetic'
import { evaluateSeries, type FourierSeries1d, heatSolution, periodicGrid } from 'aifn-methods/dynamics/pde'
import { applyModeMultiplier, fitPhysicsInformedMultiplier } from 'aifn-methods/neural/operator'
import { closePeriodic, FixedHeight } from '../_shared/figure-helpers'

/** u_t = D u_xx on the periodic interval [0, 1), output at time T. */
const D = 0.1
const T = 0.01
const K = 16
const MAX_TRAIN = 64
const MAX_VIRTUAL = 64
const TEST = 50
const PANEL = 250
/** Rough inputs: the prior N(0, σ²(−Δ + 25I)^(−1)), whose high modes carry visible energy. */
const PRIOR = { modes: K, alpha: 1 }
const ks = Array.from({ length: K }, (_, k) => k + 1)
const exact = ks.map((k) => Math.exp(-D * (2 * Math.PI * k) ** 2 * T))
const grid = Array.from(periodicGrid(128))

const draws = {
  train: periodicGaussianField(stream('pino/train'), MAX_TRAIN, PRIOR),
  noise: readonlyData(normals(stream('pino/noise'), [MAX_TRAIN, 2 * K])) as Float64Array,
  virtual: periodicGaussianField(stream('pino/virtual'), MAX_VIRTUAL, PRIOR),
  test: periodicGaussianField(stream('pino/test'), TEST, PRIOR),
}

/** The relative L2 error of a final-time multiplier on the test inputs, in coefficient space (Parseval). */
function testError(final: Float64Array) {
  let e = 0
  let q = 0
  for (const f of draws.test) {
    const p = applyModeMultiplier(final, f)
    const t = heatSolution(f, D, T)
    for (let k = 0; k < K; k++) {
      e += (p.cos[k] - t.cos[k]) ** 2 + (p.sin[k] - t.sin[k]) ** 2
      q += t.cos[k] ** 2 + t.sin[k] ** 2
    }
  }
  return Math.sqrt(e / q)
}

/**
 * A linear one-layer operator for the heat equation, one multiplier per Fourier mode and time level, fitted to solved
 * pairs whose inputs are smooth, with and without the equation's residual on unsolved rough inputs.
 */
export function PinoSpectrum() {
  const state = useFigureState({
    smooth: int(4, { ge: 1, le: K, label: 'highest frequency in the solved inputs' }),
    pairs: int(8, { ge: 0, le: MAX_TRAIN, suggestions: [0, 1, 8, 64], label: 'solved pairs' }),
    noise: slider(0, 0.05, 0.01, { step: 0.005, label: 'noise on the solutions σ' }),
    virtual: int(20, { ge: 1, le: MAX_VIRTUAL, suggestions: [1, 5, 20, 64], label: 'unsolved inputs' }),
    levels: int(10, { ge: 1, le: 200, suggestions: [2, 10, 50, 200], label: 'time levels J' }),
    weight: float(1, { ge: 0, le: 1e4, suggestions: [0.01, 1, 100], label: 'residual weight λ' }),
    instance: int(1, { ge: 1, le: TEST, label: 'test function' }),
  })
  const { smooth, pairs, noise, virtual, levels, weight, instance } = state

  const fits = useMemo(() => {
    const cut = (f: FourierSeries1d) => ({
      ...f,
      cos: f.cos.map((c, k) => (k < smooth ? c : 0)),
      sin: f.sin.map((c, k) => (k < smooth ? c : 0)),
    })
    const train = draws.train.slice(0, pairs).map((raw, i) => {
      const input = cut(raw)
      const out = heatSolution(input, D, T)
      const z = draws.noise.subarray(i * 2 * K, (i + 1) * 2 * K)
      return {
        input,
        output: {
          ...out,
          cos: out.cos.map((c, k) => c + noise * z[k]),
          sin: out.sin.map((c, k) => c + noise * z[K + k]),
        },
      }
    })
    const unsolved = draws.virtual.slice(0, virtual)
    const common = { virtual: unsolved, diffusivity: D, time: T, levels }
    const dataOnly = pairs > 0 ? fitPhysicsInformedMultiplier({ ...common, train, weight: 0 }).final : null
    const residualOnly = fitPhysicsInformedMultiplier({ ...common, train: [], weight: 1 }).final
    const both = fitPhysicsInformedMultiplier({ ...common, train, weight }).final
    return {
      dataOnly,
      residualOnly,
      both,
      errors: {
        dataOnly: dataOnly ? testError(dataOnly) : null,
        residualOnly: testError(residualOnly),
        both: testError(both),
      },
    }
  }, [smooth, pairs, noise, virtual, levels, weight])

  const view = useMemo(() => {
    const f = draws.test[instance - 1]
    const curve = (s: FourierSeries1d) => closePeriodic(grid, Array.from(evaluateSeries(s, grid)))
    return {
      input: curve(f),
      truth: curve(heatSolution(f, D, T)),
      dataOnly: fits.dataOnly ? curve(applyModeMultiplier(fits.dataOnly, f)) : null,
      both: curve(applyModeMultiplier(fits.both, f)),
    }
  }, [fits, instance])

  const kAxis = useAxis({ label: 'frequency k', range: [0.5, K + 0.5] })
  const mAxis = useAxis({ label: 'multiplier m_k(T)', range: [-0.1, 1.1] })
  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const uAxis = useAxis({ label: 'u', range: [-3, 3] })
  const percent = (e: number | null) => (e === null ? 'no pairs' : `${(100 * e).toFixed(1)}%`)

  return (
    <Figure
      title="What the residual adds"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption={`The heat equation u_t = ${D} u_xx on the periodic interval [0, 1), from an initial profile to the solution at T = ${T}. The model is a linear operator with one real multiplier m_k(t_j) per Fourier mode k and time level t_j = jT/J, the output of a one-layer linear Fourier operator on a space-time grid. The solved pairs have smooth inputs, with no energy above the chosen frequency, and solutions observed with noise σ on each Fourier coefficient. The residual (m_k(t_j) − m_k(t_{j−1}))/Δt + D(2πk)² m_k(t_j) is evaluated on unsolved inputs drawn from a rougher prior, which have energy at every frequency. Top: the fitted final-time multipliers against the exact exp(−D(2πk)²T) (dashed). Data alone cannot identify the multiplier at frequencies the solved inputs lack and leaves it at 0. The residual alone identifies every multiplier up to the time-discretisation error of J levels. Drag the vertical line to change the highest frequency present in the solved inputs. Bottom: one rough test profile (grey), its solution (dashed) and the two fitted operators' outputs.`}
      readouts={
        <>
          <Readout label="test error, data only" value={percent(fits.errors.dataOnly)} />
          <Readout label="test error, residual only" value={percent(fits.errors.residualOnly)} />
          <Readout label="test error, data and residual" value={percent(fits.errors.both)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4">
        <FixedHeight height={PANEL}>
          <Plot x={kAxis} y={mAxis} title="final-time multiplier per frequency">
            <Curve name="exact" x={ks} y={exact} emphasis dashed />
            {fits.dataOnly && <Points name="data only" x={ks} y={Array.from(fits.dataOnly)} slot={1} size={7} />}
            <Points name="residual only" x={ks} y={Array.from(fits.residualOnly)} slot={2} size={5} />
            <Points name="data and residual" x={ks} y={Array.from(fits.both)} slot={0} size={7} />
            <Handle
              kind="x"
              at={smooth + 0.5}
              label="solved inputs end"
              onDrag={(v) => state.set('smooth', Math.min(K, Math.max(1, Math.round(v - 0.5))))}
            />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xAxis} y={uAxis} title="one rough test function">
            <Curve name="initial profile" x={view.input.x} y={view.input.y} muted />
            <Curve name="true solution" x={view.truth.x} y={view.truth.y} emphasis dashed />
            {view.dataOnly && <Curve name="data only" x={view.dataOnly.x} y={view.dataOnly.y} slot={1} />}
            <Curve name="data and residual" x={view.both.x} y={view.both.y} slot={0} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
