/**
 * Kepler's equation M = E − e sin E and the three anomalies of an elliptical orbit. The residual and its derivative
 * are written here; Newton's method itself is `newtonRoot` from `aifn-compute/numerics/roots`.
 */
import { newtonRoot, type ScalarWithDerivative } from 'aifn-compute/numerics/roots'
import { run, trace } from 'aifn-compute/foundation/trace'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** f(E) = E − e sin E − M with f′(E) = 1 − e cos E, for eccentricity e in [0, 1) and mean anomaly M (radians). */
export const keplerResidual =
  (e: number, M: number): ScalarWithDerivative =>
  (E) => ({ value: E - e * Math.sin(E) - M, derivative: 1 - e * Math.cos(E) })

/** g(E) = E − e sin E, the mean anomaly as a function of the eccentric anomaly. */
export const meanFromEccentric = (e: number, E: number) => E - e * Math.sin(E)

/** The most Newton steps any run takes. */
export const MAX_NEWTON = 50

/** Newton's iterates E₀, E₁, … on Kepler's equation from E₀, until the step falls below 10⁻¹² or MAX_NEWTON steps. */
export function newtonIterates(e: number, M: number, E0: number, steps = MAX_NEWTON): number[] {
  const tr = trace(newtonRoot(keplerResidual(e, M)), { x0: E0 }, steps, {
    keep: 'none',
    stopOnNonFinite: true,
    record: { x: (s) => s.x },
  })
  return toFlat(tr.series.x)
}

/** The number of Newton steps to converge from E₀, or `cap` if the run has not converged by then. */
export function stepsToConverge(e: number, M: number, E0: number, cap = 100): number {
  const s = run(newtonRoot(keplerResidual(e, M)), { x0: E0 }, cap)
  return s.converged ? s.t : cap
}

/** The eccentric anomaly E for mean anomaly M, by Newton's method from E₀ = π (which converges for every e < 1). */
export function eccentricAnomaly(e: number, M: number): { E: number; steps: number } {
  const s = run(newtonRoot(keplerResidual(e, M)), { x0: Math.PI }, 100)
  return { E: s.x, steps: s.t }
}

/** The true anomaly ν from the eccentric anomaly: tan(ν/2) = √((1 + e)/(1 − e)) tan(E/2), on the branch of E. */
export function trueFromEccentric(e: number, E: number): number {
  const nu = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E / 2), Math.sqrt(1 - e) * Math.cos(E / 2))
  // atan2 lands in (−2π, 2π]; shift it onto E's turn so ν and E advance together.
  return nu + 2 * Math.PI * Math.round((E - nu) / (2 * Math.PI))
}
