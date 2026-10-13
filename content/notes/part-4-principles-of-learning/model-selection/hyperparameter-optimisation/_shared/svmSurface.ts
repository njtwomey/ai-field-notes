import { linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { real, space } from 'aifn-compute/foundation/space'
import { svmTuningSurface, type TuningSurface } from 'aifn-methods/data/objectives'

/**
 * The RBF-SVM tuning surface of the engine: repeated 5-fold cross-validated error on two iris species over
 * (log10 C, log10 γ) in [-2, 3] × [-3, 2], interpolated bilinearly from a 21 by 21 table. Shared by the search figures
 * of this sub-chapter.
 */
export const SVM_SURFACE: TuningSurface = svmTuningSurface()

/** The search space in the surface's coordinates: log10 C and log10 γ, each searched uniformly. */
export const SVM_SPACE = space({
  logC: real(-2, 3, { label: 'log₁₀ C' }),
  logGamma: real(-3, 2, { label: 'log₁₀ γ' }),
})

/** The lower and upper corners of the box, as [log10 C, log10 γ]. */
export const SVM_LO = [-2, -3] as const
export const SVM_HI = [3, 2] as const

/** The surface's error at a point of the box, without evaluation noise. */
export const svmError = (logC: number, logGamma: number): number => SVM_SURFACE.mean(tensor([logC, logGamma]))

/** The lowest error in the table, the value a search is measured against. */
export const SVM_BEST: number = SVM_SURFACE.minimumValue

/**
 * The surface on an m by m grid for a `Raster`: axes in the box's coordinates and z[j][i] the error at
 * (cAxis[i], gammaAxis[j]).
 */
export function svmRaster(m = 81) {
  const cAxis = toFlat(linspace(SVM_LO[0], SVM_HI[0], m))
  const gammaAxis = toFlat(linspace(SVM_LO[1], SVM_HI[1], m))
  const z = gammaAxis.map((g) => cAxis.map((c) => svmError(c, g)))
  return { cAxis, gammaAxis, z }
}

/** The running minimum of a sequence of errors: entry t is the best of the first t + 1. */
export function bestSoFar(errors: readonly number[]): number[] {
  const out: number[] = []
  errors.forEach((e, t) => out.push(t === 0 ? e : Math.min(out[t - 1], e)))
  return out
}
