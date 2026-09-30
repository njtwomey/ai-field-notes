/**
 * `aifn/numerics/linalg`: dense linear algebra on `aifn/foundation/tensor` matrices.
 *
 * - Factorisations that report failure instead of returning NaN: `cholesky` (with jitter; factored once),
 *   `luFactor` (packed, for repeated solves) and `lu` (L, U, P unpacked), `qr`, `eigh` (descending eigenvalues,
 *   vectors as columns), `svd` (thin, one-sided Jacobi), and `eig` for the general (non-symmetric) real eigenproblem
 *   (`converged`; its Francis QR is also what `aifn/numerics/polynomial` finds roots with).
 * - Solves and functions: `solveTriangular`, `choleskySolve`, `luSolve`, `solve`, `inverse`, `det`, `logDet`,
 *   `choleskyLogDet`, `pinv`, `lstsq`, `kron`, `matrixTrace`, `normFrobenius`, `conditionNumber`, `expm` (Padé,
 *   scaling and squaring). Solvers throw `LinAlgError` for a singular system.
 * - Matrix equations: `lyapunov` (continuous Lyapunov or discrete Stein); the algebraic Riccati equations as traceable
 *   algorithms, `kleinmanIteration` and `riccatiMatrixSign` (CARE), `riccatiRecursion` and `riccatiDoubling` (DARE);
 *   the Toeplitz Yule–Walker system by `levinsonDurbin`.
 * - Distances between point sets: `pairwiseDistances` (Euclidean, squared, Manhattan, Chebyshev, Minkowski, cosine)
 *   and `squaredDistances`.
 * - For inner loops on row-major `Float64Array`s: `solveDense` (reports `singular` instead of throwing).
 * - Differentiable (primitives or compositions of them): `cholesky`'s L, `solveTriangular`, `choleskySolve`,
 *   `choleskyLogDet`, `luSolve` (in A through the factor, and in B), `solve`, `inverse`, `det`, `logDet`, `kron`,
 *   `matrixTrace`, `normFrobenius`. Derivatives of `solve`, `det` and `logDet` reuse the one LU factor.
 * - Closed forms on 2×2 tuples: `det2`, `apply2`, `inv2`, `eigh2`, `eig2`, `cholesky2`, `svd2`.
 */

export { LinAlgError } from './dense'
export { solveTriangular, type TriangularOptions } from './triangular'
export { cholesky, choleskyLogDet, choleskySolve, type Cholesky, type CholeskyOptions } from './cholesky'
export { det, inverse, logDet, lu, luFactor, luSolve, signDet, solve, type LU, type LuFactor } from './lu'
export { qr, type QR } from './qr'
export { eigh, type Eigh } from './eigh'
export { conditionNumber, lstsq, pinv, svd, type LeastSquares, type SVD } from './svd'
export { kron, matrixTrace, normFrobenius } from './products'
export { eig, type Eigen } from './eig'
export { expm, type MatrixExponential } from './expm'
export { pairwiseDistances, squaredDistances, type PairwiseMetric } from './distances'
export { solveDense, type DenseSolution } from './solveDense'
export { apply2, cholesky2, det2, eig2, eigh2, inv2, svd2, type Eig2, type Mat2, type Vec2 } from './small'
export { lyapunov, type LyapunovSolution } from './lyapunov'
export {
  kleinmanIteration,
  riccatiDoubling,
  riccatiMatrixSign,
  riccatiRecursion,
  type DoublingState,
  type RiccatiFailure,
  type RiccatiOptions,
  type RiccatiProblem,
  type RiccatiState,
  type SignState,
} from './riccati'
export { levinsonDurbin, type LevinsonDurbin } from './levinson'
