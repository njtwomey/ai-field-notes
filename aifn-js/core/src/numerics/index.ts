/**
 * `aifn/numerics`: numerical methods, as scipy.special, scipy.linalg, numpy.polynomial, scipy.integrate, scipy.optimize root finding,
 * scipy.interpolate and scipy.spatial. Children: special, linalg, polynomial, quadrature, roots, geometry,
 * interpolate.
 */

export { erf, logGamma, normalCdf, sigmoid, softplus } from './special'
export { cholesky, solve, eigh, svd, qr, lu, expm } from './linalg'
export { polynomialRoots } from './polynomial'
export { integrate } from './quadrature'
export { findRoot, newtonRoot } from './roots'
export { cubicSpline } from './interpolate'
