/**
 * The spreading ring of Lynden-Bell and Pringle (1974), Pringle (1981, eq. 2.13): a ring of mass m at radius R₀ in a
 * Keplerian disc of constant kinematic viscosity ν, with zero torque at the centre. Units: radius x = R/R₀, time
 * τ = 12νt/R₀², surface density in m/(πR₀²), mass in m, angular momentum in m√(GMR₀).
 *
 *   Σ(x, τ) = τ⁻¹ x^{-1/4} exp(−(1 + x²)/τ) I_{1/4}(2x/τ).
 *
 * The engine ships I₀ and I₁ only, so I_{1/4} comes from Poisson's integral (DLMF 10.32.2) evaluated with the
 * engine's Gauss–Legendre rule; the integrals of Σ use the engine's adaptive Gauss–Kronrod.
 */
import { toFlat } from 'aifn-compute/foundation/tensor'
import { gaussLegendre, integrate } from 'aifn-compute/numerics/quadrature'
import { gamma, regularisedGammaP } from 'aifn-compute/numerics/special'

const ORDER = 0.25

// Poisson's integral with t = 1 − u: z^{-ν} e^{-z} I_ν(z) = 2^{-ν} / (√π Γ(ν + ½)) ∫₀² [u(2 − u)]^{ν−½} e^{−zu} du.
// For ν = ¼ the endpoint factors u^{-1/4} and (2 − u)^{-1/4} vanish under u = w⁴ on [0, 1] and 2 − u = w⁴ on [1, 2],
// leaving the smooth weight 4w²(2 − w⁴)^{-1/4} on w ∈ [0, 1] for both halves.
const RULE = gaussLegendre(48, [0, 1])
const W4 = toFlat(RULE.nodes).map((w) => w ** 4)
const KERNEL = toFlat(RULE.nodes).map((w, i) => toFlat(RULE.weights)[i] * 4 * w * w * (2 - W4[i]) ** -0.25)
const PREFACTOR = 2 ** -ORDER / (Math.sqrt(Math.PI) * gamma(ORDER + 0.5))

/** z^{-1/4} e^{-z} I_{1/4}(z) for z ≥ 0: finite at z = 0 (where it is 2^{-1/4}/Γ(5/4)) and free of overflow. */
export function scaledBesselQuarter(z: number): number {
  let s = 0
  for (let i = 0; i < KERNEL.length; i++) s += KERNEL[i] * (Math.exp(-z * W4[i]) + Math.exp(-z * (2 - W4[i])))
  return PREFACTOR * s
}

/**
 * Σ(x, τ) in units m/(πR₀²). With I_{1/4}(z) = z^{1/4} e^{z} · scaledBesselQuarter(z) and z = 2x/τ, the exponentials
 * combine into exp(−(1 − x)²/τ) and the powers of x cancel: Σ = 2^{1/4} τ^{-5/4} exp(−(1 − x)²/τ) · scaled(2x/τ).
 */
export function surfaceDensity(x: number, tau: number): number {
  return 2 ** 0.25 * tau ** -1.25 * Math.exp(-((1 - x) ** 2) / tau) * scaledBesselQuarter((2 * x) / tau)
}

/** dM/dx = 2πR Σ dR/dx in units m/R₀: 2x Σ. */
export const massDensity = (x: number, tau: number) => 2 * x * surfaceDensity(x, tau)

/** dJ/dx, with the Keplerian specific angular momentum √(GMR): 2x^{3/2} Σ in units m√(GMR₀)/R₀. */
export const angularMomentumDensity = (x: number, tau: number) => 2 * x ** 1.5 * surfaceDensity(x, tau)

/** The rate of mass flow onto the centre, dM/dτ = (πR₀²/4) Σ(0, τ) in units m (from Ṁ = 3πνΣ at R → 0). */
export const accretionRate = (tau: number) => surfaceDensity(0, tau) / 4

/** The outer edge of the integrals: Σ falls as a Gaussian of width √τ beyond the ring, e^{−144} by here. */
const edge = (tau: number) => 1 + 12 * Math.sqrt(tau)

/** ∫ f over [a, b] by Gauss–Kronrod, split at the initial radius when it lies inside (a narrow ring at small τ). */
function over(f: (x: number) => number, a: number, b: number): number {
  if (!(b > a)) return 0
  return integrate(f, a, b, { points: a < 1 && 1 < b ? [1] : undefined, atol: 1e-12, rtol: 1e-10 }).value
}

export type Budget = {
  tau: number
  /** Disc mass inside and outside the radius `split`, by quadrature of 2xΣ. */
  massInside: number
  massOutside: number
  /** Mass that has flowed onto the centre, by quadrature of the inflow rate over [0, τ]. */
  accreted: number
  /** Disc mass from the closed form m·P(1/4, 1/τ), for comparison. */
  discClosedForm: number
  /** Angular momentum inside and outside `split`, by quadrature of 2x^{3/2}Σ. */
  angularInside: number
  angularOutside: number
}

/** The mass and angular-momentum budget at time τ, split at radius x = `split`. */
export function budget(tau: number, split = 1): Budget {
  const top = Math.max(edge(tau), split)
  const m = (x: number) => massDensity(x, tau)
  const j = (x: number) => angularMomentumDensity(x, tau)
  return {
    tau,
    massInside: over(m, 0, split),
    massOutside: over(m, split, top),
    accreted: over(accretionRate, 0, tau),
    discClosedForm: regularisedGammaP(ORDER, 1 / tau),
    angularInside: over(j, 0, split),
    angularOutside: over(j, split, top),
  }
}
