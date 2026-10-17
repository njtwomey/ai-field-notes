/**
 * Point-vortex physics for the figure in this note: Kirchhoff's equations for N point vortices in the unbounded plane,
 * passive tracers carried by the vortices' velocity field, and the four conserved quantities (Saffman, 1992, §7.1;
 * Aref, 1983).
 *
 * Circulations are measured as strengths κ = Γ/2π, so a vortex of strength κ at distance r induces a speed κ/r. The
 * state is one flat vector: the vortex positions [x₁, y₁, …, x_N, y_N], then the tracer positions.
 */
import type { Tensor } from 'aifn-compute/foundation/tensor'

/** A vortex: its position and its strength κ = Γ/2π (positive counter-clockwise). */
export type Vortex = { x: number; y: number; kappa: number }

/** The entries of a state vector as one contiguous array (the solver's vectors are contiguous). */
const flat = (s: Tensor): ArrayLike<number> => {
  const d = s.data as Float64Array
  return s.offset === 0 && d.length === s.shape[0] ? d : d.subarray(s.offset, s.offset + s.shape[0])
}

/**
 * The right-hand side of Kirchhoff's equations with passive tracers:
 * dx_k/dt = −Σ_{j≠k} κ_j (y_k − y_j)/r²_kj and dy_k/dt = Σ_{j≠k} κ_j (x_k − x_j)/r²_kj for the vortices; the same
 * sums over all vortices for each tracer, with r² replaced by r² + a² (a = `tracerCore`) so that a tracer passing very
 * close to a vortex stays on a fixed-step integrator's grid instead of being flung off. The vortices feel no tracers.
 *
 * @param kappa The strengths κ_j of the N vortices.
 * @param tracerCore The smoothing length a for the tracers' velocity, in the figure's length unit; 0 for none.
 * @returns f(t, s) for the state s = [vortex positions, tracer positions].
 */
export function kirchhoff(kappa: readonly number[], tracerCore: number) {
  const n = kappa.length
  const a2 = tracerCore * tracerCore
  return (_t: number, s: Tensor): Float64Array => {
    const z = flat(s)
    const m = z.length / 2
    const out = new Float64Array(z.length)
    for (let k = 0; k < m; k++) {
      const xk = z[2 * k]
      const yk = z[2 * k + 1]
      const soft = k < n ? 0 : a2
      let u = 0
      let v = 0
      for (let j = 0; j < n; j++) {
        if (j === k) continue
        const dx = xk - z[2 * j]
        const dy = yk - z[2 * j + 1]
        const w = kappa[j] / (dx * dx + dy * dy + soft)
        u -= w * dy
        v += w * dx
      }
      out[2 * k] = u
      out[2 * k + 1] = v
    }
    return out
  }
}

/** The conserved quantities of N point vortices, in the units of κ = Γ/2π. */
export type Invariants = {
  /** The Hamiltonian H = −Σ_{j<k} κ_j κ_k ln r_jk (the interaction energy per unit density, divided by 2π). */
  energy: number
  /** The linear impulse Q = Σ κ_k x_k. */
  qx: number
  /** The linear impulse P = Σ κ_k y_k. */
  py: number
  /** The angular impulse I = Σ κ_k (x_k² + y_k²). */
  angular: number
}

/**
 * The invariants of the vortices in a state vector (its first 2N entries).
 *
 * @param z The state: x₁, y₁, …, x_N, y_N first (tracers after, ignored).
 * @param kappa The strengths κ_j.
 */
export function invariants(z: ArrayLike<number>, kappa: readonly number[]): Invariants {
  const n = kappa.length
  let energy = 0
  let qx = 0
  let py = 0
  let angular = 0
  for (let k = 0; k < n; k++) {
    const x = z[2 * k]
    const y = z[2 * k + 1]
    qx += kappa[k] * x
    py += kappa[k] * y
    angular += kappa[k] * (x * x + y * y)
    for (let j = k + 1; j < n; j++) {
      const r = Math.hypot(x - z[2 * j], y - z[2 * j + 1])
      energy -= kappa[j] * kappa[k] * Math.log(r)
    }
  }
  return { energy, qx, py, angular }
}

/**
 * Tracers on a regular grid over a box, skipping points within `clear` of a vortex (where the speed is too large to
 * follow with a fixed step).
 *
 * @param box The box [x0, x1] × [y0, y1].
 * @param columns Grid points across.
 * @param rows Grid points up.
 * @param vortices The vortex positions to keep clear of.
 * @param clear The clearance radius.
 * @returns The tracer positions as one flat array [x, y, x, y, …].
 */
export function tracerGrid(
  box: { x: [number, number]; y: [number, number] },
  columns: number,
  rows: number,
  vortices: readonly Vortex[],
  clear: number,
): number[] {
  const out: number[] = []
  for (let i = 0; i < columns; i++) {
    for (let j = 0; j < rows; j++) {
      const x = box.x[0] + ((i + 0.5) / columns) * (box.x[1] - box.x[0])
      const y = box.y[0] + ((j + 0.5) / rows) * (box.y[1] - box.y[0])
      if (vortices.some((v) => Math.hypot(x - v.x, y - v.y) < clear)) continue
      out.push(x, y)
    }
  }
  return out
}
