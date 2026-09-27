export type Kernel = 'gaussian' | 'epanechnikov' | 'tophat'

export const KERNEL_OPTIONS = [
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'epanechnikov', label: 'Epanechnikov' },
  { value: 'tophat', label: 'box' },
] as const satisfies readonly { value: Kernel; label: string }[]

/**
 * Kernels scaled to unit variance, so that one bandwidth h means the same spread for each: the Epanechnikov kernel on
 * [−√5, √5] and the box on [−√3, √3].
 */
export function kernel(kind: Kernel, u: number): number {
  if (kind === 'gaussian') return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI)
  if (kind === 'epanechnikov') return Math.abs(u) < Math.sqrt(5) ? (3 / (4 * Math.sqrt(5))) * (1 - (u * u) / 5) : 0
  return Math.abs(u) < Math.sqrt(3) ? 1 / (2 * Math.sqrt(3)) : 0
}

/** f̂(x) = (1/nh) Σ K((x − xᵢ)/h) on each grid point. */
export function kde(data: number[], grid: number[], h: number, kind: Kernel): number[] {
  return grid.map((x) => data.reduce((s, xi) => s + kernel(kind, (x - xi) / h), 0) / (data.length * h))
}

/** Silverman's rule of thumb: 0.9 min(σ̂, IQR/1.34) n^(−1/5). */
export const silverman = (sd: number, iqr: number, n: number) => 0.9 * Math.min(sd, iqr / 1.34) * n ** -0.2
