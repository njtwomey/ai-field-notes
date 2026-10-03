/** Toy data for the GP-LVM figures. */
import { rng } from '@/lib/math'
import { centre } from './gplvm'

export const LOOP_N = 40

/**
 * Forty points on a closed curve in four dimensions, t = 2πn/40:
 * y(t) = (sin t, 0.8 sin 2t, 0.6 cos t, 0.3 cos 3t) plus Gaussian noise of sd 0.03. The first two coordinates, which
 * carry most of the variance, trace a figure of eight; the third separates the two branches where they cross.
 */
export function loopData(): number[][] {
  const g = rng(3)
  const Y = Array.from({ length: LOOP_N }, (_, n) => {
    const t = (2 * Math.PI * n) / LOOP_N
    return [Math.sin(t), 0.8 * Math.sin(2 * t), 0.6 * Math.cos(t), 0.3 * Math.cos(3 * t)].map(
      (v) => v + 0.03 * g.normal(),
    )
  })
  return centre(Y).Y
}
