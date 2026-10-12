/** The 1-D datasets the SplineVM prototypes share. */

/** A curve with two kinks, the case a linear spline should recover with exactly two knots. */
export const kinks = (x: number) => Math.abs(x - 0.3) - 0.8 * Math.max(0, x - 0.7)

export const REGRESSION = {
  step: { label: 'step (sign x)', fn: 'step' as const, range: [-2, 2] as const },
  kinks: { label: 'two kinks', fn: kinks, range: [0, 1] as const },
  bump: { label: 'bump', fn: 'bump' as const, range: [-1, 1] as const },
  sinc: { label: 'sinc', fn: 'sinc' as const, range: [-4, 4] as const },
  doppler: { label: 'Doppler', fn: 'doppler' as const, range: [0, 1] as const },
}
export const sigmoid = (f: number) => 1 / (1 + Math.exp(-f))
export const CLASSIFICATION = {
  window: { label: 'window (two edges)', p: (x: number) => (Math.abs(x) < 0.6 ? 0.9 : 0.1) },
  threshold: { label: 'threshold', p: (x: number) => sigmoid(12 * x) },
  wave: { label: 'wave', p: (x: number) => 0.5 + 0.4 * Math.sin(1.5 * Math.PI * x) },
}
export const CLASS_RANGE = [-2, 2] as const
