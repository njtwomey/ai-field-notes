import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { logit } from 'aifn-compute/numerics/special'
import { grid } from '../_shared/geometry'

type Fn = 'arsinh' | 'artanh' | 'arcosh'
const V = 4
const DIAG = [{ from: [-V, -V] as const, to: [V, V] as const }]

type Spec = {
  label: string
  forward: (x: number) => number
  inverse: (y: number) => number
  /** Domain of the forward function shown, and of the inverse. */
  xs: number[]
  ys: number[]
  /** The logarithmic form of the inverse, drawn dashed over it. */
  closed: (y: number) => number
  closedName: string
  /** A simpler curve the inverse approaches, drawn muted. */
  approx?: { name: string; f: (y: number) => number; ys: number[] }
}

const SPECS: Record<Fn, Spec> = {
  arsinh: {
    label: 'arsinh, inverse of sinh',
    forward: Math.sinh,
    inverse: Math.asinh,
    xs: grid(-2.1, 2.1, 241),
    ys: grid(-V, V, 241),
    closed: (y) => Math.log(y + Math.sqrt(y * y + 1)),
    closedName: 'log(y + √(y² + 1))',
    approx: { name: 'log 2y', f: (y) => Math.log(2 * y), ys: grid(0.05, V, 200) },
  },
  artanh: {
    label: 'artanh, inverse of tanh',
    forward: Math.tanh,
    inverse: Math.atanh,
    xs: grid(-V, V, 241),
    ys: grid(-0.995, 0.995, 241),
    closed: (y) => 0.5 * logit((1 + y) / 2),
    closedName: '½ logit((1 + y)/2)',
  },
  arcosh: {
    label: 'arcosh, inverse of cosh on x ≥ 0',
    forward: Math.cosh,
    inverse: Math.acosh,
    xs: grid(0, 2.1, 241),
    ys: grid(1, V, 241),
    closed: (y) => Math.log(y + Math.sqrt(y * y - 1)),
    closedName: 'log(y + √(y² − 1))',
    approx: { name: 'log 2y', f: (y) => Math.log(2 * y), ys: grid(0.6, V, 200) },
  },
}

/**
 * An inverse function is its forward function reflected in the line y = x. Drag a point along the forward curve; its
 * mirror image lies on the inverse, which matches its closed logarithmic form.
 */
export function InverseGraphs() {
  const state = useFigureState({
    fn: choice<Fn>(
      (Object.keys(SPECS) as Fn[]).map((k) => ({ value: k, label: SPECS[k].label })),
      'arsinh',
      { label: 'function' },
    ),
    x0: slider(-2, 2, 1.2, { step: 0.01, onChart: true }),
  })
  const spec = SPECS[state.fn]
  const lo = spec.xs[0]
  const hi = spec.xs[spec.xs.length - 1]
  const x0 = Math.min(hi, Math.max(lo, state.x0))
  const y0 = spec.forward(x0)

  const curves = useMemo(
    () => ({
      fwd: spec.xs.map(spec.forward),
      inv: spec.ys.map(spec.inverse),
      closed: spec.ys.map(spec.closed),
      approx: spec.approx ? spec.approx.ys.map(spec.approx.f) : [],
    }),
    [spec],
  )
  const mirror = useMemo(() => [{ from: [x0, y0] as const, to: [y0, x0] as const }], [x0, y0])

  const x = useAxis({ label: 'x', range: [-V, V] })
  const y = useAxis({ label: 'y', range: [-V, V], equal: x })
  return (
    <Figure
      title="Inverse hyperbolic functions as reflections"
      state={state}
      caption="Pick a function and drag the point along the forward curve (grey). Its reflection in the line y = x (dotted) lies on the inverse function (blue), and the dashed black curve, the closed logarithmic form, lies exactly on top of it. For arsinh and arcosh the muted curve is log 2y: both grow like a logarithm for large arguments. arsinh is defined for every real number, arcosh only from 1, and artanh only on (−1, 1), where it is half the logit of (1 + y)/2."
      readouts={
        <>
          <Readout label="x" value={formatNumber(x0)} />
          <Readout label="forward value y" value={formatNumber(y0)} />
          <Readout label="inverse of y" value={formatNumber(spec.inverse(y0))} />
          <Readout label="closed form of y" value={formatNumber(spec.closed(y0))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight>
          <Segments name="y = x" segments={DIAG} muted dashed />
          <Curve name="forward function" x={spec.xs} y={curves.fwd} muted />
          {spec.approx && <Curve name={spec.approx.name} x={spec.approx.ys} y={curves.approx} muted dashed />}
          <Curve name={spec.label.split(',')[0]} x={spec.ys} y={curves.inv} slot={0} width={4} />
          <Curve name={spec.closedName} x={spec.ys} y={curves.closed} emphasis dashed width={1.5} />
          <Segments name="reflection" segments={mirror} muted dashed live />
          <Points name="mirror image" x={[y0]} y={[x0]} slot={0} size={9} live />
          <Handle kind="point" at={[x0, y0]} label="(x, f(x))" onDrag={([u]) => state.set('x0', u)} />
        </Plot>
      </div>
    </Figure>
  )
}
