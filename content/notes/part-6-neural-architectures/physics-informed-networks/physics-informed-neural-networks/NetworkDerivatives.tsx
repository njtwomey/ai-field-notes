import { useContext, useMemo, type ReactNode } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { partials } from 'aifn-compute/dynamics/fields'
import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, mul, sin, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'

const PANEL = 220
const xs = Array.from({ length: 401 }, (_, i) => -1 + i / 200)
const X = fromData(Float64Array.from(xs), [xs.length, 1])

const ACTIVATIONS = [
  { value: 'tanh', label: 'tanh' },
  { value: 'sine', label: 'sine' },
  { value: 'softplus', label: 'softplus' },
  { value: 'gelu', label: 'GELU' },
  { value: 'relu', label: 'ReLU' },
] as const

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** A panel whose vertical range fits its curve symmetrically, so a curve that is zero stays a flat line at 0. */
function Panel({ title, y, slot }: { title: string; y: number[]; slot: number }) {
  const top = Math.max(1e-6, ...y.map(Math.abs)) * 1.1
  const x = useAxis({ label: 'x', range: [-1, 1] })
  const v = useAxis({ range: [-top, top] })
  return (
    <FixedHeight height={PANEL}>
      <Plot x={x} y={v} title={title}>
        <Curve name={title} x={xs} y={y} slot={slot} />
      </Plot>
    </FixedHeight>
  )
}

/** A freshly initialised network u(x) and the first two derivatives that autodiff takes of it. */
export function NetworkDerivatives() {
  const state = useFigureState({
    activation: choice(ACTIVATIONS, 'tanh', { label: 'activation' }),
    width: int(32, { ge: 2, le: 256, suggestions: [8, 32, 128], label: 'hidden width' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { activation, width, seed } = state
  const view = useMemo(() => {
    // The sine network takes its input scaled up, as SIREN-style networks do, so it shows a few oscillations.
    const fn = activation === 'sine' ? (x: Value) => sin(mul(x, 3)) : activation
    const net = Mlp([1, width, width, 1], { activation: fn, init: xavierUniform() })
    const params = net.init(stream(`network-derivatives/${seed}`)) as Params
    const [u, ux, uxx] = partials((x) => net.apply(params as never, mul(x, 2)), X, [[], [0], [0, 0]]).map((v) =>
      Array.from(toFlat(v as Tensor)),
    )
    const flat = uxx.filter((v) => Math.abs(v) < 1e-9).length / uxx.length
    return { u, ux, uxx, flat }
  }, [activation, width, seed])
  return (
    <Figure
      title="What a network's derivatives look like"
      state={state}
      hoverReadout={false}
      caption="A multilayer perceptron with two hidden layers, at its random initial weights, on [−1, 1]: the function u (left) and its first and second derivatives (middle, right), taken by forward-mode automatic differentiation. Each panel has its own vertical scale. Smooth activations give smooth derivatives of every order. With ReLU the network is piecewise linear: u′ is a step function and u″ is zero everywhere except at the kinks, where it is undefined. A residual that contains u″ cannot see a ReLU network's curvature at all."
      readouts={
        <>
          <Readout label="share of points with u″ = 0" value={`${(100 * view.flat).toFixed(0)}%`} />
          <Readout label="largest |u″|" value={formatNumber(Math.max(...view.uxx.map(Math.abs)))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Panel title="u" y={view.u} slot={0} />
        <Panel title="u′" y={view.ux} slot={1} />
        <Panel title="u″" y={view.uxx} slot={2} />
      </div>
    </Figure>
  )
}
