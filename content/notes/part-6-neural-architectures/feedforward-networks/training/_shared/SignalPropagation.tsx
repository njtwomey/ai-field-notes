import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  BATCH,
  GAIN,
  INIT_LABEL,
  MAX_DEPTH,
  WIDTH,
  draws,
  propagate,
  type Activation,
  type InitScheme,
  type Profile,
} from './deepnet'

const ACTIVATIONS: { value: Activation; label: string }[] = [
  { value: 'linear', label: 'linear' },
  { value: 'tanh', label: 'tanh' },
  { value: 'sigmoid', label: 'sigmoid' },
  { value: 'relu', label: 'ReLU' },
]

const SCHEMES: InitScheme[] = ['small', 'xavier', 'he']

type Line = { name: string; slot: number; profile: Profile }

type Props = {
  /** `init` compares the three initialisations; `residual` compares a plain and a residual stack. */
  variant?: 'init' | 'residual'
  defaultActivation?: Activation
  defaultInit?: InitScheme
}

export function SignalPropagation({ variant = 'init', defaultActivation = 'tanh', defaultInit = 'he' }: Props) {
  const state = useFigureState({
    activation: choice<Activation>(ACTIVATIONS, defaultActivation, { label: 'activation φ' }),
    init: choice<InitScheme>(
      SCHEMES.map((s) => ({ value: s, label: INIT_LABEL[s] })),
      defaultInit,
      { label: 'initialisation', when: () => variant === 'residual' },
    ),
    depth: int(30, { min: 2, max: MAX_DEPTH, step: 1, label: 'depth L' }),
    alpha: float(1, { min: 0.05, max: 1, step: 0.01, label: 'branch scale α', when: () => variant === 'residual' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const d = useMemo(() => draws(state.seed), [state.seed])

  const lines = useMemo((): Line[] => {
    if (variant === 'init') {
      return SCHEMES.map((s, slot) => ({
        name: INIT_LABEL[s],
        slot,
        profile: propagate(d, { activation: state.activation, gain: GAIN[s], depth: state.depth }),
      }))
    }
    return [
      {
        name: 'plain',
        slot: 0,
        profile: propagate(d, { activation: state.activation, gain: GAIN[state.init], depth: state.depth }),
      },
      {
        name: `residual, α = ${formatNumber(state.alpha)}`,
        slot: 1,
        profile: propagate(d, {
          activation: state.activation,
          gain: GAIN[state.init],
          depth: state.depth,
          alpha: state.alpha,
        }),
      },
    ]
  }, [variant, d, state.activation, state.init, state.depth, state.alpha])

  const charts = useMemo(() => {
    const layers = Array.from({ length: state.depth + 1 }, (_, l) => l)
    const series = (key: keyof Profile): SeriesSpec[] =>
      lines.map((line) => ({ name: line.name, type: 'line', x: layers, y: line.profile[key], slot: line.slot }))
    return { activation: series('activation'), gradient: series('gradient') }
  }, [lines, state.depth])

  const caption =
    variant === 'init'
      ? `A ${state.depth}-layer network of width ${WIDTH}, evaluated on a batch of ${BATCH} standard-normal inputs, with weights drawn from each initialisation. Left: the root mean square of the activations hₗ at each layer. Right: the root mean square of the gradient ∂ℓ/∂hₗ, started from a standard-normal gradient at the top. The vertical axes are logarithmic. Xavier keeps a linear network at scale 1; He keeps a ReLU network near 1; the small fixed variance shrinks both signals by orders of magnitude.`
      : `The same ${state.depth}-layer network of width ${WIDTH}, with plain layers hₗ = φ(Wₗhₗ₋₁) and with residual layers hₗ = hₗ₋₁ + α Wₗφ(hₗ₋₁). Left: activation RMS per layer. Right: gradient RMS per layer, started from a standard-normal gradient at the top; the vertical axes are logarithmic. With He initialisation and α = 1 the residual stack's mean square doubles at every layer. With α = 1/√L = ${formatNumber(1 / Math.sqrt(state.depth))} it grows by a factor of about e ≈ 2.7 over the whole stack. With a tiny branch initialisation the residual stack is close to the identity, while the plain stack's signals vanish.`

  const ratio = (p: Profile) => ({
    act: p.activation[state.depth] / p.activation[0],
    grad: p.gradient[0] / p.gradient[state.depth],
  })

  const xAxis = useAxis({ label: 'layer l', hold: 'union' })
  const yAxis = useAxis({ label: 'RMS of hₗ', hold: 'union', log: true })
  const xAxis2 = useAxis({ label: 'layer l', hold: 'union' })
  const yAxis2 = useAxis({ label: 'RMS of ∂ℓ/∂hₗ', hold: 'union', log: true })
  return (
    <Figure
      title={
        variant === 'init' ? 'Signal propagation at initialisation' : 'Signal propagation with residual connections'
      }
      state={state}
      caption={caption}

      readouts={lines.map((line) => {
        const r = ratio(line.profile)
        return (
          <Readout
            key={line.name}
            label={`${line.name}: RMS(h_L)/RMS(h_0), RMS(∂ℓ/∂h_0)/RMS(∂ℓ/∂h_L) =`}
            value={`${formatNumber(r.act)}, ${formatNumber(r.grad)}`}
          />
        )
      })}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(charts.activation)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers(charts.gradient)}
        </Plot>
      </div>
    </Figure>
  )
}
