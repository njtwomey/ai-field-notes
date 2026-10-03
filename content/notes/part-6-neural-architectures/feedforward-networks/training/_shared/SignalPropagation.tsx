import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
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
  const [activation, setActivation] = useState<Activation>(defaultActivation)
  const [init, setInit] = useState<InitScheme>(defaultInit)
  const [depth, setDepth] = useState(30)
  const [alpha, setAlpha] = useState(1)
  const [seed, setSeed] = useState(1)

  const d = useMemo(() => draws(seed), [seed])

  const lines = useMemo((): Line[] => {
    if (variant === 'init') {
      return SCHEMES.map((s, slot) => ({
        name: INIT_LABEL[s],
        slot,
        profile: propagate(d, { activation, gain: GAIN[s], depth }),
      }))
    }
    return [
      { name: 'plain', slot: 0, profile: propagate(d, { activation, gain: GAIN[init], depth }) },
      {
        name: `residual, α = ${formatNumber(alpha)}`,
        slot: 1,
        profile: propagate(d, { activation, gain: GAIN[init], depth, alpha }),
      },
    ]
  }, [variant, d, activation, init, depth, alpha])

  const charts = useMemo(() => {
    const layers = Array.from({ length: depth + 1 }, (_, l) => l)
    const series = (key: keyof Profile): XYSeries[] =>
      lines.map((line) => ({ name: line.name, type: 'line', x: layers, y: line.profile[key], slot: line.slot }))
    return { activation: series('activation'), gradient: series('gradient') }
  }, [lines, depth])

  const caption =
    variant === 'init'
      ? `A ${depth}-layer network of width ${WIDTH}, evaluated on a batch of ${BATCH} standard-normal inputs, with weights drawn from each initialisation. Left: the root mean square of the activations hₗ at each layer. Right: the root mean square of the gradient ∂ℓ/∂hₗ, started from a standard-normal gradient at the top. The vertical axes are logarithmic. Xavier keeps a linear network at scale 1; He keeps a ReLU network near 1; the small fixed variance shrinks both signals by orders of magnitude.`
      : `The same ${depth}-layer network of width ${WIDTH}, with plain layers hₗ = φ(Wₗhₗ₋₁) and with residual layers hₗ = hₗ₋₁ + α Wₗφ(hₗ₋₁). Left: activation RMS per layer. Right: gradient RMS per layer, started from a standard-normal gradient at the top; the vertical axes are logarithmic. With He initialisation and α = 1 the residual stack's mean square doubles at every layer. With α = 1/√L = ${formatNumber(1 / Math.sqrt(depth))} it grows by a factor of about e ≈ 2.7 over the whole stack. With a tiny branch initialisation the residual stack is close to the identity, while the plain stack's signals vanish.`

  const ratio = (p: Profile) => ({
    act: p.activation[depth] / p.activation[0],
    grad: p.gradient[0] / p.gradient[depth],
  })

  return (
    <Interactive
      title={
        variant === 'init' ? 'Signal propagation at initialisation' : 'Signal propagation with residual connections'
      }
      caption={caption}
      controls={
        <>
          <ParamChoice label="activation φ" value={activation} onChange={setActivation} options={ACTIVATIONS} />
          {variant === 'residual' && (
            <ParamChoice
              label="initialisation"
              value={init}
              onChange={setInit}
              options={SCHEMES.map((s) => ({ value: s, label: INIT_LABEL[s] }))}
            />
          )}
          <ParamSlider label="depth L" value={depth} onChange={setDepth} min={2} max={MAX_DEPTH} step={1} />
          {variant === 'residual' && (
            <ParamSlider label="branch scale α" value={alpha} onChange={setAlpha} min={0.05} max={1} step={0.01} />
          )}
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New weights</ParamButton>
        </>
      }
      readout={lines.map((line) => {
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
        <XYChart series={charts.activation} yLog xLabel="layer l" yLabel="RMS of hₗ" height={300} />
        <XYChart series={charts.gradient} yLog xLabel="layer l" yLabel="RMS of ∂ℓ/∂hₗ" height={300} />
      </div>
    </Interactive>
  )
}
