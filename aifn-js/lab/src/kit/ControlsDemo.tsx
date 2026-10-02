import { useState } from 'react'
import {
  Button,
  ButtonGroup,
  Choice,
  Combobox,
  NumberField,
  Player,
  Select,
  Slider,
  StepControls,
  Switch,
  useParam,
} from '@lab/controls'
import { Controls, Tex } from '@lab/layout'
import { Readout, Readouts } from '@lab/viz'

const KERNELS = [
  { value: 'rbf', label: 'squared exponential' },
  { value: 'matern32', label: 'Matérn 3/2' },
  { value: 'matern52', label: 'Matérn 5/2' },
  { value: 'periodic', label: 'periodic' },
] as const
type Kernel = (typeof KERNELS)[number]['value']

const DISTRIBUTIONS = [
  'normal',
  'log-normal',
  'student-t',
  'cauchy',
  'laplace',
  'logistic',
  'gumbel',
  'exponential',
  'gamma',
  'inverse-gamma',
  'beta',
  'kumaraswamy',
  'chi-square',
  'f',
  'weibull',
  'pareto',
  'rayleigh',
  'rice',
  'nakagami',
  'von-mises',
  'uniform',
  'triangular',
  'arcsine',
  'wald',
  'levy',
  'skew-normal',
  'bernoulli',
  'binomial',
  'poisson',
  'geometric',
  'negative-binomial',
  'hypergeometric',
  'beta-binomial',
  'categorical',
  'multinomial',
  'dirichlet',
  'wishart',
  'inverse-wishart',
  'zipf',
  'skellam',
] as const
type Distribution = (typeof DISTRIBUTIONS)[number]

/** Every control, each bound to state shown in the readouts below. */
export function ControlsDemo() {
  const rate = useParam(0.35, { min: 0, max: 1 })
  const k = useParam(4, { min: 1, max: 20, step: 1 })
  const temperature = useParam(1, { min: 0.05, max: 5, step: 0.05 })
  const [noise, setNoise] = useState(0.2)
  const [seed, setSeed] = useState(7)
  const [tolerance, setTolerance] = useState(1e-6)
  const [kernel, setKernel] = useState<Kernel>('rbf')
  const [distribution, setDistribution] = useState<Distribution>('gamma')
  const [shortChoice, setShortChoice] = useState<'mean' | 'median' | 'mode'>('median')
  const [log, setLog] = useState(false)
  const [frame, setFrame] = useState(0)
  const [iteration, setIteration] = useState(0)
  const [presses, setPresses] = useState(0)

  return (
    <div className="flex flex-col gap-4 rounded-xl border bg-card p-4">
      <Controls>
        <Slider label="rate (automatic step 0.005)" param={rate} />
        <Slider label="clusters k (step 1)" param={k} />
        <Slider label={<Tex>{String.raw`\text{temperature } T`}</Tex>} param={temperature} />
        <Slider label="noise (not steppable)" value={noise} onChange={setNoise} min={0} max={2} steppable={false} />
        <Slider label="disabled" value={0.5} onChange={() => {}} min={0} max={1} disabled />
        <NumberField label="seed" value={seed} onChange={setSeed} min={0} max={2 ** 31} step={1} />
        <NumberField label="tolerance" value={tolerance} onChange={setTolerance} gt={0} scale="log10" />
        <Select label="kernel (Select, 4 options)" value={kernel} onChange={setKernel} options={KERNELS} />
        <Combobox
          label="distribution (Combobox, 40 options)"
          value={distribution}
          onChange={setDistribution}
          options={DISTRIBUTIONS}
        />
        <Choice
          label="summary (Choice → Select)"
          value={shortChoice}
          onChange={setShortChoice}
          options={['mean', 'median', 'mode']}
        />
        <Choice
          label="distribution (Choice → Combobox)"
          value={distribution}
          onChange={setDistribution}
          options={DISTRIBUTIONS}
        />
        <Switch label="log scale" checked={log} onChange={setLog} />
      </Controls>
      <div className="flex flex-wrap items-end gap-6">
        <Player className="min-w-80 flex-1" label="frame" value={frame} onChange={setFrame} count={120} loop />
        <StepControls
          onStep={() => setIteration((i) => i + 1)}
          onRun={() => setIteration(25)}
          onReset={() => setIteration(0)}
          done={iteration >= 25}
        />
        <ButtonGroup aria-label="Example group">
          <Button variant="outline" size="sm" onClick={() => setPresses((p) => p - 1)}>
            −1
          </Button>
          <Button variant="outline" size="sm" onClick={() => setPresses(0)}>
            reset
          </Button>
          <Button variant="outline" size="sm" onClick={() => setPresses((p) => p + 1)}>
            +1
          </Button>
        </ButtonGroup>
      </div>
      <Readouts>
        <Readout label="rate" value={rate.value} />
        <Readout label="k" value={k.value} />
        <Readout label="T" value={temperature.value} />
        <Readout label="noise" value={noise.toFixed(3)} />
        <Readout label="seed" value={seed} />
        <Readout label="tolerance" value={tolerance.toExponential(1)} />
        <Readout label="kernel" value={kernel} />
        <Readout label="distribution" value={distribution} />
        <Readout label="summary" value={shortChoice} />
        <Readout label="log" value={String(log)} />
        <Readout label="frame" value={frame} />
        <Readout label="iteration" value={iteration} />
        <Readout label="presses" value={presses} />
      </Readouts>
      <p className="text-xs text-muted-foreground">
        Sliders: click the track to jump, drag, use the arrows, or type in the field (Enter or blur commits with clamp
        and snap, Escape reverts, ↑/↓ step, Shift × 10). Without a step, a slider takes a 1-2-5 step from its range. The
        player plays at the chosen rate; Space toggles it when it has focus.
      </p>
    </div>
  )
}
