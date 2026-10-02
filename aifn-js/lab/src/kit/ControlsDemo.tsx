import { useState } from 'react'
import {
  Code2,
  ListFilter,
  Palette,
  PlayCircle,
  Sliders,
  Sparkles,
  ToggleLeft,
} from 'lucide-react'
import {
  Button,
  ButtonGroup,
  Choice,
  CodeEditor,
  Combobox,
  ControlLabel,
  MultiCombobox,
  NumberField,
  Player,
  RampPicker,
  RevealToggle,
  Select,
  Slider,
  StepControls,
  SwatchPicker,
  Switch,
  useParam,
} from '@lab/controls'
import { palette } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { ControlGroup, Equation, live, tex, Tex, ThemeToggle } from '@lab/layout'
import { Readout, Readouts } from '@lab/viz'

const KERNELS = [
  { value: 'rbf', label: 'squared exponential' },
  { value: 'matern32', label: 'Matérn 3/2' },
  { value: 'matern52', label: 'Matérn 5/2' },
  { value: 'periodic', label: 'periodic' },
] as const
type Kernel = (typeof KERNELS)[number]['value']

const METRICS = ['accuracy', 'cross-entropy loss', 'f1 score', 'auc-roc', 'brier score', 'log-likelihood'] as const
type Metric = (typeof METRICS)[number]

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

const SAMPLE_CODE = `// Sample a mixture of Gaussians
function sampleMixture(n = 100, k = 3) {
  const centers = math.range(k).map(i => i * 2.5)
  return array.from({ length: n }, () => {
    const c = random.choice(centers)
    return c + random.normal(0, 0.5)
  })
}`

/**
 * Structured showcase of all lab controls, grouped consistently into:
 * 1. Visual & Colour Controls (theme, palette slots, stroke, opacity, radius, revealers)
 * 2. Colour Palettes & Ramps (diverging, sequential, categorical)
 * 3. Numeric & Continuous Parameters (sliders, typed number fields)
 * 4. Categorical & Selection Controls (select, combobox, multi-combobox, choice)
 * 5. State & Visibility Toggles (switches, buttons)
 * 6. Playback & Timeline Stepping (player, step controls)
 * 7. Code & Live Formula (CodeMirror editor, live KaTeX equation)
 */
export function ControlsDemo() {
  const { preference, resolved } = useTheme()

  // Visual & Colour state
  const [colorSlot, setColorSlot] = useState(0)
  const [opacity, setOpacity] = useState(0.85)
  const [lineWidth, setLineWidth] = useState(2.5)
  const [pointRadius, setPointRadius] = useState(6)
  const [revealGrid, setRevealGrid] = useState(true)
  const [revealVectors, setRevealVectors] = useState(false)
  const [revealContours, setRevealContours] = useState(true)

  // Numeric state
  const rate = useParam(0.35, { min: 0, max: 1 })
  const k = useParam(4, { min: 1, max: 20, step: 1 })
  const temperature = useParam(1, { min: 0.05, max: 5, step: 0.05 })
  const [noise, setNoise] = useState(0.2)
  const [seed, setSeed] = useState(7)
  const [tolerance, setTolerance] = useState(1e-6)

  // Categorical state
  const [kernel, setKernel] = useState<Kernel>('rbf')
  const [distribution, setDistribution] = useState<Distribution>('gamma')
  const [shortChoice, setShortChoice] = useState<'mean' | 'median' | 'mode'>('median')
  const [selectedMetrics, setSelectedMetrics] = useState<Metric[]>(['accuracy', 'f1 score'])

  // Toggles & Actions
  const [log, setLog] = useState(false)
  const [normalize, setNormalize] = useState(true)
  const [presses, setPresses] = useState(0)

  // Stepping & Playback
  const [frame, setFrame] = useState(0)
  const [iteration, setIteration] = useState(0)

  // Code
  const [code, setCode] = useState(SAMPLE_CODE)

  const catColors = palette.categorical[resolved]
  const activeColorHex = catColors[colorSlot % catColors.length]

  return (
    <div className="flex flex-col gap-6">
      {/* 1. Visual & Colour Controls */}
      <ControlGroup
        title="Visual & Colour Controls"
        description="Theme modes, categorical colour slots, geometric marks styling, and visibility reveal toggles."
        badge="Styling & Display"
        icon={Palette}
      >
        <div className="flex flex-col gap-1.5">
          <ControlLabel>theme ({preference} → {resolved})</ControlLabel>
          <div className="flex items-center gap-2 h-9">
            <ThemeToggle />
            <span className="text-xs text-muted-foreground font-mono">active: {resolved}</span>
          </div>
        </div>

        <SwatchPicker value={colorSlot} onChange={setColorSlot} label={`categorical colour slot (${colorSlot})`} />

        <Slider
          label="point radius (px)"
          value={pointRadius}
          onChange={setPointRadius}
          min={2}
          max={16}
          step={1}
        />
        <Slider
          label="stroke width (px)"
          value={lineWidth}
          onChange={setLineWidth}
          min={1}
          max={8}
          step={0.5}
        />
        <Slider
          label="opacity"
          value={opacity}
          onChange={setOpacity}
          min={0.1}
          max={1}
          step={0.05}
          format={(v) => `${Math.round(v * 100)}%`}
        />

        <div className="col-span-full flex flex-wrap items-center gap-3 pt-1">
          <ControlLabel className="mr-2">layer reveals:</ControlLabel>
          <RevealToggle label="Grid lines" pressed={revealGrid} onChange={setRevealGrid} />
          <RevealToggle label="Vector field" pressed={revealVectors} onChange={setRevealVectors} />
          <RevealToggle label="Contours" pressed={revealContours} onChange={setRevealContours} />
        </div>

        {/* Live SVG mark preview */}
        <div className="col-span-full mt-2 flex items-center justify-between gap-4 rounded-lg border bg-muted/20 p-3">
          <div className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Live mark preview:</span> circle rendered with active slot{' '}
            <code className="font-mono text-xs text-foreground">{colorSlot}</code>, opacity{' '}
            <code className="font-mono text-xs text-foreground">{Math.round(opacity * 100)}%</code>, radius{' '}
            <code className="font-mono text-xs text-foreground">{pointRadius}px</code>.
          </div>
          <div className="flex size-14 shrink-0 items-center justify-center rounded-md border bg-card">
            <svg width="48" height="48" viewBox="0 0 48 48">
              {revealGrid && (
                <path d="M 0 24 H 48 M 24 0 V 48" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2,2" opacity={0.3} />
              )}
              <circle
                cx="24"
                cy="24"
                r={pointRadius * 1.5}
                fill={activeColorHex}
                fillOpacity={opacity}
                stroke={activeColorHex}
                strokeWidth={lineWidth}
              />
            </svg>
          </div>
        </div>
      </ControlGroup>

      {/* 2. Colour Palettes & Ramps */}
      <ControlGroup
        title="Data Palettes & Ramps"
        description="Sequential ramps for magnitude, diverging ramps for signed balances, and categorical palettes."
        badge="Continuous & Diverging"
        icon={Sparkles}
      >
        <RampPicker label={`Sequential ramp: magnitude 0 → 1 (${resolved})`} value="sequential" />
        <RampPicker label={`Diverging ramp: signed deviation −1 … 0 … +1 (${resolved})`} value="diverging" />
      </ControlGroup>

      {/* 3. Numeric & Continuous Parameters */}
      <ControlGroup
        title="Numeric & Continuous Controls"
        description="Floating-point sliders with snap and clamp, LaTeX parameter labels, and strict typed number inputs."
        badge="Sliders & NumberFields"
        icon={Sliders}
      >
        <Slider label="rate (auto step 0.005)" param={rate} />
        <Slider label="clusters k (step 1)" param={k} />
        <Slider label={<Tex>{String.raw`\text{temperature } T`}</Tex>} param={temperature} />
        <Slider label="noise (continuous, unstepped)" value={noise} onChange={setNoise} min={0} max={2} steppable={false} />
        <Slider label="disabled parameter" value={0.5} onChange={() => {}} min={0} max={1} disabled />
        <NumberField label="seed (int bound)" value={seed} onChange={setSeed} min={0} max={2 ** 31} step={1} />
        <NumberField label="tolerance (log10, strict gt)" value={tolerance} onChange={setTolerance} gt={0} scale="log10" />
      </ControlGroup>

      {/* 4. Categorical & Selection Controls */}
      <ControlGroup
        title="Categorical & Selection Controls"
        description="Dropdown selects, segmented choices, searchable autocomplete comboboxes, and multi-select tags."
        badge="Select & Combobox"
        icon={ListFilter}
      >
        <Select label="kernel (Select, 4 items)" value={kernel} onChange={setKernel} options={KERNELS} />
        <Choice
          label="summary statistic (Choice)"
          value={shortChoice}
          onChange={setShortChoice}
          options={['mean', 'median', 'mode']}
        />
        <Combobox
          label="distribution (Combobox, 40 items)"
          value={distribution}
          onChange={setDistribution}
          options={DISTRIBUTIONS}
        />
        <MultiCombobox
          label="metrics (MultiCombobox, multi-tag)"
          value={selectedMetrics}
          onChange={setSelectedMetrics}
          options={METRICS}
          placeholder="Add metric…"
        />
      </ControlGroup>

      {/* 5. State & Visibility Toggles */}
      <ControlGroup
        title="State & Visibility Flags"
        description="Boolean state switches and segmented action button groups."
        badge="Switches & Actions"
        icon={ToggleLeft}
      >
        <Switch label="logarithmic scale" checked={log} onChange={setLog} />
        <Switch label="normalize features" checked={normalize} onChange={setNormalize} />
        <div className="flex flex-col gap-1.5">
          <ControlLabel>action button group</ControlLabel>
          <ButtonGroup aria-label="Step counter">
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
      </ControlGroup>

      {/* 6. Stepping, Timeline & Playback */}
      <ControlGroup
        title="Timeline & Algorithm Stepping"
        description="Timeline player with scrubber and speed, and iterative algorithm step controls."
        badge="Player & Steps"
        icon={PlayCircle}
      >
        <div className="col-span-full flex flex-wrap items-end gap-6">
          <Player className="min-w-80 flex-1" label="frame index" value={frame} onChange={setFrame} count={120} loop />
          <StepControls
            onStep={() => setIteration((i) => i + 1)}
            onRun={() => setIteration(25)}
            onReset={() => setIteration(0)}
            done={iteration >= 25}
          />
        </div>
      </ControlGroup>

      {/* 7. Code & Live Formula */}
      <ControlGroup
        title="Code & Live Formula"
        description="CodeMirror 6 interactive editor with syntax highlighting, and KaTeX equations with live parameter bindings."
        badge="CodeMirror & KaTeX"
        icon={Code2}
      >
        <div className="col-span-full">
          <ControlLabel className="mb-2">interactive code editor</ControlLabel>
          <div className="rounded-lg border bg-background overflow-hidden">
            <CodeEditor value={code} onChange={setCode} language="javascript" />
          </div>
        </div>
        <div className="col-span-full mt-2">
          <ControlLabel className="mb-2">live equation binding</ControlLabel>
          <div className="rounded-lg bg-muted/40 p-4 overflow-x-auto">
            <Equation>
              {tex`p(x \mid T=${live(temperature.value.toFixed(2))}, k=${live(k.value)}) \propto \exp\left(-\frac{\|x - \mu\|^2}{2 \cdot ${live(temperature.value.toFixed(2))}}\right)`}
            </Equation>
          </div>
        </div>
      </ControlGroup>

      {/* Live Readouts Summary */}
      <div className="rounded-xl border bg-card p-4 space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Current State Readouts</h4>
        <Readouts>
          <Readout label="colour slot" value={colorSlot} />
          <Readout label="hex" value={activeColorHex} />
          <Readout label="opacity" value={`${Math.round(opacity * 100)}%`} />
          <Readout label="radius" value={`${pointRadius}px`} />
          <Readout label="rate" value={rate.value} />
          <Readout label="k" value={k.value} />
          <Readout label="T" value={temperature.value} />
          <Readout label="noise" value={noise.toFixed(3)} />
          <Readout label="seed" value={seed} />
          <Readout label="tolerance" value={tolerance.toExponential(1)} />
          <Readout label="kernel" value={kernel} />
          <Readout label="dist" value={distribution} />
          <Readout label="summary" value={shortChoice} />
          <Readout label="metrics" value={selectedMetrics.join(', ')} />
          <Readout label="log" value={String(log)} />
          <Readout label="frame" value={frame} />
          <Readout label="step" value={iteration} />
          <Readout label="presses" value={presses} />
        </Readouts>
        <p className="text-xs text-muted-foreground">
          Every control binds cleanly to either state schemas (`useFigureState`) or standalone parameter hooks (`useParam`),
          with instant keyboard navigation, automatic step calculation, validation clamp/snap, and reactive dark/light themes.
        </p>
      </div>
    </div>
  )
}
