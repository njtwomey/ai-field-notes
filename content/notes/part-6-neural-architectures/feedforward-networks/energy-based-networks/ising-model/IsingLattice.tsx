import { RotateCcw, StepForward } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  ImagePlot,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type ImagePlotLine,
  type PlotPointer,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { PlayButton } from '../_shared/Playback'
import { usePlayLoop } from '../_shared/usePlayLoop'
import {
  T_CRITICAL,
  latticeEnergy,
  magnetisation,
  onsagerMagnetisation,
  pixelBox,
  randomLattice,
  sweepLattice,
  uniformLattice,
  updateSpin,
  type Lattice,
  type Rule,
  type SpinUpdate,
} from '../_shared/spins'

const L = 64
const HISTORY = 1500
/** Sweeps averaged for the point on the phase diagram. */
const WINDOW = 100
const SPEEDS = [1, 4, 15, 60]

type Scenario = 'hot' | 'ordered' | 'quench' | 'critical' | 'field'
const SCENARIOS: { value: Scenario; label: string; T: number; h: number; start: 'random' | 'up' }[] = [
  { value: 'hot', label: 'hot, random start', T: 3.5, h: 0, start: 'random' },
  { value: 'ordered', label: 'all up, T = 1.8', T: 1.8, h: 0, start: 'up' },
  { value: 'quench', label: 'quench to T = 1', T: 1, h: 0, start: 'random' },
  { value: 'critical', label: 'near critical T', T: 2.27, h: 0, start: 'random' },
  { value: 'field', label: 'field against the order', T: 1.6, h: -0.25, start: 'up' },
]

/** The mutable simulation: the lattice and its trace. Re-rendering is driven by `tick`. */
type Sim = {
  id: number
  lattice: Lattice
  uniform: () => number
  sweeps: number
  /** Single-spin updates since the last whole sweep. */
  partial: number
  history: { sweep: number[]; m: number[]; e: number[] }
  last: SpinUpdate | null
}

function createSim(id: number, scenario: Scenario, seed: number): Sim {
  const spec = SCENARIOS.find((s) => s.value === scenario)!
  const { uniform } = rng(seed)
  const lattice = spec.start === 'up' ? uniformLattice(L, 1) : randomLattice(L, uniform)
  return {
    id,
    lattice,
    uniform,
    sweeps: 0,
    partial: 0,
    history: { sweep: [0], m: [magnetisation(lattice)], e: [latticeEnergy(lattice, spec.h)] },
    last: null,
  }
}

function record(sim: Sim, h: number) {
  const { history } = sim
  history.sweep.push(sim.sweeps)
  history.m.push(magnetisation(sim.lattice))
  history.e.push(latticeEnergy(sim.lattice, h))
  if (history.sweep.length > HISTORY) {
    history.sweep.shift()
    history.m.shift()
    history.e.shift()
  }
}

/** Run n whole sweeps and record the trace after each. Mutates the simulation. */
function advanceSweeps(sim: Sim, n: number, T: number, h: number, rule: Rule) {
  for (let k = 0; k < n; k++) {
    sweepLattice(sim.lattice, T, h, rule, sim.uniform)
    sim.sweeps += 1
    record(sim, h)
  }
  sim.partial = 0
  sim.last = null
}

/** One single-spin update, remembered for the readout; a completed sweep is recorded. Mutates the simulation. */
function advanceSpin(sim: Sim, T: number, h: number, rule: Rule) {
  sim.last = updateSpin(sim.lattice, T, h, rule, sim.uniform)
  sim.partial += 1
  if (sim.partial >= L * L) {
    sim.partial = 0
    sim.sweeps += 1
    record(sim, h)
  }
}

/** Flip the 5 × 5 block centred on (c0, r0) to the opposite of its centre spin. Mutates the lattice. */
function flipBlock(sim: Sim, c0: number, r0: number) {
  const { s } = sim.lattice
  const sign = -s[r0 * L + c0] as 1 | -1
  for (let dr = -2; dr <= 2; dr++)
    for (let dc = -2; dc <= 2; dc++) s[((r0 + dr + L) % L) * L + ((c0 + dc + L) % L)] = sign
}

/**
 * A 64 × 64 Ising lattice with periodic edges, simulated with Metropolis or Glauber single-spin updates. Scenarios set
 * the temperature, field and starting state; the sliders then change the temperature and field of the running system.
 */
export function IsingLattice() {
  const [scenario, setScenario] = useState<Scenario>('critical')
  const [seed, setSeed] = useState(1)
  const [rule, setRule] = useState<Rule>('metropolis')
  const [speed, setSpeed] = useState(2)
  const [playing, setPlaying] = useState(false)
  const [generation, setGeneration] = useState(0)
  const [, setTick] = useState(0)
  const T = useParam(2.27, { min: 0.5, max: 5, step: 0.01 })
  const h = useParam(0, { min: -1, max: 1, step: 0.01 })

  const sim = useMemo(() => createSim(generation, scenario, seed), [generation, scenario, seed])
  const refresh = () => setTick((k) => k + 1)

  const choose = (value: Scenario) => {
    const spec = SCENARIOS.find((s) => s.value === value)!
    setScenario(value)
    T.set(spec.T)
    h.set(spec.h)
    setGeneration((g) => g + 1)
    setPlaying(true)
  }

  const runSweeps = (n: number) => {
    advanceSweeps(sim, n, T.value, h.value, rule)
    refresh()
  }

  const stepSpin = () => {
    setPlaying(false)
    advanceSpin(sim, T.value, h.value, rule)
    refresh()
  }

  usePlayLoop(playing, SPEEDS[speed], (n) => {
    runSweeps(n)
    return true
  })

  // Clicking flips a 5 × 5 block, e.g. to seed a droplet of the opposite phase.
  const onPointer = (event: PlotPointer) => {
    if (event.type !== 'click') return
    const [x, y] = event.point
    const c0 = Math.round(x)
    const r0 = Math.round(y)
    if (c0 < 0 || r0 < 0 || c0 >= L || r0 >= L) return
    flipBlock(sim, c0, r0)
    refresh()
  }

  const values = Float32Array.from(sim.lattice.s)
  const last = sim.last
  const lines = useMemo(
    (): ImagePlotLine[] =>
      last ? [{ name: 'updated spin', ...pixelBox(last.site % L, Math.floor(last.site / L)), emphasis: true }] : [],
    [last],
  )

  const { history } = sim
  const m = history.m[history.m.length - 1]
  const e = history.e[history.e.length - 1]
  const recent = history.m.slice(-WINDOW)
  const meanAbsM = recent.reduce((a, v) => a + Math.abs(v), 0) / recent.length

  const traces: XYSeries[] = [
    { name: 'magnetisation m', type: 'line', x: [...history.sweep], y: [...history.m], slot: 0 },
    { name: 'energy per spin', type: 'line', x: [...history.sweep], y: [...history.e], slot: 1 },
  ]

  const curveT = useMemo(() => linspace(0.5, 5, 181), [])
  const curve = useMemo(() => curveT.map(onsagerMagnetisation), [curveT])
  const phase = useMemo(
    (): XYSeries[] => [
      { name: 'Onsager–Yang |m|, infinite lattice', type: 'line', x: curveT, y: curve, slot: 2 },
      { name: `simulated |m|, last ${WINDOW} sweeps`, type: 'scatter', x: [T.value], y: [meanAbsM], emphasis: true },
    ],
    [curveT, curve, T.value, meanAbsM],
  )
  const phaseHandles = useMemo((): Handle[] => [{ kind: 'x', at: T.value, label: 'T', onDrag: T.set }], [T])

  return (
    <Interactive
      title="Ising lattice explorer"
      caption={
        <>
          A 64 × 64 lattice of spins (red +1, blue −1) with periodic edges and coupling J = 1. Each sweep makes 4096
          single-spin updates at random sites. Pick a scenario, press Play, and change the temperature T while it runs.
          Below the critical temperature 2.269 domains grow and the magnetisation settles near ±m(T); above it the spins
          stay mixed; near it domains of every size appear. Step one spin to see the flip cost ΔE and the flip
          probability. Click the lattice to flip a 5 × 5 block, and drag the temperature line on the phase diagram.
        </>
      }
      controls={
        <>
          <ParamChoice label="scenario" value={scenario} onChange={choose} options={SCENARIOS} />
          <ParamSlider label="temperature T" param={T} />
          <ParamSlider label="field h" param={h} />
          <ParamChoice
            label="update rule"
            value={rule}
            onChange={setRule}
            options={[
              { value: 'metropolis', label: 'Metropolis' },
              { value: 'glauber', label: 'Glauber (heat bath)' },
            ]}
          />
          <ParamSlider
            label="sweeps per second"
            value={speed}
            onChange={setSpeed}
            min={0}
            max={SPEEDS.length - 1}
            step={1}
            format={(v) => String(SPEEDS[v])}
          />
          <div className="flex flex-wrap gap-2 self-end">
            <PlayButton playing={playing} onToggle={() => setPlaying((p) => !p)} />
            <ParamButton
              onClick={() => {
                setPlaying(false)
                runSweeps(1)
              }}
            >
              <StepForward /> Sweep
            </ParamButton>
            <ParamButton onClick={stepSpin}>
              <StepForward /> One spin
            </ParamButton>
            <ParamButton
              onClick={() => {
                setSeed((s) => s + 1)
                setPlaying(false)
              }}
            >
              <RotateCcw /> Restart
            </ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="sweep" value={sim.sweeps + (sim.partial ? ` + ${sim.partial} spins` : '')} />
          <Readout label="m" value={formatNumber(m)} />
          <Readout label="energy per spin" value={formatNumber(e)} />
          <Readout label="T / critical T" value={formatNumber(T.value / T_CRITICAL)} />
          {last && (
            <Readout
              label="last spin"
              value={`ΔE = ${formatNumber(last.dE)}, p(flip) = ${formatNumber(last.p)}, ${last.flipped ? 'flipped' : 'kept'}`}
            />
          )}
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <div className="mx-auto w-full max-w-[420px]">
          <ImagePlot
            width={L}
            height={L}
            values={values}
            scale="diverging"
            range={[-1, 1]}
            lines={lines}
            onPointer={onPointer}
            ariaLabel="Spins of the Ising lattice"
          />
        </div>
        <div className="flex flex-col gap-2">
          <XYChart
            height={200}
            xLabel="sweep"
            yLabel="per spin"
            yRange={[-2.2, 1.1]}
            series={traces}
            ariaLabel="Magnetisation and energy per spin over sweeps"
          />
          <XYChart
            height={200}
            xLabel="temperature T"
            yLabel="|m|"
            xRange={[0.5, 5]}
            yRange={[0, 1.05]}
            series={phase}
            handles={phaseHandles}
            ariaLabel="Spontaneous magnetisation against temperature, with the simulated value"
          />
        </div>
      </div>
    </Interactive>
  )
}
