import { RotateCcw, StepForward } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Pixels,
  Plot,
  type PlotPointer,
  Points,
  Readout,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn-compute/foundation/random'
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
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

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

const PIXEL_RANGE: [number, number] = [-1, 1]

/** Each scenario sets the temperature, field and starting state; T and h then change the running system. */
const SCENARIO = variants(
  Object.fromEntries(
    SCENARIOS.map((sc) => [
      sc.value,
      {
        label: sc.label,
        params: {
          T: float(sc.T, { min: 0.5, max: 5, step: 0.01, label: 'temperature T' }),
          h: float(sc.h, { min: -1, max: 1, step: 0.01, label: 'field h' }),
        },
      },
    ]),
  ) as Record<Scenario, { label: string; params: { T: ReturnType<typeof float>; h: ReturnType<typeof float> } }>,
  { label: 'Scenario', choiceLabel: 'scenario', initial: 'critical' },
)

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
  const g = stream(seed)
  const uniform = () => drawUniform(g)
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
  const state = useFigureState({
    scenario: SCENARIO,
    rule: choice<Rule>(
      [
        { value: 'metropolis', label: 'Metropolis' },
        { value: 'glauber', label: 'Glauber (heat bath)' },
      ],
      'metropolis',
      { label: 'update rule' },
    ),
    speed: int(2, {
      min: 0,
      max: SPEEDS.length - 1,
      step: 1,
      label: 'sweeps per second',
      format: (v) => String(SPEEDS[v]),
    }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const scenario = state.scenario.key as Scenario
  const { T, h } = state.scenario.values as { T: number; h: number }
  const seed = state.seed
  const [playing, setPlaying] = useState(false)
  const [generation, setGeneration] = useState(0)
  const [, setTick] = useState(0)

  const sim = useMemo(() => createSim(generation, scenario, seed), [generation, scenario, seed])
  const refresh = () => setTick((k) => k + 1)

  const runSweeps = (n: number) => {
    advanceSweeps(sim, n, T, h, state.rule)
    refresh()
  }

  const stepSpin = () => {
    setPlaying(false)
    advanceSpin(sim, T, h, state.rule)
    refresh()
  }

  usePlayLoop(playing, SPEEDS[state.speed], (n) => {
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
  const box = useMemo(() => (last ? pixelBox(last.site % L, Math.floor(last.site / L)) : null), [last])

  const { history } = sim
  const m = history.m[history.m.length - 1]
  const e = history.e[history.e.length - 1]
  const recent = history.m.slice(-WINDOW)
  const meanAbsM = recent.reduce((a, v) => a + Math.abs(v), 0) / recent.length

  const traces = [
    { name: 'magnetisation m', x: [...history.sweep], y: [...history.m], slot: 0 },
    { name: 'energy per spin', x: [...history.sweep], y: [...history.e], slot: 1 },
  ] as const

  const curveT = useMemo(() => toFlat(linspace(0.5, 5, 181)), [])
  const curve = useMemo(() => curveT.map(onsagerMagnetisation), [curveT])
  const phase = useMemo(
    () =>
      [
        { name: 'Onsager–Yang |m|, infinite lattice', x: curveT, y: curve, slot: 2 },
        { name: `simulated |m|, last ${WINDOW} sweeps`, x: [T], y: [meanAbsM], emphasis: true },
      ] as const,
    [curveT, curve, T, meanAbsM],
  )

  const xAxis = useAxis({ label: 'sweep', hold: 'union' })
  const yAxis = useAxis({ label: 'per spin', range: [-2.2, 1.1] })
  const xAxis2 = useAxis({ label: 'temperature T', range: [0.5, 5] })
  const yAxis2 = useAxis({ label: '|m|', range: [0, 1.05] })
  const latticeX = useAxis({ range: [-0.5, L - 0.5], nice: false })
  const latticeY = useAxis({ range: [-0.5, L - 0.5], nice: false, inverse: true, equal: latticeX })
  return (
    <Figure
      title="Ising lattice explorer"
      state={state}
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
          <div className="flex flex-wrap gap-2 self-end">
            <PlayButton playing={playing} onToggle={() => setPlaying((p) => !p)} />
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setPlaying(false)
                runSweeps(1)
              }}
            >
              <StepForward /> Sweep
            </Button>
            <Button variant="outline" size="sm" onClick={stepSpin}>
              <StepForward /> One spin
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setGeneration((k) => k + 1)
                setPlaying(false)
              }}
            >
              <RotateCcw /> Restart
            </Button>
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="sweep" value={sim.sweeps + (sim.partial ? ` + ${sim.partial} spins` : '')} />
          <Readout label="m" value={formatNumber(m)} />
          <Readout label="energy per spin" value={formatNumber(e)} />
          <Readout label="T / critical T" value={formatNumber(T / T_CRITICAL)} />
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
          <Plot
            x={latticeX}
            y={latticeY}
            bare
            height={400}
            onPointer={onPointer}
            ariaLabel="Spins of the Ising lattice"
          >
            <Pixels width={L} height={L} values={values} scale="diverging" range={PIXEL_RANGE} />
            {box && <Curve name="updated spin" x={box.x} y={box.y} emphasis live />}
          </Plot>
        </div>
        <div className="flex flex-col gap-2">
          <Plot x={xAxis} y={yAxis} height={200} ariaLabel={'Magnetisation and energy per spin over sweeps'}>
            <Curve {...traces[0]} />
            <Curve {...traces[1]} />
          </Plot>
          <Plot
            x={xAxis2}
            y={yAxis2}
            height={200}
            ariaLabel={'Spontaneous magnetisation against temperature, with the simulated value'}
          >
            <Curve {...phase[0]} />
            <Points {...phase[1]} />
            <Handle {...state.handle('scenario.T', { label: 'T' })} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
