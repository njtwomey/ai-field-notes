import { Shuffle } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  seriesLayers,
  slider,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn/foundation/random'
import {
  codeOf,
  gray,
  grayInverse,
  hebbian,
  hopfieldEnergy,
  isFixedPoint,
  randomPattern,
  runAsync,
  spinsOf,
  stateAt,
  type Weights,
} from './spins'

/** Ten units: 2^10 = 1024 states, drawn as a 32 × 32 map. Columns code units 1–5 and rows units 6–10. */
const N = 10
const SIDE = 32
const AXIS = Array.from({ length: SIDE }, (_, i) => i)

export type LandscapeScenario = 'ferromagnet' | 'hopfield-2' | 'hopfield-3' | 'spin-glass'

const SCENARIOS: { value: LandscapeScenario; label: string }[] = [
  { value: 'ferromagnet', label: 'ferromagnet' },
  { value: 'hopfield-2', label: 'Hopfield, 2 memories' },
  { value: 'hopfield-3', label: 'Hopfield, 3 memories' },
  { value: 'spin-glass', label: 'spin glass' },
]

type Network = { W: Weights; patterns: Int8Array[] }

/** Seeds chosen so each scenario shows its point: 3 memories with seed 4 leave spurious minima beside the memories. */
function network(scenario: LandscapeScenario): Network {
  if (scenario === 'ferromagnet') {
    const w = new Float64Array(N * N)
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) w[i * N + j] = 1 / N
    return { W: { n: N, w }, patterns: [] }
  }
  if (scenario === 'spin-glass') {
    // Sherrington–Kirkpatrick couplings J_ij ~ N(0, 1/N), symmetric.
    const g = stream(3)
    const w = new Float64Array(N * N)
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) w[i * N + j] = w[j * N + i] = normal(g) / Math.sqrt(N)
    return { W: { n: N, w }, patterns: [] }
  }
  const P = scenario === 'hopfield-2' ? 2 : 3
  const g = stream(scenario === 'hopfield-2' ? 1 : 4)
  const patterns = Array.from({ length: P }, () => randomPattern(N, () => uniform(g)))
  return { W: hebbian(patterns, N), patterns }
}

/** Seeded uniform draws for the update order. */
function draws(seed: number) {
  const g = stream(seed)
  return () => uniform(g)
}

/** Map cell (column, row) of a state: Gray-code positions, so neighbouring cells differ in one unit. */
const cellOf = (code: number): [number, number] => [grayInverse(code & 31), grayInverse(code >> 5)]
const codeAt = (col: number, row: number) => gray(col) | (gray(row) << 5)

const spinText = (s: ArrayLike<number>) => Array.from(s, (v) => (v > 0 ? '+' : '−')).join('')

/**
 * Every state of a ten-unit network on one map, coloured by energy, with the network's fixed points marked. Drag the
 * start state anywhere and step through asynchronous updates one unit at a time.
 */
export function EnergyLandscape({ initial = 'hopfield-3' }: { initial?: LandscapeScenario }) {
  const state = useFigureState({
    network: choice<LandscapeScenario>(SCENARIOS, initial, { label: 'network' }),
    temperature: slider(0, 1.5, 0, { step: 0.05, label: 'temperature T' }),
    seed: int(1, { ge: 0, label: 'update-order seed' }),
    // The start state's map cell, moved by its handle.
    col: slider(0, SIDE - 1, 20, { step: 1, onChart: true }),
    row: slider(0, SIDE - 1, 9, { step: 1, onChart: true }),
  })
  const { network: scenario, temperature, seed } = state
  const start = codeAt(state.col, state.row)

  const net = useMemo(() => network(scenario), [scenario])
  const energies = useMemo(() => {
    const e = new Float64Array(1 << N)
    for (let c = 0; c < e.length; c++) e[c] = hopfieldEnergy(net.W, spinsOf(c, N))
    return e
  }, [net])
  const z = useMemo(() => AXIS.map((row) => AXIS.map((col) => energies[codeAt(col, row)])), [energies])
  const fixed = useMemo(() => {
    const out: number[] = []
    for (let c = 0; c < 1 << N; c++) if (isFixedPoint(net.W, spinsOf(c, N))) out.push(c)
    return out
  }, [net])

  const memoryCodes = useMemo(() => new Set(net.patterns.map((p) => codeOf(p))), [net])
  const reversedCodes = useMemo(() => new Set(net.patterns.map((p) => codeOf(p.map((v) => -v)))), [net])

  const traj = useMemo(
    () => runAsync(net.W, spinsOf(start, N), draws(seed), { T: temperature, maxSweeps: temperature > 0 ? 30 : 20 }),
    [net, start, seed, temperature],
  )
  const total = traj.sites.length
  // The walk through the updates restarts at 0 for a new run.
  const [position, setPosition] = useState({ traj, t: 0 })
  const shown = position.traj === traj ? Math.min(position.t, total) : 0
  const go = (v: number) => setPosition({ traj, t: Math.max(0, Math.min(total, Math.round(v))) })
  const spins = useMemo(() => stateAt(traj, shown), [traj, shown])

  // The visited states in order, one point per change of state.
  const path = useMemo(() => {
    const s = Int8Array.from(traj.start)
    const cells = [cellOf(codeOf(s))]
    for (let k = 0; k < shown; k++)
      if (traj.flipped[k]) {
        s[traj.sites[k]] = -s[traj.sites[k]]
        cells.push(cellOf(codeOf(s)))
      }
    return { x: cells.map((c) => c[0]), y: cells.map((c) => c[1]) }
  }, [traj, shown])

  const overlay = useMemo((): SeriesSpec[] => {
    const points = (codes: number[]) => ({ x: codes.map((c) => cellOf(c)[0]), y: codes.map((c) => cellOf(c)[1]) })
    const hasMemories = net.patterns.length > 0
    const other = fixed.filter((c) => !memoryCodes.has(c) && !reversedCodes.has(c))
    const layers: SeriesSpec[] = [{ name: 'path', type: 'line', ...path, slot: 3, showPoints: true }]
    if (hasMemories) {
      layers.push({ name: 'stored pattern', type: 'scatter', ...points([...memoryCodes]), slot: 0 })
      layers.push({ name: 'reversed pattern', type: 'scatter', ...points([...reversedCodes]), slot: 1 })
      if (other.length) layers.push({ name: 'spurious minimum', type: 'scatter', ...points(other), slot: 2 })
    } else layers.push({ name: 'local minimum', type: 'scatter', ...points(fixed), slot: 2 })
    return layers
  }, [net, fixed, memoryCodes, reversedCodes, path])

  const code = codeOf(spins)
  const updates = useMemo(() => Array.from({ length: total + 1 }, (_, i) => i), [total])
  const lastSite = shown > 0 ? traj.sites[shown - 1] : null
  const series = useMemo(
    () =>
      [
        { name: 'energy', x: updates, y: traj.energy, slot: 3 },
        { name: 'now', x: [shown], y: [traj.energy[shown]], emphasis: true },
      ] as const,
    [updates, traj, shown],
  )
  const [lo, hi] = useMemo(() => [Math.min(...energies), Math.max(...energies)], [energies])

  const xAxis = useAxis({ label: 'units 1–5 (Gray-code position)' })
  const yAxis = useAxis({ label: 'units 6–10 (Gray-code position)', equal: xAxis })
  const xAxis2 = useAxis({ label: 'update', hold: 'union' })
  const yAxis2 = useAxis({ label: 'energy E', hold: 'union' })
  return (
    <Figure
      title="Energy landscape of a ten-unit network"
      state={state}
      caption={
        <>
          All 1024 states of ten ±1 units, coloured by energy (light is low). Units 1–5 set the column and units 6–10
          the row, each in Gray-code order, so side-by-side cells differ in one unit and the map wraps around at its
          edges. Markers show the fixed points: states that no single flip can lower. Drag the start state anywhere on
          the map, then play or step through the asynchronous updates one unit at a time; drag along the energy chart to
          go back to any update. At temperature 0 every step keeps or lowers the energy and the path ends in a fixed
          point; above 0 the path can climb out of shallow minima.
        </>
      }
      controls={
        <>
          <Player value={shown} onChange={go} count={total + 1} label="update" format={(v) => `update ${v}`} />
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const [c, r] = cellOf(Math.floor(uniform(stream(seed * 7919 + start)) * (1 << N)))
              state.set('col', c)
              state.set('row', r)
            }}
          >
            <Shuffle /> Random start
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="state" value={spinText(spins)} />
          <Readout label="energy" value={formatNumber(energies[code])} />
          <Readout label="last update" value={lastSite === null ? '–' : `unit ${lastSite + 1}`} />
          <Readout label="fixed points" value={fixed.length} />
          <Readout
            label="now"
            value={isFixedPoint(net.W, spins) ? 'at a fixed point' : traj.converged || temperature > 0 ? 'moving' : '–'}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Plot
          x={xAxis}
          y={yAxis}
          ariaLabel={'Energy of every state of a ten-unit network, with fixed points and the path of the current run'}
        >
          <Raster x={AXIS} y={AXIS} z={z} range={[lo, hi]} valueLabel={'E'} />
          {seriesLayers(overlay, { live: true })}
          {cellOf(code) && <Points x={[cellOf(code)[0]]} y={[cellOf(code)[1]]} emphasis live />}
          <Handle {...state.handle(['col', 'row'], { label: 'start' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300} ariaLabel={'Energy after each asynchronous update'}>
          <Curve {...series[0]} />
          <Points {...series[1]} />
          <Handle kind="x" at={shown} label="update" onDrag={go} />
        </Plot>
      </div>
    </Figure>
  )
}
