import { Shuffle } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { PlayButton } from './Playback'
import { usePlayLoop } from './usePlayLoop'
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
    const { normal } = rng(3)
    const w = new Float64Array(N * N)
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) w[i * N + j] = w[j * N + i] = normal() / Math.sqrt(N)
    return { W: { n: N, w }, patterns: [] }
  }
  const P = scenario === 'hopfield-2' ? 2 : 3
  const { uniform } = rng(scenario === 'hopfield-2' ? 1 : 4)
  const patterns = Array.from({ length: P }, () => randomPattern(N, uniform))
  return { W: hebbian(patterns, N), patterns }
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
  const [scenario, setScenario] = useState<LandscapeScenario>(initial)
  const [start, setStart] = useState<number>(() => codeAt(20, 9))
  const [temperature, setTemperature] = useState(0)
  const [seed, setSeed] = useState(1)
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)

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
    () =>
      runAsync(net.W, spinsOf(start, N), rng(seed).uniform, { T: temperature, maxSweeps: temperature > 0 ? 30 : 20 }),
    [net, start, seed, temperature],
  )
  const total = traj.sites.length
  const shown = Math.min(t, total)
  const state = useMemo(() => stateAt(traj, shown), [traj, shown])

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

  const overlay = useMemo((): HeatmapOverlay[] => {
    const points = (codes: number[]) => ({ x: codes.map((c) => cellOf(c)[0]), y: codes.map((c) => cellOf(c)[1]) })
    const hasMemories = net.patterns.length > 0
    const other = fixed.filter((c) => !memoryCodes.has(c) && !reversedCodes.has(c))
    const layers: HeatmapOverlay[] = [{ name: 'path', type: 'line', ...path, slot: 3, showPoints: true }]
    if (hasMemories) {
      layers.push({ name: 'stored pattern', type: 'scatter', ...points([...memoryCodes]), slot: 0 })
      layers.push({ name: 'reversed pattern', type: 'scatter', ...points([...reversedCodes]), slot: 1 })
      if (other.length) layers.push({ name: 'spurious minimum', type: 'scatter', ...points(other), slot: 2 })
    } else layers.push({ name: 'local minimum', type: 'scatter', ...points(fixed), slot: 2 })
    return layers
  }, [net, fixed, memoryCodes, reversedCodes, path])

  const [sc, sr] = cellOf(start)
  const handles = useMemo(
    (): Handle[] => [
      {
        kind: 'point',
        at: [sc, sr],
        label: 'start',
        onDrag: ([x, y]) => {
          const col = Math.min(SIDE - 1, Math.max(0, Math.round(x)))
          const row = Math.min(SIDE - 1, Math.max(0, Math.round(y)))
          setStart(codeAt(col, row))
          setT(0)
          setPlaying(false)
        },
      },
    ],
    [sc, sr],
  )

  usePlayLoop(playing, 6, (n) => {
    const next = Math.min(total, shown + n)
    setT(next)
    if (next >= total) setPlaying(false)
    return next < total
  })

  const code = codeOf(state)
  const updates = useMemo(() => Array.from({ length: total + 1 }, (_, i) => i), [total])
  const lastSite = shown > 0 ? traj.sites[shown - 1] : null
  const series = useMemo(
    (): XYSeries[] => [
      { name: 'energy', type: 'line', x: updates, y: traj.energy, slot: 3 },
      { name: 'now', type: 'scatter', x: [shown], y: [traj.energy[shown]], emphasis: true },
    ],
    [updates, traj, shown],
  )
  const traceHandles = useMemo(
    (): Handle[] => [
      {
        kind: 'x',
        at: shown,
        label: 'update',
        onDrag: (x) => {
          setPlaying(false)
          setT(Math.max(0, Math.min(total, Math.round(x))))
        },
      },
    ],
    [shown, total],
  )
  const [lo, hi] = useMemo(() => [Math.min(...energies), Math.max(...energies)], [energies])

  return (
    <Interactive
      title="Energy landscape of a ten-unit network"
      caption={
        <>
          All 1024 states of ten ±1 units, coloured by energy (light is low). Units 1–5 set the column and units 6–10
          the row, each in Gray-code order, so side-by-side cells differ in one unit and the map wraps around at its
          edges. Markers show the fixed points: states that no single flip can lower. Drag the start state anywhere on
          the map, then step through the asynchronous updates one unit at a time. At temperature 0 every step keeps or
          lowers the energy and the path ends in a fixed point; above 0 the path can climb out of shallow minima.
        </>
      }
      controls={
        <>
          <ParamChoice
            label="network"
            value={scenario}
            onChange={(v) => {
              setScenario(v)
              setT(0)
              setPlaying(false)
            }}
            options={SCENARIOS}
          />
          <ParamSlider
            label="temperature T"
            value={temperature}
            onChange={(v) => {
              setTemperature(v)
              setT(0)
            }}
            min={0}
            max={1.5}
            step={0.05}
          />
          <ParamSlider
            label="update"
            value={shown}
            onChange={(v) => {
              setPlaying(false)
              setT(v)
            }}
            min={0}
            max={total}
            step={1}
            withArrows
            debounceMs={0}
          />
          <ParamSlider
            label="update-order seed"
            value={seed}
            onChange={(v) => {
              setSeed(v)
              setT(0)
            }}
            min={1}
            max={20}
            step={1}
          />
          <div className="flex flex-wrap gap-2 self-end">
            <PlayButton
              playing={playing}
              disabled={shown >= total && !playing}
              onToggle={() => setPlaying((p) => !p)}
            />
            <ParamButton
              onClick={() => {
                setStart(Math.floor(rng(seed * 7919 + start).uniform() * (1 << N)))
                setT(0)
                setPlaying(false)
              }}
            >
              <Shuffle /> Random start
            </ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="state" value={spinText(state)} />
          <Readout label="energy" value={formatNumber(energies[code])} />
          <Readout label="last update" value={lastSite === null ? '–' : `unit ${lastSite + 1}`} />
          <Readout label="fixed points" value={fixed.length} />
          <Readout
            label="now"
            value={isFixedPoint(net.W, state) ? 'at a fixed point' : traj.converged || temperature > 0 ? 'moving' : '–'}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={z}
          xLabel="units 1–5 (Gray-code position)"
          yLabel="units 6–10 (Gray-code position)"
          valueLabel="E"
          range={[lo, hi]}
          overlay={overlay}
          marker={cellOf(code)}
          handles={handles}
          equalAspect
          ariaLabel="Energy of every state of a ten-unit network, with fixed points and the path of the current run"
        />
        <XYChart
          height={300}
          xLabel="update"
          yLabel="energy E"
          series={series}
          handles={traceHandles}
          ariaLabel="Energy after each asynchronous update"
        />
      </div>
    </Interactive>
  )
}
