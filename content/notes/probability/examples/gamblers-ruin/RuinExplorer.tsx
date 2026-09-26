import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

/** P(reach N before 0 | start at i), winning each unit bet with probability p. */
function winProbability(i: number, N: number, p: number): number {
  if (Math.abs(p - 0.5) < 1e-12) return i / N
  const r = (1 - p) / p
  return (1 - r ** i) / (1 - r ** N)
}

/** Expected number of bets until the gambler reaches 0 or N. */
function expectedDuration(i: number, N: number, p: number): number {
  if (Math.abs(p - 0.5) < 1e-12) return i * (N - i)
  const q = 1 - p
  return i / (q - p) - (N / (q - p)) * winProbability(i, N, p)
}

const SHOWN_PATHS = 6
const MAX_STEPS = 200_000

/** Sample paths of the gambler's fortune, and the exact win probability against the starting fortune. */
export function RuinExplorer() {
  const [p, setP] = useState(0.5)
  const [target, setTarget] = useState(20)
  const [runs, setRuns] = useState(1000)
  const [seed, setSeed] = useState(1)
  const start = useParam(10, { min: 1, max: 99, step: 1 })
  const i = Math.min(start.value, target - 1)

  const sim = useMemo(() => {
    const { uniform } = rng(seed)
    const paths: XYSeries[] = []
    let wins = 0
    let totalSteps = 0
    for (let r = 0; r < runs; r++) {
      let x = i
      let steps = 0
      const record = r < SHOWN_PATHS
      const ys = record ? [x] : []
      while (x > 0 && x < target && steps < MAX_STEPS) {
        x += uniform() < p ? 1 : -1
        steps++
        if (record) ys.push(x)
      }
      if (x >= target) wins++
      totalSteps += steps
      if (record)
        paths.push({
          name: r === 0 ? 'one path' : 'other paths',
          type: 'line',
          x: ys.map((_, t) => t),
          y: ys,
          muted: r > 0,
        })
    }
    return { paths, win: wins / runs, duration: totalSteps / runs }
  }, [p, target, runs, seed, i])

  const curve = useMemo((): XYSeries[] => {
    const xs = Array.from({ length: target + 1 }, (_, k) => k)
    return [
      { name: 'exact P(win)', type: 'line', x: xs, y: xs.map((k) => winProbability(k, target, p)), slot: 0 },
      { name: 'fair game', type: 'line', x: [0, target], y: [0, 1], dashed: true, slot: 1 },
      { name: 'simulated', type: 'scatter', x: [i], y: [sim.win], emphasis: true },
    ]
  }, [target, p, i, sim.win])

  const handles: Handle[] = [{ kind: 'x', at: i, label: 'start', onDrag: (x) => start.set(Math.round(x)) }]

  return (
    <Interactive
      title="Ruin or riches"
      caption="Left: sample paths of the gambler's fortune, one bet per step, until it reaches 0 or the target. Right: the exact probability of reaching the target against the starting fortune, with the share of simulated runs that did. Drag the line labelled start to change the starting fortune. A small edge against the gambler, such as p = 18/38 ≈ 0.474 in American roulette, makes a large target almost unreachable."
      controls={
        <>
          <ParamSlider label="P(win a bet) p" value={p} onChange={setP} min={0.4} max={0.6} step={0.002} />
          <ParamSlider label="starting fortune i" param={start} />
          <ParamSlider label="target N" value={target} onChange={setTarget} min={2} max={100} step={1} />
          <ParamSlider label="simulated runs" value={runs} onChange={setRuns} min={100} max={5000} step={100} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={30} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="P(win), exact" value={formatNumber(winProbability(i, target, p))} />
          <Readout label="P(win), simulated" value={formatNumber(sim.win)} />
          <Readout label="expected bets, exact" value={formatNumber(expectedDuration(i, target, p))} />
          <Readout label="mean bets, simulated" value={formatNumber(sim.duration)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart height={300} xLabel="bet" yLabel="fortune" series={sim.paths} yRange={[0, target]} />
        <XYChart
          height={300}
          xLabel="starting fortune i"
          yLabel="P(reach N before 0)"
          series={curve}
          xRange={[0, target]}
          yRange={[0, 1]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
