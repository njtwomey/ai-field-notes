import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type Segment } from '@/components/viz'
import { rng } from '@/lib/math'

const STEPS = 4000
const EVERY = 100
const N_DATA = 400

const LATTICES = [
  { value: 'grid', label: '8 × 8 grid' },
  { value: 'chain', label: 'chain of 40' },
] as const
type Lattice = (typeof LATTICES)[number]['value']

/** Grid positions of the units: an 8 × 8 square lattice or a 1 × 40 chain. */
function lattice(kind: Lattice): [number, number][] {
  if (kind === 'chain') return Array.from({ length: 40 }, (_, i) => [i, 0])
  return Array.from({ length: 64 }, (_, i) => [i % 8, Math.floor(i / 8)])
}

/** Learning rate and neighbourhood width, both decaying exponentially over training. */
function schedule(step: number, sigma0: number): { eta: number; sigma: number } {
  const f = step / STEPS
  return { eta: 0.5 * (0.01 / 0.5) ** f, sigma: sigma0 * (0.5 / sigma0) ** f }
}

function train(kind: Lattice) {
  const r = rng(7)
  const data = Array.from({ length: N_DATA }, () => [r.uniform(), r.uniform()] as [number, number])
  const grid = lattice(kind)
  const sigma0 = kind === 'chain' ? 10 : 4
  // Start every unit near the centre, so that training has to unfold the lattice.
  const w = grid.map(() => [0.5 + 0.05 * (r.uniform() - 0.5), 0.5 + 0.05 * (r.uniform() - 0.5)])
  const snapshots: number[][][] = [w.map((p) => [...p])]
  for (let step = 1; step <= STEPS; step++) {
    const x = data[Math.floor(r.uniform() * N_DATA)]
    let c = 0
    let best = Infinity
    w.forEach((p, j) => {
      const d = (p[0] - x[0]) ** 2 + (p[1] - x[1]) ** 2
      if (d < best) {
        best = d
        c = j
      }
    })
    const { eta, sigma } = schedule(step, sigma0)
    w.forEach((p, j) => {
      // Neighbourhood on the lattice, not in data space: this is what makes the map topology-preserving.
      const g2 = (grid[j][0] - grid[c][0]) ** 2 + (grid[j][1] - grid[c][1]) ** 2
      const h = Math.exp(-g2 / (2 * sigma * sigma))
      p[0] += eta * h * (x[0] - p[0])
      p[1] += eta * h * (x[1] - p[1])
    })
    if (step % EVERY === 0) snapshots.push(w.map((p) => [...p]))
  }
  return { data, grid, snapshots, sigma0 }
}

/** Mean distance to the best-matching unit, and the share of points whose two best units are not lattice neighbours. */
function quality(data: [number, number][], w: number[][], grid: [number, number][]) {
  let qe = 0
  let te = 0
  for (const x of data) {
    const d = w.map((p, j) => [Math.hypot(p[0] - x[0], p[1] - x[1]), j]).sort((a, b) => a[0] - b[0])
    qe += d[0][0]
    const [a, b] = [grid[d[0][1]], grid[d[1][1]]]
    if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) > 1) te++
  }
  return { qe: qe / data.length, te: te / data.length }
}

export function SomUnfolding() {
  const [kind, setKind] = useState<Lattice>('grid')
  const run = useMemo(() => train(kind), [kind])
  const [step, setStep] = useState(STEPS)
  const w = run.snapshots[step / EVERY]
  const segments = useMemo(() => {
    const index = new Map(run.grid.map((g, j) => [`${g[0]},${g[1]}`, j]))
    const out: Segment[] = []
    run.grid.forEach((g, j) => {
      for (const [dx, dy] of [
        [1, 0],
        [0, 1],
      ]) {
        const k = index.get(`${g[0] + dx},${g[1] + dy}`)
        if (k !== undefined) out.push({ from: [w[j][0], w[j][1]], to: [w[k][0], w[k][1]] })
      }
    })
    return out
  }, [run, w])
  const q = useMemo(() => quality(run.data, w, run.grid), [run, w])
  const { eta, sigma } = schedule(step, run.sigma0)

  return (
    <Interactive
      title="A self-organising map unfolds over the data"
      caption="400 points uniform in the unit square (grey) and the prototypes of the units (blue), joined along the lattice. All units start near the centre. Step through training: while the neighbourhood is wide, each update drags large parts of the lattice together and the map unfolds; as it narrows, units spread to cover the square. A chain of 40 units folds into a curve that fills the square while keeping its order. The topographic error counts points whose two nearest units are not lattice neighbours."
      controls={
        <>
          <ParamChoice label="lattice" value={kind} onChange={setKind} options={LATTICES} />
          <ParamSlider
            label="training step"
            value={step}
            onChange={setStep}
            min={0}
            max={STEPS}
            step={EVERY}
            debounceMs={0}
            withArrows
          />
        </>
      }
      readout={
        <>
          <Readout label="learning rate η" value={formatNumber(eta)} />
          <Readout label="neighbourhood width σ (lattice units)" value={formatNumber(sigma)} />
          <Readout label="quantisation error" value={formatNumber(q.qe)} />
          <Readout label="topographic error" value={formatNumber(q.te)} />
        </>
      }
    >
      <XYChart
        equalAspect
        xRange={[-0.05, 1.05]}
        yRange={[-0.05, 1.05]}
        xLabel="x₁"
        yLabel="x₂"
        segments={segments}
        series={[
          {
            name: 'data',
            type: 'scatter',
            x: run.data.map((p) => p[0]),
            y: run.data.map((p) => p[1]),
            muted: true,
          },
          { name: 'units', type: 'scatter', x: w.map((p) => p[0]), y: w.map((p) => p[1]), slot: 0 },
        ]}
      />
    </Interactive>
  )
}
