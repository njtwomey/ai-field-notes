import { useMemo } from 'react'
import {
  Contours,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import { lagrangePoints, linspace, potentialGrid, runParticle } from './restricted'

const XS = linspace(-1.5, 1.5, 201)
const YS = linspace(-1.5, 1.5, 201)
const FRAMES = 1200

/** Which necks of the zero-velocity curve are open for Jacobi constant C, given 2Ω at L1, L2, L3 and L4. */
function regime(c: number, [c1, c2, c3, c4]: number[]): string {
  if (c >= c1) return 'closed at L1: confined near one mass, or outside both'
  if (c >= c2) return 'open at L1 only: can pass between the masses'
  if (c >= c3) return 'open at L1 and L2: forbidden crescent through L4, L3 and L5'
  if (c > c4) return 'open at L3 too: forbidden only round L4 and L5'
  return 'no forbidden region'
}

export function TestParticle() {
  const state = useFigureState({
    mu: slider(0.0001, 0.05, 0.001, { step: 0.0001, label: 'mass ratio μ' }),
    x0: slider(-1.5, 1.5, 0, { step: 0.001, onChart: true, label: 'start x' }),
    y0: slider(-1.5, 1.5, 1, { step: 0.001, onChart: true, label: 'start y' }),
    vx0: float(0, { min: -0.5, max: 0.5, label: 'start velocity ẋ (rotating frame)', suggestions: [-0.01, 0, 0.01] }),
    vy0: float(0, { min: -0.5, max: 0.5, label: 'start velocity ẏ (rotating frame)', suggestions: [-0.01, 0, 0.01] }),
    tEnd: float(600, {
      min: 10,
      max: 5000,
      label: 'end time t (2π per orbit of the masses)',
      suggestions: [100, 600, 2000],
    }),
    rtol: float(1e-10, { min: 1e-13, max: 1e-4, label: 'relative tolerance', suggestions: [1e-6, 1e-8, 1e-10] }),
  })
  const { mu, x0, y0, vx0, vy0, tEnd, rtol } = state

  const points = useMemo(() => lagrangePoints(mu), [mu])
  const sim = useComputed(() => runParticle(mu, [x0, y0, vx0, vy0], tEnd, rtol), [mu, x0, y0, vx0, vy0, tEnd, rtol], {
    mode: 'release',
  })
  const run = sim.value
  const c = run.jacobi
  const cl = points.slice(0, 4).map((p) => p.C)

  const grid = useMemo(() => potentialGrid(mu, XS, YS, c + 1), [mu, c])
  // The forbidden region 2Ω < C, drawn as a muted mask; allowed cells are left empty.
  const forbidden = useMemo(() => grid.map((row) => row.map((v) => (v < c ? 1 : NaN))), [grid, c])

  // Frames at even times; the trail keeps a step once the particle has moved 0.004 since the last kept one.
  const view = useMemo(() => {
    const last = run.time[run.time.length - 1]
    const frames: number[] = []
    let i = 0
    for (let k = 0; k < FRAMES; k++) {
      const t = (last * k) / (FRAMES - 1)
      while (i + 1 < run.time.length && run.time[i + 1] <= t) i++
      frames.push(i)
    }
    const kept = [0]
    for (let j = 1; j < run.time.length; j++) {
      const r = kept[kept.length - 1]
      if (Math.hypot(run.x[j] - run.x[r], run.y[j] - run.y[r]) > 0.004) kept.push(j)
    }
    return { frames, kept }
  }, [run])

  const [frame, setFrame] = usePlayhead(FRAMES)
  const step = view.frames[Math.min(frame, FRAMES - 1)]
  const tx: number[] = []
  const ty: number[] = []
  for (const j of view.kept) {
    if (j > step) break
    tx.push(run.x[j])
    ty.push(run.y[j])
  }
  tx.push(run.x[step])
  ty.push(run.y[step])

  const xa = useAxis({ label: 'x (rotating frame)', range: [-1.5, 1.5], nice: false })
  const ya = useAxis({ label: 'y', range: [-1.5, 1.5], nice: false, equal: xa })

  return (
    <Figure
      title="A test particle near the Lagrange points"
      state={state}
      defaultSize="L"
      caption={`A massless particle in the rotating frame of two masses with ratio μ (separation 1, G = 1, one orbit of the masses every 2π). Drag the start point; the start velocity is in the rotating frame. The particle is integrated by Dormand–Prince with the Coriolis terms. Shaded: the forbidden region 2Ω < C, bounded by the zero-velocity curve of the particle's Jacobi constant C, which the particle can touch only at rest. Started at rest at (0, 1), 30° from L4, it librates on a tadpole round L4. Started at rest at L3, (−1, 0), it traces a horseshoe that wraps round L4, L3 and L5 and turns back short of m₂.${run.failure ? ` The run ${run.failure}.` : ''}`}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES}
          label="time"
          format={(k) => `t = ${formatNumber(run.time[view.frames[Math.min(k, FRAMES - 1)]])}`}
        />
      }
      readouts={
        <>
          <Readout label="Jacobi constant C" value={c.toFixed(6)} />
          <Readout label="2Ω at L1, L2, L3, L4" value={cl.map((v) => v.toFixed(4)).join(', ')} />
          <Readout label="zero-velocity curve" value={regime(c, cl)} />
          <Readout label="largest drift |C(t) − C(0)|" value={run.jacobiDrift.toExponential(1)} />
          <Readout label="time t" value={formatNumber(run.time[step])} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Raster x={XS} y={YS} z={forbidden} range={[0, 1]} colorBar={false} fillOpacity={0.18} valueLabel="forbidden" />
        <Contours x={XS} y={YS} z={grid} levels={[c]} labels={false} />
        <Points name="masses" x={[-mu, 1 - mu]} y={[0, 0]} labels={['m₁', 'm₂']} size={12} muted />
        <Points
          name="Lagrange points"
          x={points.map((p) => p.x)}
          y={points.map((p) => p.y)}
          labels={points.map((p) => p.name)}
          size={6}
          muted
        />
        <Curve id="trail" name="particle" x={tx} y={ty} slot={0} thin live />
        <Points id="now" name="particle" x={[run.x[step]]} y={[run.y[step]]} slot={0} size={10} live />
        <Handle {...state.handle(['x0', 'y0'], { label: 'start' })} />
      </Plot>
    </Figure>
  )
}
