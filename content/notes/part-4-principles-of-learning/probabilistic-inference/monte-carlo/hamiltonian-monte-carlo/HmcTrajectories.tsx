import { banana, funnel, nonCentredFunnel } from 'aifn-methods/data/targets'
import { hmc, type HmcState, type LogDensity } from 'aifn/inference/stochastic'
import { stream } from 'aifn/foundation/random'
import { linspace, tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import {
  Bars,
  Curve,
  Dashboard,
  DashboardCell,
  DashboardRow,
  Figure,
  Handle,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Switch,
  formatNumber,
  row,
  slider,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'

const f3 = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
type P2 = [number, number]

type Target = {
  density: LogDensity
  toPlot: (theta: ArrayLike<number>) => P2
  fromPlot: (p: P2) => number[]
  shown: LogDensity
  shownAt: (p: P2) => number[]
  box: { x: P2; y: P2 }
  labels: { x: string; y: string; first: string; second: string }
  start: P2
}

const FUNNEL = funnel()
const NC_FUNNEL = nonCentredFunnel()
const BANANA_PARAMS = { a: 1, b: 2 }
const HARD_BANANA = banana(BANANA_PARAMS)
const FUNNEL_BOX = { x: [-12, 12] as P2, y: [-8, 8] as P2 }
const funnelLabels = { x: 'x', y: 'v (log variance)', first: 'v', second: 'x' }

const TARGETS: Record<'centred funnel' | 'non-centred funnel' | 'hard banana', Target> = {
  'centred funnel': {
    density: FUNNEL,
    toPlot: (t) => [t[1], t[0]],
    fromPlot: ([x, v]) => [v, x],
    shown: FUNNEL,
    shownAt: ([x, v]) => [v, x],
    box: FUNNEL_BOX,
    labels: funnelLabels,
    start: [0.2, -4.5],
  },
  'non-centred funnel': {
    density: NC_FUNNEL,
    toPlot: (t) => {
      const c = toFlat(NC_FUNNEL.toOriginal(tensor(Array.from(t))) as Tensor)
      return [c[1], c[0]]
    },
    fromPlot: ([x, v]) => [v, x * Math.exp(-v / 2)],
    shown: FUNNEL,
    shownAt: ([x, v]) => [v, x],
    box: FUNNEL_BOX,
    labels: funnelLabels,
    start: [0.2, -4.5],
  },
  'hard banana': {
    density: HARD_BANANA,
    toPlot: (t) => [t[0], t[1]],
    fromPlot: (p) => [...p],
    shown: HARD_BANANA,
    shownAt: (p) => [...p],
    box: { x: [-4, 4], y: [-4, 22] },
    labels: { x: 'x', y: 'y', first: 'x', second: 'y' },
    start: [-2.5, 10],
  },
}

const startOf = (t: Target) => ({
  sx: slider(t.box.x[0], t.box.x[1], t.start[0], { onChart: true, label: 'start x' }),
  sy: slider(t.box.y[0], t.box.y[1], t.start[1], { onChart: true, label: 'start y' }),
})
const TARGET = variants(
  {
    'centred funnel': { label: 'centred funnel', params: startOf(TARGETS['centred funnel']) },
    'non-centred funnel': { label: 'non-centred funnel', params: startOf(TARGETS['non-centred funnel']) },
    'hard banana': { label: 'hard banana', params: startOf(TARGETS['hard banana']) },
  },
  { label: '1 · target', choiceLabel: 'target' },
)
const integrator = (label: string) =>
  row(label, {
    eps: slider(0.02, 1, 0.3, { label: 'step size ε', step: 0.01 }),
    L: slider(2, 60, 25, { label: 'path length L (steps)', step: 1 }),
  })

function logDensityGrid(t: Target, n = 110) {
  const x = grid(t.box.x[0], t.box.x[1], n)
  const y = grid(t.box.y[0], t.box.y[1], n)
  const z = y.map((yi) => x.map((xj) => Math.max(-12, t.shown.logDensity(tensor(t.shownAt([xj, yi]))) as number)))
  return { x, y, z }
}

function trajectoryPoints(traj: Tensor, t: Target): P2[] {
  const flat = toFlat(traj)
  return Array.from({ length: flat.length / 2 }, (_, i) => t.toPlot([flat[2 * i], flat[2 * i + 1]]))
}

const ITERATIONS = 40

type Iteration = {
  points: P2[]
  dH: number[]
  energyError: number
  divergent: boolean
  accepted: boolean
  sample: P2
}

export function HmcTrajectories() {
  const state = useFigureState({ target: TARGET, integrator: integrator('2 · integrator') })
  const key = state.target.key as 'centred funnel' | 'non-centred funnel' | 'hard banana'
  const { eps, L } = state.integrator
  const [played, setPlayed] = useState({ key, pos: 0 })
  const pos = played.key === key ? played.pos : 0
  const setPos = (p: number) => setPlayed({ key, pos: p })
  const target = TARGETS[key]
  const surface = useMemo(() => logDensityGrid(target), [target])
  const { sx, sy } = state.target.values
  const start = useMemo((): P2 => [sx, sy], [sx, sy])
  const x0 = useMemo(() => target.fromPlot(start), [target, start])

  const frameCount = ITERATIONS * L
  const at = Math.min(pos, frameCount)
  const need = at === 0 ? 0 : Math.floor((at - 1) / L) + 1

  const iterations = useMemo(() => {
    const alg = hmc(target.density, { stepSize: eps, steps: L })
    const tr = trace(alg, { x0: tensor(x0) }, ITERATIONS, {
      stream: stream('showcase-hmc'),
      keep: 'all',
    }) as Trace<HmcState>
    return tr.steps.slice(1).map((s): Iteration => {
      const H = toFlat(s.energies)
      return {
        points: trajectoryPoints(s.trajectory, target),
        dH: H.map((h) => h - H[0]),
        energyError: s.energyError,
        divergent: s.divergent,
        accepted: s.accepted,
        sample: target.toPlot(toFlat(s.x)),
      }
    })
  }, [target, eps, L, x0])

  const fi = need - 1
  const current = fi >= 0 ? iterations[fi] : null
  const fk = current ? Math.min(((at - 1) % L) + 1, current.points.length - 1) : 0
  const finished = fi >= 0 && ((at - 1) % L) + 1 === L ? fi + 1 : Math.max(fi, 0)
  const done = useMemo(() => iterations.slice(0, finished), [iterations, finished])

  const layers = useMemo(() => {
    const past = iterations.slice(0, Math.max(fi, 0)).flatMap((it) => [...it.points, [NaN, NaN] as P2])
    const partial = current ? current.points.slice(0, fk + 1) : []
    const samples = [start, ...done.map((d) => d.sample)]
    const divergentStarts = done.flatMap((d, j) => (d.divergent ? [j === 0 ? start : done[j - 1].sample] : []))
    const xy = (pts: P2[]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) })
    return { past: xy(past), partial: xy(partial), samples: xy(samples), divergent: xy(divergentStarts) }
  }, [iterations, fi, fk, current, done, start])

  const energy = useMemo(() => {
    const dH = current ? current.dH.slice(0, fk + 1) : [0]
    return { x: dH.map((_, k) => k), y: dH }
  }, [current, fk])

  const perIteration = useMemo(() => {
    const xs = done.map((_, j) => j + 1)
    return {
      x: xs,
      ok: done.map((d) => (d.divergent ? NaN : clamp(d.energyError, -3, 3))),
      bad: done.map((d) => (d.divergent ? 3 : NaN)),
    }
  }, [done])

  const px = useAxis({ label: target.labels.x, range: target.box.x, key: key === 'hard banana' ? 'banana' : 'funnel' })
  const py = useAxis({ label: target.labels.y, range: target.box.y, key: key === 'hard banana' ? 'banana' : 'funnel' })
  const stepAxis = useAxis({ label: 'leapfrog step', range: [0, L] })
  const dhAxis = useAxis({ label: 'H − H₀', range: [-3, 3] })
  const itAxis = useAxis({ label: 'iteration', range: [0, ITERATIONS + 1] })
  const finalAxis = useAxis({ label: 'final ΔH', range: [-3, 3] })

  const divergences = done.filter((d) => d.divergent).length
  const accepted = done.filter((d) => d.accepted).length

  return (
    <Figure
      title="HMC on the funnel and a hard banana, leapfrog step by leapfrog step"
      purpose="A fixed step size that suits the funnel's mouth is unstable in its neck, so trajectories there blow up (divergences); the non-centred parameterisation turns the funnel into a Gaussian and the divergences vanish."
      defaultSize="L"
      state={state}
      controls={
        <>
          <Switch
            label="non-centred (x = z·exp(v/2))"
            checked={key === 'non-centred funnel'}
            disabled={key === 'hard banana'}
            onChange={(on: boolean) => state.set('target', on ? 'non-centred funnel' : 'centred funnel')}
          />
          <Player
            label="3 · play"
            value={at}
            onChange={setPos}
            count={frameCount + 1}
            format={(p) => (p === 0 ? 'start' : `iteration ${Math.floor((p - 1) / L) + 1}, step ${((p - 1) % L) + 1}`)}
          />
        </>
      }
      readouts={{
        'so far': (
          <>
            <Readout label="iteration" value={fi >= 0 ? `${fi + 1} of ${ITERATIONS}` : '0'} />
            <Readout label="ΔH now" value={current ? f3(current.dH[fk]) : '0'} />
            <Readout label="divergent trajectories" value={`${divergences} of ${done.length}`} />
            <Readout label="accepted" value={`${accepted} of ${done.length}`} />
          </>
        ),
      }}
      caption="Drag the start (the large dot) to rerun the chain from there. Left: log π (floored at −12) with the finished trajectories (thin), the current one up to the played leapfrog step, the samples, and a diamond where each divergent trajectory began. Top right: the energy error H − H₀ along the current trajectory; an exact flow keeps it at 0. Bottom right: the final energy error of each finished trajectory, clipped to ±3; divergent ones (|ΔH| > 1000 or non-finite) are drawn at the top in their own colour. On the centred funnel, started in the neck at v = −4.5 with ε = 0.3, ε is above the neck's stable step (about 2·exp(v/2) ≈ 0.21 there), so most trajectories gain energy and are rejected, the chain sticks, and about a quarter diverge. Switch to non-centred and every trajectory keeps |ΔH| small, is accepted, and none diverge. On the non-centred funnel the sampler runs on (v, z) and every point is mapped back to (x, v)."
    >
      <Dashboard>
        <DashboardRow minHeight={380}>
          <DashboardCell ratio={1.3}>
            <Plot x={px} y={py} renderer="canvas">
              <Raster {...surface} valueLabel="log π" colorBar={false} fillOpacity={0.75} />
              <Curve name="earlier trajectories" {...layers.past} slot={1} thin live />
              <Curve name="current trajectory" {...layers.partial} slot={2} showPoints live />
              <Points name="samples" {...layers.samples} slot={3} live />
              <Points name="divergent start" {...layers.divergent} emphasis shape={4} live />
              <Handle {...state.handle(['target.sx', 'target.sy'], { label: 'start' })} />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={2}>
              <Plot x={stepAxis} y={dhAxis} renderer="canvas">
                <Curve name="ΔH = H − H₀ along the trajectory" {...energy} slot={2} showPoints live />
                <Curve name="zero" x={[0, L]} y={[0, 0]} muted dashed />
              </Plot>
              <Plot x={itAxis} y={finalAxis} renderer="canvas">
                <Bars name="final ΔH (clipped to ±3)" x={perIteration.x} y={perIteration.ok} slot={0} live />
                <Bars name="divergent" x={perIteration.x} y={perIteration.bad} slot={3} live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
