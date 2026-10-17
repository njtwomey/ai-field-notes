import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { Params } from 'aifn-compute/foundation/pytree'
import { normals, stream } from 'aifn-compute/foundation/random'
import { float64Data, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { adamRule } from 'aifn-compute/optim/first-order'
import { moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import {
  classAndFakeProbabilities,
  complementSampler,
  generatedPoints,
  initSemiSupervisedGan,
  semiSupervisedGan,
  semiSupervisedGanTraining,
  type CriticParams,
  type FakeSource,
  type SemiSupervisedGan,
  type SemiSupervisedGanProblem,
} from 'aifn-methods/generative/gan'
import { argmaxField, grid, percent, pointsOf, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'

const BOX = { x: [-1.5, 2.5] as const, y: [-1, 1.5] as const }
const GRID = grid(BOX.x, BOX.y, 51, 33)
const SHOWN = 150
const EVERY = 10

const SOURCES = [
  { value: 'feature-matching', label: 'generator trained by feature matching' },
  { value: 'non-saturating', label: 'generator trained by the non-saturating GAN loss' },
  { value: 'complement', label: 'complement generator: uniform, away from the data' },
  { value: 'none', label: 'no generated points: labels only' },
] as const
type Source = (typeof SOURCES)[number]['value']

type Problem = SemiSupervisedGanProblem & { truth: Int32Array; rows: Float64Array; cls: number[]; allX: Tensor }

type Setup = { problem: Problem; source: Source; radius: number; steps: number; rate: number; seed: number }

type Checkpoint = { step: number; net: SemiSupervisedGan; critic: CriticParams; fakes: Float64Array; accuracy: number }

function accuracy(net: SemiSupervisedGan, critic: CriticParams, p: Problem): number {
  const q = classAndFakeProbabilities(net, critic, p.unlabelledX)
  let right = 0
  p.truth.forEach((c, i) => {
    if ((q[3 * i + 1] > q[3 * i] ? 1 : 0) === c) right++
  })
  return right / p.truth.length
}

/** Trains the semi-supervised GAN and yields a checkpoint every EVERY steps, from step 0. */
function* sganRun(s: Setup): Generator<Checkpoint> {
  const net = semiSupervisedGan({})
  const all = s.problem.allX
  const sampler = complementSampler(all, BOX, s.radius)
  const fakes: FakeSource = s.source === 'complement' ? { kind: 'fixed', sampler } : ({ kind: s.source } as FakeSource)
  const rule = () => adamRule({ stepSize: s.rate, beta1: 0.5 })
  const alg = semiSupervisedGanTraining(s.problem, {
    net,
    fakes,
    criticOptimizer: rule(),
    generatorOptimizer: rule(),
  })
  const z = normals(stream(`gssl/sgan/shown/${s.seed}`), [SHOWN, net.latent])
  const shownFixed =
    s.source === 'complement' ? Float64Array.from(float64Data(sampler(stream('gssl/sgan/shown'), SHOWN))) : null
  const it = live(alg, initSemiSupervisedGan(net, stream(`gssl/sgan/init/${s.seed}`)))
  for (let k = 0; k <= s.steps; k++) {
    const next = it.next()
    if (next.done) return
    const st = next.value.state
    const critic = st.critic as unknown as CriticParams
    if (k % EVERY === 0 || k === s.steps || st.diverged)
      yield {
        step: k,
        net,
        critic,
        fakes:
          s.source === 'none' ? new Float64Array(0) : (shownFixed ?? generatedPoints(net, st.generator as Params[], z)),
        accuracy: accuracy(net, critic, s.problem),
      }
    if (st.diverged) return
  }
}

/** A circle of radius r around a centre, as a closed polyline. */
function circle(c: readonly [number, number], r: number) {
  const t = Array.from({ length: 49 }, (_, k) => (2 * Math.PI * k) / 48)
  return { x: t.map((a) => c[0] + r * Math.cos(a)), y: t.map((a) => c[1] + r * Math.sin(a)) }
}

/**
 * A semi-supervised GAN on two moons with a few labels: the K + 1 regions of the discriminator, the generated points,
 * and the accuracy against the same classifier trained on the labels alone, for three sources of generated points.
 */
export function SganExplorer() {
  const state = useFigureState({
    source: choice(SOURCES, 'feature-matching', { label: 'generated points' }),
    perClass: int(4, { ge: 1, le: 20, label: 'labels per class', suggestions: [1, 2, 4, 10] }),
    radius: float(0.25, {
      gt: 0,
      le: 0.8,
      label: 'complement: distance from the data r',
      suggestions: [0.1, 0.25, 0.4],
      onChart: true,
    }),
    steps: int(500, { ge: 10, le: 2000, label: 'training steps', suggestions: [200, 500, 1000] }),
    rate: float(0.003, { gt: 0, le: 0.05, label: 'Adam step size', suggestions: [0.001, 0.003, 0.01] }),
    seed: int(1, { ge: 1, le: 999, label: 'data seed' }),
  })
  const source = state.source as Source
  const { perClass, radius, steps, rate, seed } = state

  const problem = useMemo(() => {
    const d = moons(stream(`gssl/sgan/moons/${seed}`), { n: 200, noise: 0.1 })
    const y = toFlat(d.y)
    const sp = semiSupervisedSplit(stream(`gssl/sgan/split/${seed}`), y, perClass, {
      x: d.x,
      draw: 'global',
      strict: true,
    })
    const cls = y.map((c) => -1 - c)
    sp.labelled.forEach((i) => (cls[i] = y[i]))
    return {
      labelledX: sp.labelledX,
      labelledY: sp.labelledY,
      unlabelledX: sp.unlabelledX,
      classes: 2,
      truth: sp.unlabelledTruth,
      rows: Float64Array.from(float64Data(d.x)),
      cls,
      allX: d.x,
    }
  }, [seed, perClass])

  // The radius matters only to the complement generator, so other runs do not restart when it moves.
  const r = source === 'complement' ? radius : 0
  const setup = useMemo<Setup>(
    () => ({ problem, source, radius: r, steps, rate, seed }),
    [problem, source, r, steps, rate, seed],
  )
  const baseSetup = useMemo<Setup>(
    () => ({ problem, source: 'none', radius: 0, steps, rate, seed }),
    [problem, steps, rate, seed],
  )
  const run = useSlicedRun(setup, sganRun)
  const base = useSlicedRun(baseSetup, sganRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const baseCp = base.checkpoints[Math.min(at, base.checkpoints.length - 1)]

  const field = useMemo(
    () => (cp ? argmaxField(classAndFakeProbabilities(cp.net, cp.critic, GRID.points), 3, GRID) : null),
    [cp],
  )
  // The share of the shown generated points that the discriminator labels "generated".
  const fakeShare = useMemo(() => {
    if (!cp || cp.fakes.length === 0) return NaN
    const n = cp.fakes.length / 2
    const p = classAndFakeProbabilities(cp.net, cp.critic, fromData(cp.fakes, [n, 2]))
    let fake = 0
    for (let i = 0; i < n; i++) if (p[3 * i + 2] > p[3 * i] && p[3 * i + 2] > p[3 * i + 1]) fake++
    return fake / n
  }, [cp])

  const centre: [number, number] = [problem.rows[0], problem.rows[1]]
  const rim = circle(centre, radius)
  const xa = useAxis({ label: 'x₁', range: BOX.x })
  const ya = useAxis({ label: 'x₂', range: BOX.y, equal: xa })

  return (
    <Figure
      title="A K + 1 discriminator on two moons"
      state={state}
      defaultSize="L"
      caption={`Two moons, 200 points with noise of sd 0.1, of which ${perClass} per class are labelled (large marks); small marks are unlabelled points coloured by their true class. The discriminator is a classifier with two hidden layers of 32 leaky-ReLU units and two logits; a third logit, fixed at 0, stands for "generated". It is trained on the cross-entropy of the labelled points plus the GAN loss: unlabelled points should look real, generated points (third colour) generated. The colours show which of the three classes the discriminator picks at each point of the plane. A generator trained by feature matching matches the mean of the discriminator's last hidden layer on real and generated points; the non-saturating generator tries to make its points look real; the complement generator does not learn and draws uniformly from the box, keeping points farther than r from every data point. Drag the handle on the dashed circle to set r. The baseline is the same classifier trained on the labels alone.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="accuracy on unlabelled points" value={percent(cp?.accuracy)} />
          <Readout label="labels only, same network" value={percent(baseCp?.accuracy)} />
          <Readout label="generated points called generated" value={percent(fakeShare)} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        {field && (
          <Raster
            x={GRID.xs}
            y={GRID.ys}
            z={field}
            scale="categorical"
            categoryNames={['class 0', 'class 1', 'generated']}
            boundary
            fillOpacity={0.35}
            stale={!run.done}
          />
        )}
        {[0, 1].map((c) => {
          const p = pointsOf(problem.rows, problem.cls, -1 - c)
          return (
            p.x.length > 0 && <Points key={`u${c}`} name={`class ${c}, unlabelled`} x={p.x} y={p.y} slot={c} thin />
          )
        })}
        {[0, 1].map((c) => {
          const p = pointsOf(problem.rows, problem.cls, c)
          return (
            p.x.length > 0 && (
              <Points key={`l${c}`} name={`class ${c}, labelled`} x={p.x} y={p.y} slot={c} emphasis size={11} />
            )
          )
        })}
        {cp && cp.fakes.length > 0 && (
          <Points
            name="generated"
            x={Array.from({ length: cp.fakes.length / 2 }, (_, i) => cp.fakes[2 * i])}
            y={Array.from({ length: cp.fakes.length / 2 }, (_, i) => cp.fakes[2 * i + 1])}
            slot={2}
            shape={2}
            thin
          />
        )}
        {source === 'complement' && <Curve name="distance r" x={rim.x} y={rim.y} emphasis dashed silent />}
        {source === 'complement' && (
          <Handle
            kind="point"
            at={[centre[0] + radius, centre[1]]}
            label="r"
            onDrag={([x, y]) => {
              const r = Math.hypot(x - centre[0], y - centre[1])
              state.set('radius', Math.round(Math.max(0.05, Math.min(0.8, r)) * 100) / 100)
            }}
          />
        )}
      </Plot>
    </Figure>
  )
}
