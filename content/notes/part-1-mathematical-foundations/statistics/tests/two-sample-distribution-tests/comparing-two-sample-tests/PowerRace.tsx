import { useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Histogram,
  int,
  Player,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  allPValues,
  ONE_D_TESTS,
  oneDimensionalPair,
  SHIFT_LABELS,
  type OneDTest,
  type ShiftKind,
} from '../_shared/twoSample'

const ALPHA = 0.05
const RESAMPLES = 99

/**
 * Power by simulation, one experiment at a time: each step draws a fresh pair of samples, runs all six tests, and
 * updates each test's share of rejections at α = 0.05. With δ = 0 the shares estimate the false-positive rate.
 */
export function PowerRace() {
  const state = useFigureState({
    kind: choice<ShiftKind>(
      (Object.keys(SHIFT_LABELS) as ShiftKind[]).map((value) => ({ value, label: SHIFT_LABELS[value] })),
      'scale',
      { label: 'how the second sample differs' },
    ),
    delta: float(0.6, { min: 0, max: 1, step: 0.01, label: 'strength δ (0: no difference)' }),
    n: int(40, { min: 10, max: 80, label: 'points in each sample', suggestions: [10, 40, 80] }),
    reps: int(100, { min: 20, max: 200, label: 'experiments', suggestions: [50, 100, 200] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const run = useComputed(
    () => {
      const root = stream(state.seed)
      return Array.from({ length: state.reps }, (_, r) => {
        const pair = oneDimensionalPair(child(root, 'experiment', r), state.n, state.n, state.kind, state.delta)
        return { ...pair, p: allPValues(child(root, 'tests', r), pair.x, pair.y, RESAMPLES) }
      })
    },
    [state.seed, state.reps, state.n, state.kind, state.delta],
    { mode: 'release' },
  )
  const experiments = run.value

  // The walk-through restarts at 0 experiments whenever the run changes.
  const [position, setPosition] = useState({ experiments, k: 0 })
  const k = position.experiments === experiments ? Math.min(position.k, experiments.length) : 0

  // Running share of rejections after 1 … k experiments, per test.
  const share = (t: OneDTest) => {
    let hits = 0
    return experiments.slice(0, k).map((e, i) => {
      if (e.p[t] <= ALPHA) hits++
      return hits / (i + 1)
    })
  }
  const steps = Array.from({ length: k }, (_, i) => i + 1)
  const last = k > 0 ? experiments[k - 1] : undefined

  const xa = useAxis({ label: 'experiments', range: [0, state.reps] })
  const ya = useAxis({ label: 'share rejected at α = 0.05', range: [0, 1] })
  const xb = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const yb = useAxis({ label: 'density', range: [0, undefined] })
  return (
    <Figure
      title="Power, one experiment at a time"
      state={state}
      caption="Each experiment draws a fresh sample of n points from N(0, 1) and n from the alternative and runs all six tests. Left: each test's share of rejections so far, which settles on its power; the dashed line is α = 0.05, the share every test should reach when δ = 0. Right: the two samples of the latest experiment. Play to run the experiments. A mean shift is found best by the t-test, Mann–Whitney and the energy test. A change of spread or shape leaves the t-test and Mann–Whitney near α while the distribution tests climb, slowly for a change of shape; raise n to see them separate."
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ experiments, k: v })}
          count={experiments.length + 1}
          label="experiments"
        />
      }
      readouts={
        <>
          <Readout label="experiments" value={k} />
          {ONE_D_TESTS.map((t) => {
            const s = share(t)
            return <Readout key={t} label={t} value={k ? formatNumber(s[k - 1]) : '—'} />
          })}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <Plot x={xa} y={ya} height={320}>
            <Curve name="α" x={[0, state.reps]} y={[ALPHA, ALPHA]} dashed muted />
            {k > 0 &&
              ONE_D_TESTS.map((t, i) => <Curve key={t} name={t} x={steps} y={share(t)} slot={i} stale={run.stale} />)}
          </Plot>
        </div>
        <Plot x={xb} y={yb} height={320}>
          {last && (
            <Histogram name="from N(0, 1)" values={last.x} bins={15} range={[-4.5, 4.5]} slot={0} opacity={0.35} />
          )}
          {last && (
            <Histogram
              name="from the alternative"
              values={last.y}
              bins={15}
              range={[-4.5, 4.5]}
              slot={1}
              opacity={0.35}
            />
          )}
        </Plot>
      </div>
    </Figure>
  )
}
