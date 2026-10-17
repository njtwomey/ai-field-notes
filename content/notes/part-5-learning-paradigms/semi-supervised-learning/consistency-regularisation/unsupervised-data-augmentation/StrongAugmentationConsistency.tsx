import { useMemo } from 'react'
import { choice, Curve, Figure, float, int, Player, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { BoundaryMap, MOON_NAMES } from '../_shared/BoundaryMap'
import type { AnnealingSchedule } from 'aifn-methods/neural/semi-supervised'
import {
  byClass,
  classifier,
  consistencyRun,
  field,
  moonsProblem,
  MOONS_BOX,
  MOONS_GRID,
  percent,
  type RunSetup,
} from '../_shared/consistency-runs'
import { FixedHeight, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'

const EVERY = 5
const PANEL = 250
/** The perturbation of the comparison run: a weak jitter in place of the strong one. */
const WEAK_SD = 0.05
const SCHEDULES = [
  { value: 'none', label: 'no annealing' },
  { value: 'log', label: 'log schedule' },
  { value: 'linear', label: 'linear schedule' },
  { value: 'exp', label: 'exp schedule' },
] as const

/**
 * UDA on two moons with eight labels: the consistency target from the clean point against a strongly perturbed copy,
 * compared with the same run on a weakly perturbed copy, with optional training-signal annealing.
 */
export function StrongAugmentationConsistency() {
  const state = useFigureState({
    strong: float(0.3, { ge: 0, le: 0.8, label: 'strong jitter sd', suggestions: [0.15, 0.3, 0.5] }),
    annealing: choice(SCHEDULES, 'none', { label: 'training-signal annealing' }),
    temperature: float(0.4, { gt: 0, le: 1, label: 'sharpening temperature τ', suggestions: [0.4, 0.7, 1] }),
    confidence: float(0.8, { ge: 0, le: 0.99, label: 'confidence mask β', suggestions: [0, 0.6, 0.8] }),
    steps: int(400, { ge: 10, le: 2000, label: 'training steps', suggestions: [100, 400, 800] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { strong, annealing, temperature, confidence, steps, rate, seed } = state
  const problem = useMemo(() => moonsProblem(seed), [seed])
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: 'uda', temperature, confidence, annealing: annealing as AnnealingSchedule },
      steps,
      rate,
      weight: 1,
      weak: 0,
      strong,
      seed,
      every: EVERY,
    }),
    [problem, temperature, confidence, annealing, steps, rate, strong, seed],
  )
  const weakSetup = useMemo<RunSetup>(() => ({ ...setup, strong: WEAK_SD }), [setup])
  const run = useSlicedRun(setup, consistencyRun)
  const weakRun = useSlicedRun(weakSetup, consistencyRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(2), [])

  const view = useMemo(() => {
    if (!cp) return null
    const used = Int32Array.from(cp.state.pseudo, (c, i) => (cp.step > 0 && cp.state.mask[i] > 0 ? c : -1))
    const unused = Int32Array.from(cp.state.pseudo, (_, i) => (cp.step > 0 && cp.state.mask[i] === 0 ? 0 : -1))
    return { values: field(net, cp.state.params, MOONS_GRID, 2), used, unused }
  }, [cp, net])

  const curves = useMemo(() => {
    const series = (cps: typeof run.checkpoints) => ({
      step: cps.map((c) => c.step),
      accuracy: cps.map((c) => c.accuracy),
      loss: cps.filter((c) => c.step > 0).map((c) => c.state.unsupervisedLoss),
      lossStep: cps.filter((c) => c.step > 0).map((c) => c.step),
      kept: cps.filter((c) => c.step > 0).map((c) => c.state.labelledKept),
    })
    return { strong: series(run.checkpoints), weak: series(weakRun.checkpoints) }
  }, [run.checkpoints, weakRun.checkpoints])
  const weakEnd = weakRun.done ? weakRun.checkpoints[weakRun.checkpoints.length - 1] : undefined
  const stepNow = cp?.step ?? 0
  const topLoss = Math.max(
    0.05,
    ...curves.strong.loss.filter(Number.isFinite),
    ...curves.weak.loss.filter(Number.isFinite),
  )

  const sa = useAxis({ label: 'step', range: [0, steps] })
  const aa = useAxis({ label: 'accuracy on unlabelled points', range: [0.5, 1] })
  const sb = useAxis({ label: 'step', range: [0, steps] })
  const la = useAxis({ label: 'consistency loss; share of labels kept', range: [0, Math.max(1, topLoss)] })

  return (
    <Figure
      title="Consistency against a strong perturbation"
      state={state}
      defaultSize="XL"
      caption={`Two moons with 8 labelled points (large marks) and 192 unlabelled ones; two hidden layers of 32 tanh units, trained full batch with Adam for both runs from the same start. UDA's target for each unlabelled point is the network's prediction at the clean point, sharpened by the temperature τ and kept only where the unsharpened confidence exceeds β; the network is trained to match it on a perturbed copy with Gaussian jitter. The main run uses the strong jitter sd; the comparison run uses a weak jitter of sd ${WEAK_SD}. Left: the main run's boundary, with the points inside the confidence mask coloured by the target's class and the rest grey. Middle: accuracy on the unlabelled points for both runs. Right: the consistency loss for both runs, and the share of labelled points that training-signal annealing keeps in the supervised loss. On the default seed the weak copy leaves the boundary where the labels put it; the strong copy moves it into the gap. Annealing withholds most labels for most of a short run, and on this problem it slows learning.`}
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
          <Readout label="accuracy, strong copy" value={percent(cp?.accuracy)} />
          <Readout label="accuracy, weak copy, at the end" value={percent(weakEnd?.accuracy)} />
          <Readout label="points inside the mask" value={percent(cp?.state.maskRate)} />
          <Readout label="labels kept by annealing" value={percent(cp?.state.labelledKept)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <FixedHeight height={PANEL}>
          <BoundaryMap
            problem={problem}
            grid={MOONS_GRID}
            box={MOONS_BOX}
            values={view?.values ?? null}
            hideUnlabelled={Boolean(cp && cp.step > 0)}
            title="main run"
          >
            {view && cp && cp.step > 0 && (
              <>
                <Points name="outside the mask" {...byClass(problem.unlabelled, view.unused, 0)} muted thin />
                {[0, 1].map((c) => (
                  <Points
                    key={c}
                    name={`target: ${MOON_NAMES[c]}`}
                    {...byClass(problem.unlabelled, view.used, c)}
                    slot={c}
                    thin
                  />
                ))}
              </>
            )}
          </BoundaryMap>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sa} y={aa} title="accuracy">
            <Curve name="strong copy" x={curves.strong.step} y={curves.strong.accuracy} slot={2} />
            <Curve name="weak copy" x={curves.weak.step} y={curves.weak.accuracy} slot={3} dashed />
            <Curve name="checkpoint" x={[stepNow, stepNow]} y={[0.5, 1]} muted dashed silent />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sb} y={la} title="loss and labels kept">
            <Curve name="consistency loss, strong" x={curves.strong.lossStep} y={curves.strong.loss} slot={2} />
            <Curve name="consistency loss, weak" x={curves.weak.lossStep} y={curves.weak.loss} slot={3} dashed />
            <Curve name="labels kept" x={curves.strong.lossStep} y={curves.strong.kept} emphasis thin />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
