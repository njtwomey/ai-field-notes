import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { fromData } from 'aifn-compute/foundation/tensor'
import { BoundaryMap, MOON_NAMES } from '../_shared/BoundaryMap'
import { rowArgmax } from 'aifn-methods/learning/semi-supervised'
import { predictProbabilities } from 'aifn-methods/neural/semi-supervised'
import {
  byClass,
  circle,
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

const EVERY = 10
const PANEL = 300
/** Histogram bins of the confidence max_k p_k, from ½ (two classes) to 1. */
const EDGES = Array.from({ length: 21 }, (_, k) => 0.5 + k * 0.025)
const clampTo = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/**
 * FixMatch on two moons with eight labels: the pseudo-labels that pass the threshold, the weak and strong
 * perturbation radii around a probe point, and the distribution of confidences against the threshold.
 */
export function FixMatchViews() {
  const state = useFigureState({
    tau: float(0.95, { ge: 0.5, le: 0.999, label: 'threshold τ', suggestions: [0.6, 0.8, 0.95, 0.99], onChart: true }),
    weak: float(0.05, { ge: 0, le: 0.5, label: 'weak jitter sd', suggestions: [0, 0.05, 0.1] }),
    strong: float(0.3, { ge: 0, le: 0.8, label: 'strong jitter sd', suggestions: [0.05, 0.3, 0.5] }),
    steps: int(400, { ge: 10, le: 2000, label: 'training steps', suggestions: [100, 400, 800] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
    px: float(0.5, { ge: -1.5, le: 2.5, label: 'probe x₁', onChart: true }),
    py: float(0.25, { ge: -1, le: 1.5, label: 'probe x₂', onChart: true }),
  })
  const { tau, weak, strong, steps, rate, seed, px, py } = state
  const problem = useMemo(() => moonsProblem(seed), [seed])
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: 'fixmatch', threshold: tau },
      steps,
      rate,
      weight: 1,
      weak,
      strong,
      seed,
      every: EVERY,
    }),
    [problem, tau, steps, rate, weak, strong, seed],
  )
  const run = useSlicedRun(setup, consistencyRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(2), [])

  const view = useMemo(() => {
    if (!cp) return null
    const values = field(net, cp.state.params, MOONS_GRID, 2)
    const { conf } = rowArgmax(predictProbabilities(net, cp.state.params, problem.unlabelledX), 2)
    const counts = EDGES.slice(1).map(() => 0)
    conf.forEach((c) => (counts[Math.min(counts.length - 1, Math.floor((c - 0.5) / 0.025))] += 1))
    // Points used by the last step, by pseudo-label; the rest drawn muted.
    const kept = Int32Array.from(cp.state.pseudo, (c, i) => (cp.state.mask[i] > 0 ? c : -1))
    const left = Int32Array.from(cp.state.pseudo, (_, i) => (cp.step > 0 && cp.state.mask[i] === 0 ? 0 : -1))
    const probe = predictProbabilities(net, cp.state.params, fromData(new Float64Array([px, py]), [1, 2]))
    return { values, counts, kept, left, probe }
  }, [cp, net, problem, px, py])

  const ca = useAxis({ label: 'confidence max_k p_k(x)', range: [0.5, 1] })
  const na = useAxis({ label: 'unlabelled points' })
  const centres = EDGES.slice(1).map((e) => e - 0.0125)
  const top = view ? Math.max(...view.counts) : 1
  const probeConf = view ? Math.max(view.probe[0], view.probe[1]) : NaN
  const weakRim = circle([px, py], weak)
  const strongRim = circle([px, py], strong)

  return (
    <Figure
      title="FixMatch: a confident weak view labels a strong view"
      state={state}
      defaultSize="XL"
      caption={`Two moons with 8 labelled points (large marks) and 192 unlabelled ones; the network has two hidden layers of 32 tanh units and trains full batch with Adam, with unsupervised weight λ_u = 1. At every step each unlabelled point gets a weak copy (Gaussian jitter of the weak sd) and a strong copy (jitter of the strong sd). Where the network's confidence on the weak copy reaches τ, its arg max becomes a pseudo-label, and the network is trained to predict that label on the strong copy. Left: the decision boundary (black), the points used in the last step coloured by their pseudo-label, and the points below the threshold in grey. The two dashed circles around the probe mark (drag it) have the weak and strong sd as radii. Right: the confidences on the unperturbed unlabelled points; drag the vertical line to set τ. A strong sd equal to the weak sd turns FixMatch into pseudo-labelling of jittered copies: on the default seed the accuracy after 400 steps is then 86.5%, against 94.8% with a strong sd of 0.3 and 84.4% for the labelled points alone.`}
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
          <Readout label="points above τ (last step)" value={percent(cp?.state.maskRate)} />
          <Readout label="their pseudo-labels correct" value={percent(cp?.state.pseudoAccuracy)} />
          <Readout
            label="probe confidence"
            value={Number.isFinite(probeConf) ? `${probeConf.toFixed(3)} ${probeConf >= tau ? '≥' : '<'} τ` : '…'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <BoundaryMap
            problem={problem}
            grid={MOONS_GRID}
            box={MOONS_BOX}
            values={view?.values ?? null}
            hideUnlabelled={Boolean(view && cp && cp.step > 0)}
            title="pseudo-labels on the plane"
          >
            {view && cp && cp.step > 0 && (
              <>
                <Points name="below τ" {...byClass(problem.unlabelled, view.left, 0)} muted thin />
                {[0, 1].map((c) => (
                  <Points
                    key={c}
                    name={`pseudo-label: ${MOON_NAMES[c]}`}
                    {...byClass(problem.unlabelled, view.kept, c)}
                    slot={c}
                    thin
                  />
                ))}
              </>
            )}
            <Curve name="weak sd" x={weakRim.x} y={weakRim.y} emphasis dashed silent />
            <Curve name="strong sd" x={strongRim.x} y={strongRim.y} muted dashed silent />
            <Handle
              kind="point"
              at={[px, py]}
              label="probe"
              onDrag={([x, y]) => {
                state.set('px', clampTo(x, -1.5, 2.5))
                state.set('py', clampTo(y, -1, 1.5))
              }}
            />
          </BoundaryMap>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={ca} y={na} title="confidence on unlabelled points">
            {view && <Bars name="points" x={centres} y={view.counts} edges={EDGES} slot={2} />}
            <Curve name="threshold τ" x={[tau, tau]} y={[0, 1.05 * top]} emphasis silent />
            <Handle
              kind="x"
              at={tau}
              label="τ"
              onDrag={(x) => state.set('tau', Math.round(clampTo(x, 0.5, 0.999) * 1000) / 1000)}
            />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
