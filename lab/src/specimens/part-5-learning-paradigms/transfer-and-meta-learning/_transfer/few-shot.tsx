/**
 * Few-shot learning and label shift (`aifn-methods/learning/transfer`): prototypical networks trained episodically on
 * direction classes, with the nearest-prototype map of the input plane and the embedding; and BBSE against prior-shift
 * EM estimating the target's class priors as the shift grows.
 */
import { useMemo } from 'react'
import { labelShiftDomains } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import {
  blackBoxShiftEstimate,
  priorShiftEm,
  reweightPosteriors,
  type PrototypicalRun,
} from 'aifn-methods/learning/transfer'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, take, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, slider, useComputed, useFigureState, type Task } from 'aifn-render/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'
import { fmt, usePicked } from './shared'

const rowsOf = (v: ArrayLike<number>, g: number) =>
  Array.from({ length: g }, (_, i) => Array.from({ length: g }, (_, j) => v[i * g + j]))
const xs = (v: ArrayLike<number>) => Array.from({ length: v.length / 2 }, (_, i) => v[2 * i])
const ys = (v: ArrayLike<number>) => Array.from({ length: v.length / 2 }, (_, i) => v[2 * i + 1])

// ── Prototypical networks ───────────────────────────────────────────────────────────────────────────────────────────

type ProtoSettings = Record<string, number>
const protoTask = (options: ProtoSettings): Task<PrototypicalRun> =>
  call<PrototypicalRun>('applied/learning/transfer/prototypicalRun', options)

export function PrototypicalSpecimen() {
  const state = useFigureState({
    episodes: row('1 · episodes', {
      ways: int(5, { ge: 2, le: 8, suggestions: [3, 5, 8], label: 'ways N (test)' }),
      shots: int(1, { ge: 1, le: 20, suggestions: [1, 5], label: 'shots K' }),
      trainWays: int(8, { ge: 2, le: 20, suggestions: [5, 8, 15], label: 'ways while training' }),
      noise: float(0.06, { ge: 0, suggestions: [0.03, 0.06, 0.12], label: 'angular noise (radians)' }),
    }),
    training: row('2 · training', {
      count: int(1000, { ge: 1, suggestions: [300, 1000, 3000], label: 'episodes' }),
      rate: float(3e-3, { gt: 0, scale: 'log10', suggestions: [1e-3, 3e-3, 1e-2], label: 'Adam step size' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    view: row('3 · map', {
      map: choice(['learned embedding', 'raw inputs'], 'learned embedding', { label: 'prototypes compared in' }),
    }),
  })
  const settings: ProtoSettings = {
    ways: state.episodes.ways,
    shots: state.episodes.shots,
    trainWays: state.episodes.trainWays,
    angularNoise: state.episodes.noise,
    episodes: state.training.count,
    stepSize: state.training.rate,
    seed: state.training.seed,
  }
  const trained = useTrainedRun(settings, protoTask)
  const run = trained.run.value
  const shots = run?.checkpoints ?? []
  const [index, pick] = usePicked(trained.trained, shots.length)
  const shot = shots[index]
  const raw = state.view.map === 'raw inputs'
  const key = JSON.stringify(trained.trained)
  const g = run?.gridX.length ?? 0
  const map = useMemo(() => (run && shot && g ? rowsOf(raw ? run.rawMap : shot.map, g) : null), [run, shot, g, raw])
  const ways = run?.ways ?? state.episodes.ways
  const names = Array.from({ length: ways }, (_, c) => `class ${c + 1}`)
  const x1 = useAxis({ label: 'x₁', range: [-3.2, 3.2] })
  const x2 = useAxis({ label: 'x₂', range: [-3.2, 3.2], equal: x1 })
  const e1 = useAxis({ label: 'embedding 1', hold: 'union', key })
  const e2 = useAxis({ label: 'embedding 2', hold: 'union', key, equal: e1 })
  const epAxis = useAxis({ label: 'training episode', range: [0, run?.episodes ?? state.training.count], key })
  const accAxis = useAxis({ label: 'query accuracy', range: [0, 1] })
  const pickEpisode = (v: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.episode - v) < Math.abs(shots[best].episode - v)) best = i
    })
    pick(best)
  }
  return (
    <Figure
      title="Prototypical networks: few-shot classes by distance to a mean"
      purpose="A prototypical network classifies a query by the nearest class mean of a few labelled examples in a learned embedding; training on a new episode of new classes every step teaches a metric that ignores what does not identify a class, here the distance from the origin."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? run.done / run.episodes : 0}
            progressText={run ? `${run.done} / ${run.episodes} episodes` : ''}
          />
          <ControlRow label="4 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `episode ${shots[i]?.episode ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="episode" value={shot ? shot.episode : '—'} />
          <Readout label="test accuracy (embedding)" value={shot ? fmt(shot.accuracy) : '—'} />
          <Readout label="test accuracy (raw inputs)" value={run ? fmt(run.rawAccuracy) : '—'} />
        </>
      }
      caption="aifn fewShotEpisode and prototypicalRun: each class is a direction from the origin (angle plus Gaussian noise) at a radius uniform on [0.3, 3], and every episode draws new classes. A 2 → 32 → 32 → 2 tanh embedding is trained by Adam on one episode per step (the cross-entropy of softmax(−‖f(x) − c_k‖²) over its queries, 5 per class). Left: a fixed test episode, its support points (large, class slots) and queries (small), over the class of the nearest prototype for each point of the plane, compared in the learned embedding or in the raw inputs. Middle: the episode in the embedding, prototypes in ink. Right: accuracy over 100 unseen test episodes at the checkpoints (ink; class colours are kept for the classes), against raw-input prototypes (dashed), and each training episode's query accuracy (faint). Press Train; play the checkpoints or drag the episode marker."
    >
      <Plots cols={3}>
        <Plot x={x1} y={x2} title={!trained.trained ? 'press Train to start' : 'nearest prototype in the plane'}>
          {map && run && (
            <Raster
              x={run.gridX}
              y={run.gridX}
              z={map}
              scale="categorical"
              categoryNames={names}
              fillOpacity={0.35}
              boundary
            />
          )}
          {run && (
            <Points
              name="support"
              x={xs(run.demo.support)}
              y={ys(run.demo.support)}
              group={Array.from(run.demo.supportY)}
              groupNames={names}
              size={9}
            />
          )}
          {run && (
            <Points
              name="queries"
              x={xs(run.demo.query)}
              y={ys(run.demo.query)}
              group={Array.from(run.demo.queryY)}
              thin
              size={4}
            />
          )}
        </Plot>
        <Plot x={e1} y={e2} title="the episode in the embedding">
          {shot && run && (
            <Points
              name="queries"
              x={xs(shot.query)}
              y={ys(shot.query)}
              group={Array.from(run.demo.queryY)}
              thin
              size={4}
            />
          )}
          {shot && run && (
            <Points
              name="support"
              x={xs(shot.support)}
              y={ys(shot.support)}
              group={Array.from(run.demo.supportY)}
              size={8}
            />
          )}
          {shot && (
            <Points
              name="prototypes"
              x={xs(shot.prototypes)}
              y={ys(shot.prototypes)}
              emphasis
              size={10}
              labels={names.map((_, c) => String(c + 1))}
            />
          )}
        </Plot>
        <Plot x={epAxis} y={accAxis} title="accuracy on unseen classes">
          {run && <Curve name="training episode" muted thin x={run.history.episode} y={run.history.accuracy} />}
          {run && (
            <Curve
              name="raw-input prototypes"
              x={[0, run.episodes]}
              y={[run.rawAccuracy, run.rawAccuracy]}
              muted
              dashed
            />
          )}
          {run && (
            <Curve
              name="test episodes"
              emphasis
              x={shots.map((c) => c.episode)}
              y={shots.map((c) => c.accuracy)}
              showPoints
            />
          )}
          {shot && <Handle kind="x" at={shot.episode} onDrag={pickEpisode} label={`episode ${shot.episode}`} />}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Label shift ─────────────────────────────────────────────────────────────────────────────────────────────────────

const SKEWS = {
  'class 1 dominates (0.8, 0.15, 0.05)': [0.8, 0.15, 0.05],
  'class 3 rare (0.45, 0.45, 0.1)': [0.45, 0.45, 0.1],
  'class 3 dominates (0.1, 0.2, 0.7)': [0.1, 0.2, 0.7],
}
const UNIFORM = [1 / 3, 1 / 3, 1 / 3]
const SHIFTS = Array.from({ length: 11 }, (_, i) => i / 10)
const K = 3
const tv = (p: ArrayLike<number>, q: ArrayLike<number>) =>
  0.5 * Array.from(p).reduce((t, v, c) => t + Math.abs(v - q[c]), 0)
const argmaxRows = (p: ArrayLike<number>) =>
  Int32Array.from({ length: p.length / K }, (_, i) => {
    let best = 0
    for (let c = 1; c < K; c++) if (p[i * K + c] > p[i * K + best]) best = c
    return best
  })

type ShiftSweep = {
  truth: number[][]
  bbse: number[][]
  em: number[][]
  error: { bbse: number[]; em: number[]; none: number[] }
  accuracy: { none: number[]; em: number[]; oracle: number[] }
}

/** BBSE and EM against the truth at each shift s: target priors (1 − s)·uniform + s·skew, averaged over trials. */
function labelShiftSweep(
  skew: number[],
  separation: number,
  sourceSize: number,
  n: number,
  trials: number,
  seed: number,
): ShiftSweep {
  // The source: a classifier fitted on one half, the confusion matrix from the other.
  const src = labelShiftDomains(child(stream(seed), 'source'), { n: 2 * sourceSize, separation }).source
  const half = Array.from({ length: sourceSize }, (_, i) => i)
  const rows = (t: Tensor, ids: number[]) => unwrap(take(t, ids)) as Tensor
  const model = logisticRegression({ multinomial: true }).fit(dataset(rows(src.x, half), rows(src.y, half)))
  const posteriors = (x: Tensor) => {
    const z = toFlat(model.forward(x) as Tensor)
    const out = new Float64Array(z.length)
    for (let i = 0; i < z.length / K; i++) {
      const m = Math.max(z[i * K], z[i * K + 1], z[i * K + 2])
      let t = 0
      for (let c = 0; c < K; c++) t += out[i * K + c] = Math.exp(z[i * K + c] - m)
      for (let c = 0; c < K; c++) out[i * K + c] /= t
    }
    return out
  }
  // EM and the correction divide by the priors the classifier was fitted under: its training labels' frequencies.
  const fitted = [0, 0, 0]
  for (const v of toFlat(rows(src.y, half))) fitted[v] += 1 / half.length
  const held = half.map((i) => i + sourceSize)
  const heldY = toFlat(rows(src.y, held))
  const heldPred = argmaxRows(posteriors(rows(src.x, held)))
  const out: ShiftSweep = {
    truth: [],
    bbse: [],
    em: [],
    error: { bbse: [], em: [], none: [] },
    accuracy: { none: [], em: [], oracle: [] },
  }
  SHIFTS.forEach((sh, step) => {
    const truth = UNIFORM.map((u, c) => (1 - sh) * u + sh * skew[c])
    const b = [0, 0, 0]
    const e = [0, 0, 0]
    const err = { bbse: 0, em: 0, none: 0 }
    const acc = { none: 0, em: 0, oracle: 0 }
    for (let t = 0; t < trials; t++) {
      const tgt = labelShiftDomains(child(stream(seed), 'target', step, t), {
        n,
        separation,
        targetPriors: truth,
      }).target
      const y = toFlat(tgt.y)
      const p = posteriors(tgt.x)
      const bb = blackBoxShiftEstimate(heldY, heldPred, argmaxRows(p), K).priors
      const em = priorShiftEm(fromData(p, [n, K]), fitted).priors
      const score = (q: ArrayLike<number>) => argmaxRows(q).reduce((h, v, i) => h + (v === y[i] ? 1 : 0), 0) / n
      for (let c = 0; c < K; c++) {
        b[c] += bb[c] / trials
        e[c] += em[c] / trials
      }
      err.bbse += tv(bb, truth) / trials
      err.em += tv(em, truth) / trials
      err.none += tv(UNIFORM, truth) / trials
      acc.none += score(p) / trials
      acc.em += score(reweightPosteriors(fromData(p, [n, K]), fitted, em)) / trials
      acc.oracle += score(reweightPosteriors(fromData(p, [n, K]), fitted, truth)) / trials
    }
    out.truth.push(truth)
    out.bbse.push(b)
    out.em.push(e)
    out.error.bbse.push(err.bbse)
    out.error.em.push(err.em)
    out.error.none.push(err.none)
    out.accuracy.none.push(acc.none)
    out.accuracy.em.push(acc.em)
    out.accuracy.oracle.push(acc.oracle)
  })
  return out
}

export function LabelShiftSpecimen() {
  const state = useFigureState({
    domains: row('1 · domains', {
      skew: choice(Object.keys(SKEWS) as (keyof typeof SKEWS)[], 'class 1 dominates (0.8, 0.15, 0.05)', {
        label: 'target priors at full shift',
      }),
      separation: slider(0.5, 5, 2, { step: 0.1, label: 'class separation' }),
      shift: slider(0, 1, 0.6, { step: 0.1, label: 'shift s shown' }),
    }),
    sample: row('2 · samples', {
      source: int(2000, {
        ge: 50,
        le: 20000,
        suggestions: [200, 2000, 10000],
        label: 'labelled source points (each half)',
      }),
      n: int(1000, { ge: 50, le: 20000, suggestions: [200, 1000, 5000], label: 'unlabelled target points' }),
      trials: int(5, { ge: 1, le: 50, suggestions: [1, 5, 20], label: 'trials per shift' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { skew, separation, shift } = state.domains
  const { source, n, trials, seed } = state.sample
  const sweep = useComputed(
    () => labelShiftSweep(SKEWS[skew], separation, source, n, trials, seed),
    [skew, separation, source, n, trials, seed],
    { mode: 'release' },
  )
  const r = sweep.value
  const at = Math.round(shift * 10)
  const sAxis = useAxis({ label: 'shift s (0 = source priors)', range: [0, 1] })
  const pAxis = useAxis({ label: 'target prior', range: [0, 1] })
  const eAxis = useAxis({ label: 'total-variation error of the priors', hold: 'union', key: skew })
  const aAxis = useAxis({ label: 'target accuracy', hold: 'union', key: `${skew}/${separation}` })
  const setShift = (v: number) => state.set('domains.shift', Math.min(1, Math.max(0, Math.round(v * 10) / 10)))
  const fmtP = (p: number[] | undefined) => (p ? p.map((v) => fmt(v, 2)).join(', ') : '—')
  return (
    <Figure
      title="Label shift: estimating the target's class priors"
      purpose="When only the class priors change between domains, a source classifier's predictions on unlabelled target data reveal the new priors: BBSE inverts the source confusion matrix, EM re-weights the posteriors until the priors are self-consistent, and either estimate corrects the classifier."
      state={state}
      defaultSize="XL"
      readouts={
        <>
          <Readout label="true priors" value={fmtP(r?.truth[at])} />
          <Readout label="BBSE" value={fmtP(r?.bbse[at])} />
          <Readout label="EM" value={fmtP(r?.em[at])} />
          <Readout
            label="accuracy uncorrected → EM-corrected"
            value={r ? `${fmt(r.accuracy.none[at])} → ${fmt(r.accuracy.em[at])}` : '—'}
          />
        </>
      }
      caption="aifn labelShiftDomains (three unit-variance Gaussian classes with the same class-conditionals in both domains) with target priors (1 − s)·uniform + s·(the chosen skew); a multinomial logistic regression fitted on the chosen number of uniform-prior source points, its confusion matrix from as many more (one source sample, so its errors bias every estimate the same way; more source points shrink them); EM and the correction divide by the class frequencies of its training half. At each s, blackBoxShiftEstimate and priorShiftEm estimate the target priors from the unlabelled target sample, averaged over the trials. Left: the true priors (lines, class slots) against BBSE (circles) and EM (squares). Middle: total-variation error of each estimate and of assuming no shift (dashed). Right: target accuracy of the source classifier, of its posteriors re-weighted by the EM priors (reweightPosteriors), and by the true priors. Drag the shift marker on any panel."
    >
      <Plots cols={3}>
        <Plot x={sAxis} y={pAxis} title="target priors">
          {r &&
            [0, 1, 2].map((c) => (
              <Curve key={`t${c}`} name={`class ${c + 1}`} slot={c} x={SHIFTS} y={r.truth.map((p) => p[c])} />
            ))}
          {r &&
            [0, 1, 2].map((c) => (
              <Points
                key={`b${c}`}
                name={`BBSE ${c + 1}`}
                slot={c}
                shape={0}
                size={7}
                x={SHIFTS}
                y={r.bbse.map((p) => p[c])}
              />
            ))}
          {r &&
            [0, 1, 2].map((c) => (
              <Points
                key={`e${c}`}
                name={`EM ${c + 1}`}
                slot={c}
                shape={1}
                size={7}
                x={SHIFTS}
                y={r.em.map((p) => p[c])}
              />
            ))}
          <Handle kind="x" at={shift} onDrag={setShift} label={`s = ${fmt(shift, 2)}`} />
        </Plot>
        <Plot x={sAxis} y={eAxis} title="error in the priors">
          {r && <Curve name="no correction" x={SHIFTS} y={r.error.none} emphasis dashed />}
          {r && <Curve name="BBSE" slot={3} x={SHIFTS} y={r.error.bbse} showPoints />}
          {r && <Curve name="EM" slot={4} x={SHIFTS} y={r.error.em} showPoints />}
          <Handle kind="x" at={shift} onDrag={setShift} label={`s = ${fmt(shift, 2)}`} />
        </Plot>
        <Plot x={sAxis} y={aAxis} title="target accuracy">
          {r && <Curve name="uncorrected" slot={3} x={SHIFTS} y={r.accuracy.none} showPoints />}
          {r && <Curve name="EM-corrected" slot={4} x={SHIFTS} y={r.accuracy.em} showPoints />}
          {r && <Curve name="true priors" x={SHIFTS} y={r.accuracy.oracle} emphasis dashed />}
          <Handle kind="x" at={shift} onDrag={setShift} label={`s = ${fmt(shift, 2)}`} />
        </Plot>
      </Plots>
    </Figure>
  )
}
