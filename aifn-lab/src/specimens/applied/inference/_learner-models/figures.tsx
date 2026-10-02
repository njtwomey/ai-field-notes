/**
 * Zero-inflated learner models (Twomey et al., 2022) on simulated students: IRT, a linear knowledge-tracing machine and
 * IRT-ZILM fitted to the same responses in the worker, with ability estimates against the truth by number of
 * neurodivergent conditions, the bias by group, and the posterior that each zero is structural; and an equity sweep
 * over the zero-inflation rate. Data from `aifn-applied/data/synthetic` `learnerResponses`; fits and measures from
 * `aifn-applied/inference/learner-models`.
 */
import { useState } from 'react'
import type {
  LearnerModelKind,
  LearnerModelResult,
  LearnerModelRunResult,
  LearnerSweepResult,
} from 'aifn-applied/inference/learner-models'
import { Figure } from '@lab/layout'
import { call, choice, int, row, slider, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Bars, Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number | undefined, digits = 3) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—'

/** Model names and fixed colour slots (groups use 0–2, zero kinds 6–7). */
const MODELS: Record<LearnerModelKind, { name: string; slot: number }> = {
  irt: { name: 'IRT (2PL)', slot: 3 },
  ktm: { name: 'linear KTM', slot: 4 },
  zilm: { name: 'IRT-ZILM', slot: 5 },
}
const ORDER: readonly LearnerModelKind[] = ['irt', 'ktm', 'zilm']
const GROUPS = ['no condition', 'dyslexia', 'dyscalculia', 'SPD', 'two or more']
const COUNTS = ['no condition', 'one condition', 'two or more']
const ZEROS = ['structural (context)', 'incorrect (ability)']

/** The learners' settings, shared by both figures. */
const learners = (label: string) =>
  row(label, {
    students: int(300, { ge: 20, le: 2000, suggestions: [100, 300, 600], label: 'students' }),
    items: int(60, { ge: 5, le: 300, suggestions: [30, 60, 120], label: 'items in the bank' }),
    attempts: int(20, { ge: 2, le: 300, suggestions: [10, 20, 40], label: 'items each student attempts' }),
    spread: slider(0.5, 3, 2, { step: 0.1, label: 'difficulty spread (b ~ U(−s, s))' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })
const fitting = (label: string) =>
  row(label, {
    maxSteps: int(1000, { ge: 1, le: 5000, suggestions: [100, 300, 1000, 3000], label: 'L-BFGS steps per fit' }),
  })

type Learners = { students: number; items: number; attempts: number; spread: number; seed: number }
const dataTask = (d: Learners, inflation: Record<string, number>, baseRate: number) =>
  call('applied/data/synthetic/learnerResponses', call('foundation/random/stream', `zilm/${d.seed}`), {
    students: d.students,
    items: d.items,
    attempts: Math.min(d.attempts, d.items),
    difficultySpread: d.spread,
    inflation,
    baseRate,
  })

// ── 1 · Ability without the zeros' bias ──────────────────────────────────────────────────────────────────────────────

type RunSettings = { data: Learners; inflation: Record<string, number>; baseRate: number; maxSteps: number }
const runTask = (s: RunSettings): Task<LearnerModelRunResult> =>
  call<LearnerModelRunResult>(
    'applied/inference/learner-models/learnerModelRun',
    dataTask(s.data, s.inflation, s.baseRate),
    { seed: s.data.seed, maxSteps: s.maxSteps },
  )

export function ZeroInflatedAbilitySpecimen() {
  const state = useFigureState({
    data: learners('1 · students and items'),
    inflation: row('2 · zero-inflation rate on a fully unsuitable item', {
      dyslexia: slider(0, 0.95, 0.6, { step: 0.05, label: 'dyslexia r₁' }),
      dyscalculia: slider(0, 0.95, 0.6, { step: 0.05, label: 'dyscalculia r₂' }),
      spd: slider(0, 0.95, 0.6, { step: 0.05, label: 'SPD r₃' }),
      baseRate: slider(0.001, 0.2, 0.02, { step: 0.001, label: 'every student, suitable item π₀' }),
    }),
    fit: fitting('3 · fit'),
    show: row('4 · compare', {
      baseline: choice(
        [
          { value: 'irt', label: 'IRT (2PL)' },
          { value: 'ktm', label: 'linear KTM' },
        ],
        'irt',
        { label: 'baseline in the first scatter' },
      ),
    }),
  })
  const { dyslexia, dyscalculia, spd, baseRate } = state.inflation
  const settings: RunSettings = {
    data: state.data,
    inflation: { dyslexia, dyscalculia, spd },
    baseRate,
    maxSteps: state.fit.maxSteps,
  }
  const trained = useTrainedRun(settings, runTask)
  const run = trained.run.value
  const key = JSON.stringify(trained.trained)
  const of = (m: LearnerModelKind): LearnerModelResult | undefined => run?.results.find((r) => r.model === m)
  const baseline = of(state.show.baseline as LearnerModelKind)
  const zilm = of('zilm')
  const counts = run ? Array.from(run.truth.conditionCount, (c) => Math.min(2, c)) : []
  const truthAbility = run ? Array.from(run.truth.ability) : []
  const tAxis = useAxis({ label: 'true ability θ', hold: 'union', key })
  const eAxis = useAxis({ label: 'estimated ability θ̂', hold: 'union', key, equal: tAxis })
  const eAxis2 = useAxis({ label: 'estimated ability θ̂', hold: 'union', key, equal: tAxis })
  const tAxis2 = useAxis({ label: 'true ability θ', hold: 'union', key })
  const gAxis = useAxis({ label: 'group', categories: GROUPS })
  const bAxis = useAxis({ label: 'mean θ̂ − θ (bias)', hold: 'union', key })
  const rAxis = useAxis({ label: 'true θ − b of the student and item', hold: 'union', key })
  const pAxis = useAxis({ label: 'posterior that the zero is structural', range: [0, 1] })
  const diagonal = (r: LearnerModelResult | undefined) => {
    if (!r || !run) return null
    const lo = Math.min(...truthAbility, ...r.fit.ability)
    const hi = Math.max(...truthAbility, ...r.fit.ability)
    return <Curve name="θ̂ = θ" x={[lo, hi]} y={[lo, hi]} emphasis dashed thin />
  }
  const zeros = zilm?.zeros
  const zeroX =
    zeros && run ? Array.from(zeros.student, (p, j) => run.truth.ability[p] - run.truth.difficulty[zeros.item[j]]) : []
  const zeroKind = zeros ? Array.from(zeros.structural, (s) => (s ? 0 : 1)) : []
  const progress = run ? run.results.length / Math.max(1, run.total) : 0
  return (
    <Figure
      title="Ability estimates with and without zero inflation"
      purpose="A learner model reads every zero as low ability. When an item's delivery or response type is unsuitable for a student's condition, some zeros come from the context instead, and IRT underestimates those students. IRT-ZILM gives each zero a second cause, a structural zero with probability π that depends on the student's conditions and the item's format, and its ability estimates lose most of the bias."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls
          run={trained as never}
          progress={progress}
          progressText={run ? `${run.results.length} / ${run.total} models` : ''}
        />
      }
      readouts={
        <>
          {ORDER.map((m) => {
            const r = of(m)
            return (
              <Readout
                key={m}
                label={`${MODELS[m].name}: equity gap · ability RMSE · held-out NLL`}
                value={r ? `${fmt(r.equity.gap)} · ${fmt(r.equity.rmse)} · ${fmt(r.test.nll)}` : '—'}
              />
            )
          })}
          <Readout label="IRT-ZILM π₀ (fitted)" value={zilm ? fmt(1 / (1 + Math.exp(-zilm.fit.intercept))) : '—'} />
          <Readout label="structural-zero AUROC" value={fmt(zeros?.auroc)} />
          {trained.run.running && <Readout label="fitting" value={<span aria-busy="true">in a worker…</span>} />}
        </>
      }
      caption="aifn learnerResponses simulates the paper's setting (Table 1): θ ~ N(0, 1) independent of the conditions (dyslexia 10%, dyscalculia 6%, SPD 11%), items with b ~ U(−s, s), a ~ U(0.5, 4), guessing c ~ U(0, 0.15) and a random delivery, response and content type; a response is a structural zero with probability π, where logit π = logit π₀ + Σₖ zₖ uₖ(item)(logit rₖ − logit π₀) and uₖ ∈ [0, 1] is how unsuitable the item's format is for condition k. learnerModelRun holds out 20% of the answers, then fits IRT, a linear KTM (the same condition × format features added to the logit) and IRT-ZILM by penalised joint maximum likelihood (autodiff gradients, L-BFGS). Top: estimated against true ability for the chosen baseline and for IRT-ZILM, coloured by the student's number of conditions. Bottom left: the mean signed error of θ̂ in each group; the equity gap is the mean error of students with a condition minus that of students without. Bottom right: every observed zero by how far the student's true ability is above the item's difficulty, against IRT-ZILM's posterior π/(π + (1 − π)(1 − p)) that it is structural, coloured by what it truly was. Press Train; hover a point for its values."
    >
      <Plots cols={2}>
        <Plot
          x={tAxis}
          y={eAxis}
          title={
            !trained.trained
              ? 'press Train to start'
              : `${MODELS[(baseline?.model ?? state.show.baseline) as LearnerModelKind].name}`
          }
        >
          {diagonal(baseline)}
          {baseline && (
            <Points
              name="students"
              x={truthAbility}
              y={Array.from(baseline.fit.ability)}
              group={counts}
              groupNames={COUNTS}
              size={5}
            />
          )}
        </Plot>
        <Plot x={tAxis2} y={eAxis2} title="IRT-ZILM">
          {diagonal(zilm)}
          {zilm && (
            <Points
              name="students"
              x={truthAbility}
              y={Array.from(zilm.fit.ability)}
              group={counts}
              groupNames={COUNTS}
              size={5}
            />
          )}
        </Plot>
        <Plot x={gAxis} y={bAxis} title="ability bias by group">
          {ORDER.map((m, j) => {
            const r = of(m)
            return r ? (
              <Bars
                key={m}
                name={MODELS[m].name}
                slot={MODELS[m].slot}
                x={r.equity.groups.map((g) => g.group + (j - 1) * 0.27)}
                y={r.equity.groups.map((g) => (Number.isFinite(g.bias) ? g.bias : 0))}
                width={0.25}
              />
            ) : null
          })}
          {run && <Curve name="no bias" x={[-0.5, GROUPS.length - 0.5]} y={[0, 0]} muted dashed thin />}
        </Plot>
        <Plot x={rAxis} y={pAxis} title="which zeros came from the context">
          {zeros &&
            ZEROS.map((label, kind) => {
              const pick = zeroKind.flatMap((k, j) => (k === kind ? [j] : []))
              return (
                <Points
                  key={label}
                  name={label}
                  slot={6 + kind}
                  x={pick.map((j) => zeroX[j])}
                  y={pick.map((j) => zeros.posterior[j])}
                  shape={kind}
                  size={4}
                  thin
                />
              )
            })}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · Equity as the zero-inflation gap grows ───────────────────────────────────────────────────────────────────────

type SweepSettings = { data: Learners; rates: number[]; baseRate: number; maxSteps: number }
const sweepTask = (s: SweepSettings): Task<LearnerSweepResult> =>
  call<LearnerSweepResult>(
    'applied/inference/learner-models/learnerEquitySweep',
    s.rates.map((r) => dataTask(s.data, { dyslexia: r, dyscalculia: r, spd: r }, s.baseRate)),
    s.rates,
    { seed: s.data.seed, maxSteps: s.maxSteps },
  )

export function LearnerEquitySweepSpecimen() {
  const state = useFigureState({
    data: learners('1 · students and items'),
    sweep: row('2 · zero-inflation rates swept', {
      top: slider(0.1, 0.95, 0.8, { step: 0.05, label: 'largest rate r (every condition)' }),
      points: int(6, { ge: 2, le: 20, suggestions: [4, 6, 10], label: 'rates from π₀ to r' }),
      baseRate: slider(0.001, 0.2, 0.02, { step: 0.001, label: 'π₀' }),
    }),
    fit: fitting('3 · fit'),
  })
  const { top, points, baseRate } = state.sweep
  const rates = Array.from({ length: points }, (_, j) => baseRate + ((top - baseRate) * j) / (points - 1))
  const settings: SweepSettings = { data: state.data, rates, baseRate, maxSteps: state.fit.maxSteps }
  const trained = useTrainedRun(settings, sweepTask)
  const run = trained.run.value
  const key = JSON.stringify(trained.trained)
  const pts = run?.points ?? []
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = Math.min(
    picked && picked.run === trained.trained ? picked.index : pts.length - 1,
    Math.max(0, pts.length - 1),
  )
  const at = pts[index]
  const trainedRates = (trained.trained as SweepSettings | null)?.rates ?? rates
  const pickRate = (v: number) => {
    let best = 0
    pts.forEach((p, i) => {
      if (Math.abs(p.rate - v) < Math.abs(pts[best].rate - v)) best = i
    })
    setPicked({ run: trained.trained, index: best })
  }
  const rAxis = useAxis({ label: 'zero-inflation rate r on an unsuitable item', range: [0, Math.max(...trainedRates)] })
  const gapAxis = useAxis({ label: 'equity gap (mean θ̂ − θ: with − without a condition)', hold: 'union', key })
  const rAxis2 = useAxis({
    label: 'zero-inflation rate r on an unsuitable item',
    range: [0, Math.max(...trainedRates)],
  })
  const rmseAxis = useAxis({ label: 'ability RMSE', hold: 'union', key })
  const gAxis = useAxis({ label: 'group', categories: GROUPS })
  const bAxis = useAxis({ label: 'mean θ̂ − θ (bias)', hold: 'union', key })
  const series = (m: LearnerModelKind, f: (p: (typeof pts)[number]['models'][number]) => number) =>
    pts.map((p) => {
      const r = p.models.find((x) => x.model === m)
      return r ? f(r) : NaN
    })
  const x = pts.map((p) => p.rate)
  return (
    <Figure
      title="Equity of learner models as the zero-inflation gap grows"
      purpose="The more often an unsuitable format turns a neurodivergent student's answers into zeros, the further IRT and a context-aware logistic model underestimate those students' abilities; IRT-ZILM attributes most of the extra zeros to the format and keeps the gap between groups much smaller."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls
          run={trained as never}
          progress={run ? pts.length / Math.max(1, run.total) : 0}
          progressText={run ? `${pts.length} / ${run.total} rates` : ''}
        />
      }
      readouts={
        <>
          <Readout label="rate r" value={at ? fmt(at.rate) : '—'} />
          <Readout
            label="structural zeros: with · without a condition"
            value={at ? `${fmt(at.structuralShare.with)} · ${fmt(at.structuralShare.without)}` : '—'}
          />
          {ORDER.map((m) => {
            const r = at?.models.find((v) => v.model === m)
            return (
              <Readout
                key={m}
                label={`${MODELS[m].name}: gap · ability r · held-out Brier`}
                value={r ? `${fmt(r.equity.gap)} · ${fmt(r.ability.pearson)} · ${fmt(r.test.brier)}` : '—'}
              />
            )
          })}
          {trained.run.running && <Readout label="fitting" value={<span aria-busy="true">in a worker…</span>} />}
        </>
      }
      caption="Each rate r gets its own dataset from aifn learnerResponses (same seed, so the same students and items; every condition's rate on a fully unsuitable item set to r) and learnerEquitySweep fits IRT, the linear KTM and IRT-ZILM to 80% of its answers. Left: the equity gap, the mean ability error of students with a condition minus that of students without (0 is equitable; negative means they are underestimated). Middle: the ability RMSE over all students. Right: the bias in each group at the rate marked on the left. Press Train; drag the marker on the first two charts to choose the rate shown on the right."
    >
      <Plots cols={3}>
        <Plot x={rAxis} y={gapAxis} title={!trained.trained ? 'press Train to start' : 'equity gap'}>
          {run && <Curve name="equitable" x={[0, Math.max(...trainedRates)]} y={[0, 0]} muted dashed thin />}
          {pts.length > 0 &&
            ORDER.map((m) => (
              <Curve
                key={m}
                name={MODELS[m].name}
                slot={MODELS[m].slot}
                x={x}
                y={series(m, (r) => r.equity.gap)}
                showPoints
              />
            ))}
          {at && <Handle kind="x" at={at.rate} onDrag={pickRate} label={`r = ${fmt(at.rate, 2)}`} />}
        </Plot>
        <Plot x={rAxis2} y={rmseAxis} title="ability error">
          {pts.length > 0 &&
            ORDER.map((m) => (
              <Curve
                key={m}
                name={MODELS[m].name}
                slot={MODELS[m].slot}
                x={x}
                y={series(m, (r) => r.equity.rmse)}
                showPoints
              />
            ))}
          {at && <Handle kind="x" at={at.rate} onDrag={pickRate} label={`r = ${fmt(at.rate, 2)}`} />}
        </Plot>
        <Plot x={gAxis} y={bAxis} title={at ? `bias by group at r = ${fmt(at.rate, 2)}` : 'bias by group'}>
          {at &&
            ORDER.map((m, j) => {
              const r = at.models.find((v) => v.model === m)
              return r ? (
                <Bars
                  key={m}
                  name={MODELS[m].name}
                  slot={MODELS[m].slot}
                  x={r.equity.groups.map((g) => g.group + (j - 1) * 0.27)}
                  y={r.equity.groups.map((g) => (Number.isFinite(g.bias) ? g.bias : 0))}
                  width={0.25}
                />
              ) : null
            })}
          {at && <Curve name="no bias" x={[-0.5, GROUPS.length - 0.5]} y={[0, 0]} muted dashed thin />}
        </Plot>
      </Plots>
    </Figure>
  )
}
