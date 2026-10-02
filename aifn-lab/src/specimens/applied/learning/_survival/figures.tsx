/**
 * Survival analysis from Kaplan–Meier to Cox: censored data with a known Weibull truth, the Kaplan–Meier curves of two
 * arms with the log-rank test, and the Cox and AFT model curves against the truth; then the Cox partial likelihood with
 * Newton's steps, under Efron's and Breslow's handling of ties. Data from `aifn-applied/data/synthetic`; the
 * estimators from `aifn/probability/tests`; the models from `aifn-applied/learning/survival`.
 */
import { useMemo } from 'react'
import { censoredSurvival, weibullPhSurvival } from 'aifn-applied/data/synthetic'
import {
  aftModel,
  aftSurvival,
  coxPartialLikelihood,
  coxPh,
  coxSurvival,
  harrellConcordance,
} from 'aifn-applied/learning/survival'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { kaplanMeier, logRankTest } from 'aifn/probability/tests'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, float, int, row, setting, slider, useFigureState } from '@lab/state'
import { Area, Curve, formatNumber, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

/** A right-continuous step function through (time, value) pairs, from (0, start). */
function steps(time: ArrayLike<number>, value: ArrayLike<number>, end: number, start = 1) {
  const x: number[] = [0]
  const y: number[] = [start]
  for (let i = 0; i < time.length; i++) {
    x.push(time[i], time[i])
    y.push(i === 0 ? start : value[i - 1], value[i])
  }
  x.push(end)
  y.push(time.length ? value[time.length - 1] : start)
  return { x, y }
}

const dataRow = () =>
  row('1 · data', {
    n: int(200, { ge: 10, le: 5000, suggestions: [50, 100, 200, 1000], label: 'subjects' }),
    treatment: slider(-2, 2, -0.7, { step: 0.05, label: 'treatment log hazard ratio β₁' }),
    feature: slider(-2, 2, 0.5, { step: 0.05, label: 'feature log hazard ratio β₂' }),
    shape: slider(0.5, 3, 1.5, { step: 0.05, label: 'Weibull shape k' }),
    censoring: float(0.03, { ge: 0, le: 1, suggestions: [0, 0.01, 0.03, 0.1], label: 'censoring rate' }),
    followUp: float(30, { gt: 0, suggestions: [10, 20, 30, 100], label: 'follow-up ends at' }),
    round: setting(false, 'round times up (ties)'),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })

function useSurvivalData(d: {
  n: number
  treatment: number
  feature: number
  shape: number
  censoring: number
  followUp: number
  round: boolean
  seed: number
}) {
  const { n, treatment, feature, shape, censoring, followUp, round, seed } = d
  return useMemo(
    () =>
      censoredSurvival(stream(`survival/${seed}`), {
        n,
        coefficients: [treatment, feature],
        shape,
        censoringRate: censoring,
        followUp,
        round,
      }),
    [n, treatment, feature, shape, censoring, followUp, round, seed],
  )
}

// ── 1 · Kaplan–Meier, log-rank, Cox and AFT ─────────────────────────────────────────────────────────────────────────

export function SurvivalCurvesSpecimen() {
  const state = useFigureState({
    data: dataRow(),
    show: row('2 · models', {
      aft: choice(
        [
          { value: 'weibull', label: 'Weibull AFT' },
          { value: 'log-normal', label: 'log-normal AFT' },
        ],
        'weibull',
        { label: 'parametric model' },
      ),
      ties: choice(['efron', 'breslow'], 'efron', { label: 'Cox ties' }),
      truth: setting(true, 'true survival'),
      bands: setting(true, 'Kaplan–Meier 95% bands'),
    }),
  })
  const data = useSurvivalData(state.data)
  const { aft: family, ties, truth, bands } = state.show
  const fitted = useMemo(() => {
    const x = toFlat(data.x)
    const t = toFlat(data.time)
    const e = toFlat(data.event)
    const arms = [0, 1].map((arm) => {
      const idx = Array.from({ length: t.length }, (_, i) => i).filter((i) => x[2 * i] === arm)
      const ti = idx.map((i) => t[i])
      const ei = idx.map((i) => e[i])
      const km = idx.length ? kaplanMeier(ti, ei) : null
      return { km, censored: idx.filter((i) => e[i] === 0).map((i) => t[i]) }
    })
    const group = Float64Array.from({ length: t.length }, (_, i) => x[2 * i])
    const both = arms.every((a) => a.km)
    const test = both ? logRankTest(t, e, group) : null
    const cox = coxPh(data.x, data.time, data.event, { ties: ties as 'efron' | 'breslow' })
    const aft = aftModel(data.x, data.time, data.event, { family: family as 'weibull' | 'log-normal' })
    const risk = Float64Array.from(
      { length: t.length },
      (_, i) => x[2 * i] * cox.coefficients[0] + x[2 * i + 1] * cox.coefficients[1],
    )
    return { arms, test, cox, aft, concordance: harrellConcordance(data.time, data.event, risk) }
  }, [data, family, ties])
  const end = useMemo(() => Math.max(...toFlat(data.time)) * 1.02, [data])
  const grid = useMemo(() => Float64Array.from({ length: 201 }, (_, i) => 0.01 + (end * i) / 200), [end])
  const tAxis = useAxis({ label: 'time t', range: [0, end], key: Math.round(end) })
  const sAxis = useAxis({ label: 'survival S(t)', range: [0, 1] })
  const { cox, aft, arms, test } = fitted
  const label = ['control', 'treated']
  return (
    <Figure
      title="Kaplan–Meier, log-rank, Cox and AFT"
      purpose="Kaplan–Meier multiplies the conditional survival 1 − d/n at each event time and uses censored subjects only while they are at risk; the log-rank test compares observed and expected events between arms; Cox and AFT models explain the gap with covariates, one through the hazard and one through the time scale."
      state={state}
      defaultSize="L"
      readouts={{
        'log-rank': (
          <>
            <Readout label="χ²" value={test ? fmt(test.statistic) : '—'} />
            <Readout label="p-value" value={test ? fmt(test.pValue, 2) : '—'} />
            <Readout label="KM medians" value={arms.map((a) => (a.km ? fmt(a.km.median) : '—')).join(' / ')} />
          </>
        ),
        Cox: (
          <>
            <Readout
              label="hazard ratio (treated)"
              value={`${fmt(cox.hazardRatios[0])} (β̂₁ ${fmt(cox.coefficients[0])} ± ${fmt(cox.standardErrors[0], 2)})`}
            />
            <Readout label="β̂₂" value={`${fmt(cox.coefficients[1])} ± ${fmt(cox.standardErrors[1], 2)}`} />
            <Readout label="C-index" value={fmt(fitted.concordance)} />
          </>
        ),
        AFT: (
          <>
            <Readout label="time ratio (treated)" value={fmt(Math.exp(aft.coefficients[0]))} />
            <Readout label="σ" value={fmt(aft.scale)} />
            <Readout label="truth −β₁/k" value={fmt(-state.data.treatment / state.data.shape)} />
          </>
        ),
      }}
      caption="aifn censoredSurvival draws Weibull proportional-hazards times (a 0/1 treatment and a standard-normal feature) censored at an exponential rate and at the end of follow-up. Left: kaplanMeier per arm (steps, with log–log 95% bands and ticks at censored times) and logRankTest between arms. Right: the same arms by model, at feature 0: coxPh (Breslow's baseline times e^{xβ̂}), the AFT fit (aftModel) and the true curves (dashed ink). Under a Weibull truth both models are right; a log-normal AFT bends the wrong way in the tail. Rounding times up creates ties, where Efron's and Breslow's partial likelihoods differ."
    >
      <Plots cols={2}>
        <Plot x={tAxis} y={sAxis} title="Kaplan–Meier by arm">
          {arms.map((a, k) => {
            if (!a.km || !bands) return null
            // Where the log–log band is undefined (Ŝ at 0 or 1) it collapses onto the curve.
            const S = toFlat(a.km.survival)
            const lo = steps(
              toFlat(a.km.time),
              toFlat(a.km.lower).map((v, j) => (Number.isFinite(v) ? v : S[j])),
              end,
            )
            const hi = steps(
              toFlat(a.km.time),
              toFlat(a.km.upper).map((v, j) => (Number.isFinite(v) ? v : S[j])),
              end,
            )
            return (
              <Area key={`b${k}`} name={label[k]} slot={k} x={hi.x} y={hi.y} base={lo.y} opacity={0.12} line={false} />
            )
          })}
          {arms.map((a, k) => {
            if (!a.km) return null
            const s = steps(toFlat(a.km.time), toFlat(a.km.survival), end)
            return <Curve key={`k${k}`} name={label[k]} slot={k} x={s.x} y={s.y} />
          })}
          {arms.map((a, k) => {
            if (!a.km) return null
            const kmT = toFlat(a.km.time)
            const kmS = toFlat(a.km.survival)
            const at = (t: number) => {
              let v = 1
              for (let j = 0; j < kmT.length && kmT[j] <= t; j++) v = kmS[j]
              return v
            }
            return (
              <Points key={`c${k}`} name={label[k]} slot={k} x={a.censored} y={a.censored.map(at)} shape={3} size={6} />
            )
          })}
        </Plot>
        <Plot x={tAxis} y={sAxis} title="models against the truth (feature = 0)">
          {[0, 1].map((arm) => {
            const s = steps(cox.baseline.time, coxSurvival(cox, [arm, 0]), end)
            return <Curve key={`cox${arm}`} name={`Cox, ${label[arm]}`} slot={arm} x={s.x} y={s.y} />
          })}
          {[0, 1].map((arm) => (
            <Curve
              key={`aft${arm}`}
              name={`AFT (smooth), ${label[arm]}`}
              slot={arm}
              x={grid}
              y={aftSurvival(aft, [arm, 0], grid)}
              thin
            />
          ))}
          {truth &&
            [0, 1].map((arm) => (
              <Curve
                key={`true${arm}`}
                name="truth (dashed)"
                x={grid}
                y={weibullPhSurvival(data.truth, [arm, 0], grid)}
                emphasis
                dashed
                thin
              />
            ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · The partial likelihood and Newton's method ──────────────────────────────────────────────────────────────────

export function PartialLikelihoodSpecimen() {
  const state = useFigureState({ data: dataRow() })
  const d = state.data
  const data = useSurvivalData(d)
  const result = useMemo(() => {
    const fits = (['efron', 'breslow'] as const).map((ties) => coxPh(data.x, data.time, data.event, { ties }))
    const b2 = fits[0].coefficients[1]
    const grid = Float64Array.from({ length: 121 }, (_, i) => -3 + i * 0.05)
    const profiles = (['efron', 'breslow'] as const).map((ties) =>
      Float64Array.from(grid, (b) => coxPartialLikelihood(data.x, data.time, data.event, [b, b2], ties)),
    )
    return { fits, grid, profiles, b2 }
  }, [data])
  const efron = result.fits[0]
  const iterations = efron.iterations + 1
  const [k, setK] = usePlayhead(iterations)
  const betaAxis = useAxis({ label: 'β₁ (treatment), β₂ at its fit', range: [-3, 3] })
  const llAxis = useAxis({ label: 'partial log-likelihood ℓ', hold: 'union', key: JSON.stringify(d) })
  const pathAxis = useAxis({ label: 'β₁', range: [-3, 3] })
  const path2Axis = useAxis({ label: 'β₂', range: [-3, 3] })
  const pathX = Array.from({ length: iterations }, (_, i) => efron.coefficientPath[2 * i])
  const pathY = Array.from({ length: iterations }, (_, i) => efron.coefficientPath[2 * i + 1])
  const at = Math.min(k, iterations - 1)
  const tiesCount = useMemo(() => {
    const t = toFlat(data.time)
    const e = toFlat(data.event)
    const seen = new Map<number, number>()
    for (let i = 0; i < t.length; i++) if (e[i] === 1) seen.set(t[i], (seen.get(t[i]) ?? 0) + 1)
    let tied = 0
    for (const c of seen.values()) if (c > 1) tied += c
    return tied
  }, [data])
  return (
    <Figure
      title="The partial likelihood and Newton's steps"
      purpose="Cox's partial likelihood depends on the times only through their order, so the baseline hazard drops out; it is concave in β, and Newton's method with the exact information matrix reaches its maximum in a few steps."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · Newton steps">
          <Player className="col-span-full" value={at} onChange={setK} count={iterations} label="step" />
        </ControlRow>
      }
      readouts={{
        [`step ${at}`]: (
          <>
            <Readout label="β" value={`(${fmt(pathX[at])}, ${fmt(pathY[at])})`} />
            <Readout label="ℓ(β)" value={fmt(efron.path[at], 6)} />
          </>
        ),
        ties: (
          <>
            <Readout label="tied events" value={tiesCount} />
            <Readout
              label="β̂₁ Efron / Breslow"
              value={`${fmt(efron.coefficients[0], 4)} / ${fmt(result.fits[1].coefficients[0], 4)}`}
            />
          </>
        ),
      }}
      caption="aifn coxPartialLikelihood along β₁ with β₂ held at its Efron fit (left, Efron and Breslow), and coxPh's Newton iterates from β = 0 (right; Efron ties). The ink point on the left is the current iterate's β₁ on the profile. Play the steps. Without ties the two curves coincide; with rounded times (data row) Breslow treats the tied subjects as all at risk for each event and shrinks the estimate towards zero."
    >
      <Plots cols={2}>
        <Plot x={betaAxis} y={llAxis} title="profile of ℓ along β₁">
          <Curve name="Efron" slot={0} x={result.grid} y={result.profiles[0]} />
          <Curve name="Breslow" slot={1} x={result.grid} y={result.profiles[1]} dashed />
          <Points
            name="Newton iterate"
            x={[pathX[at]]}
            y={[coxPartialLikelihood(data.x, data.time, data.event, [pathX[at], result.b2])]}
            emphasis
            size={8}
          />
        </Plot>
        <Plot x={pathAxis} y={path2Axis} title="Newton's path in (β₁, β₂)">
          <Curve name="path" x={pathX.slice(0, at + 1)} y={pathY.slice(0, at + 1)} slot={0} showPoints />
          <Points name="truth" x={[d.treatment]} y={[d.feature]} emphasis size={8} shape={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}
