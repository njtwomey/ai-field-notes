import { useMemo, useState } from 'react'
import { censoredSurvival, weibullPhSurvival } from 'aifn-methods/data/synthetic'
import {
  aftModel,
  aftSurvival,
  coxPartialLikelihood,
  coxPh,
  coxSurvival,
  harrellConcordance,
} from 'aifn-methods/learning/survival'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { kaplanMeier, logRankTest } from 'aifn/probability/tests'
import {
  Area,
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  Slider,
  useAxis,
} from 'aifn-render'

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

export function CoxExplorer() {
  const [view, setView] = useState<'curves' | 'likelihood'>('curves')
  const [treatmentBeta, setTreatmentBeta] = useState(-0.7)
  const [roundTies, setRoundTies] = useState(false)
  const [tiesMethod, setTiesMethod] = useState<'efron' | 'breslow'>('efron')
  const [newtonStep, setNewtonStep] = useState(0)

  const n = 200
  const censoringRate = 0.03

  const data = useMemo(() => {
    const s = stream('cox-specimen-data')
    return censoredSurvival(s, {
      n,
      coefficients: [treatmentBeta, 0.5],
      shape: 1.5,
      scale: 10,
      censoringRate,
      followUp: 30,
      round: roundTies,
    })
  }, [treatmentBeta, roundTies])

  const tFlat = useMemo(() => toFlat(data.time), [data])
  const eFlat = useMemo(() => toFlat(data.event), [data])
  const xFlat = useMemo(() => toFlat(data.x), [data])

  const fitted = useMemo(() => {
    const arms = [0, 1].map((arm) => {
      const idx = Array.from({ length: tFlat.length }, (_, i) => i).filter((i) => xFlat[2 * i] === arm)
      const ti = idx.map((i) => tFlat[i])
      const ei = idx.map((i) => eFlat[i])
      const km = idx.length ? kaplanMeier(ti, ei) : null
      return { km, censored: idx.filter((i) => eFlat[i] === 0).map((i) => tFlat[i]) }
    })

    const group = Float64Array.from({ length: tFlat.length }, (_, i) => xFlat[2 * i])
    const both = arms.every((a) => a.km)
    const test = both ? logRankTest(tFlat, eFlat, group) : null
    const cox = coxPh(data.x, data.time, data.event, { ties: tiesMethod })
    const aft = aftModel(data.x, data.time, data.event, { family: 'weibull' })

    const risk = Float64Array.from(
      { length: tFlat.length },
      (_, i) => xFlat[2 * i] * cox.coefficients[0] + xFlat[2 * i + 1] * cox.coefficients[1],
    )
    const concordance = harrellConcordance(data.time, data.event, risk)

    return { arms, test, cox, aft, concordance }
  }, [data, tFlat, eFlat, xFlat, tiesMethod])

  const end = useMemo(() => Math.max(...tFlat) * 1.02, [tFlat])
  const grid = useMemo(() => Float64Array.from({ length: 151 }, (_, i) => 0.01 + (end * i) / 150), [end])

  // Profiles along β₁ for partial likelihood view
  const likelihoodResult = useMemo(() => {
    const fits = (['efron', 'breslow'] as const).map((ties) => coxPh(data.x, data.time, data.event, { ties }))
    const b2 = fits[0].coefficients[1]
    const betaGrid = Float64Array.from({ length: 121 }, (_, i) => -3 + i * 0.05)
    const profiles = (['efron', 'breslow'] as const).map((ties) =>
      Float64Array.from(betaGrid, (b) => coxPartialLikelihood(data.x, data.time, data.event, [b, b2], ties)),
    )
    return { fits, betaGrid, profiles, b2 }
  }, [data])

  const efronFit = likelihoodResult.fits[0]
  const iterations = efronFit.iterations + 1
  const currentStep = Math.min(newtonStep, iterations - 1)
  const pathX = Array.from({ length: iterations }, (_, i) => efronFit.coefficientPath[2 * i])
  const pathY = Array.from({ length: iterations }, (_, i) => efronFit.coefficientPath[2 * i + 1])

  const tiesCount = useMemo(() => {
    const seen = new Map<number, number>()
    for (let i = 0; i < tFlat.length; i++) {
      if (eFlat[i] === 1) seen.set(tFlat[i], (seen.get(tFlat[i]) ?? 0) + 1)
    }
    let tied = 0
    for (const c of seen.values()) if (c > 1) tied += c
    return tied
  }, [tFlat, eFlat])

  const tAxis = useAxis({ label: 'time t', range: [0, end] })
  const sAxis = useAxis({ label: 'survival S(t)', range: [0, 1] })
  const betaAxis = useAxis({ label: 'treatment coefficient β₁', range: [-3, 3] })
  const llAxis = useAxis({ label: 'partial log-likelihood ℓ(β)' })
  const pathAxisX = useAxis({ label: 'β₁ (treatment)', range: [-3, 2] })
  const pathAxisY = useAxis({ label: 'β₂ (feature)', range: [-2, 3] })

  const { cox, aft, arms, test } = fitted
  const label = ['control (x₁=0)', 'treated (x₁=1)']

  return (
    <Figure
      title="Cox proportional hazards: survival curves & partial likelihood"
      purpose="Explore the semiparametric Cox proportional hazards model, baseline hazard estimation, and partial likelihood optimization under Breslow and Efron ties."
      caption="Interactive exploration of the Cox semiparametric model. In 'Survival curves' view, compare non-parametric Kaplan–Meier step curves, semiparametric Cox survival predictions (Breslow baseline), and true Weibull curves across treatment arms. In 'Partial likelihood' view, observe Newton's optimization steps on ℓ(β) and examine how Breslow vs Efron tie approximations diverge when event times are tied."
    >
      <ControlRow>
        <Select
          label="Explorer view"
          value={view}
          onChange={(v) => setView(v as 'curves' | 'likelihood')}
          options={[
            { value: 'curves', label: 'Survival curves (KM vs Cox vs AFT)' },
            { value: 'likelihood', label: 'Partial likelihood & Newton steps' },
          ]}
        />
        <Slider
          label="Treatment log HR β₁"
          value={treatmentBeta}
          min={-1.8}
          max={1.2}
          step={0.1}
          onChange={setTreatmentBeta}
        />
        <Select
          label="Ties handling"
          value={tiesMethod}
          onChange={(v) => setTiesMethod(v as 'efron' | 'breslow')}
          options={[
            { value: 'efron', label: 'Efron approximation' },
            { value: 'breslow', label: 'Breslow approximation' },
          ]}
        />
        <Select
          label="Round times"
          value={roundTies ? 'yes' : 'no'}
          onChange={(v) => setRoundTies(v === 'yes')}
          options={[
            { value: 'no', label: 'Continuous (no ties)' },
            { value: 'yes', label: 'Rounded (creates ties)' },
          ]}
        />
      </ControlRow>

      {view === 'likelihood' && (
        <ControlRow>
          <Player label="Newton step" value={currentStep} count={iterations} onChange={setNewtonStep} />
        </ControlRow>
      )}

      <div className="my-2 flex flex-wrap gap-4 font-mono text-xs text-muted-foreground">
        <Readout label="HR treated" value={formatNumber(cox.hazardRatios[0])} />
        <Readout
          label="β̂₁ (Cox)"
          value={`${formatNumber(cox.coefficients[0])} ± ${formatNumber(cox.standardErrors[0])}`}
        />
        <Readout label="C-index" value={formatNumber(fitted.concordance)} />
        <Readout
          label="log-rank p"
          value={test ? (test.pValue < 0.001 ? '< 0.001' : formatNumber(test.pValue)) : '—'}
        />
        <Readout label="tied events" value={tiesCount} />
      </div>

      {view === 'curves' ? (
        <Plots cols={2}>
          <Plot x={tAxis} y={sAxis} title="Kaplan–Meier by treatment arm">
            {arms.map((a, k) => {
              if (!a.km) return null
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
                <Area
                  key={`b${k}`}
                  name={`${label[k]} 95% CI`}
                  slot={k}
                  x={hi.x}
                  y={hi.y}
                  base={lo.y}
                  opacity={0.15}
                  line={false}
                />
              )
            })}
            {arms.map((a, k) => {
              if (!a.km) return null
              const s = steps(toFlat(a.km.time), toFlat(a.km.survival), end)
              return <Curve key={`km${k}`} name={`KM: ${label[k]}`} slot={k} x={s.x} y={s.y} />
            })}
            {arms.map((a, k) => {
              if (!a.km) return null
              const kmT = toFlat(a.km.time)
              const kmS = toFlat(a.km.survival)
              const at = (time: number) => {
                let v = 1
                for (let j = 0; j < kmT.length && kmT[j] <= time; j++) v = kmS[j]
                return v
              }
              return (
                <Points
                  key={`c${k}`}
                  name={`Censored ${label[k]}`}
                  slot={k}
                  x={a.censored}
                  y={a.censored.map(at)}
                  size={5}
                />
              )
            })}
          </Plot>

          <Plot x={tAxis} y={sAxis} title="Cox & AFT fits vs Weibull Ground Truth">
            {[0, 1].map((arm) => {
              const s = steps(cox.baseline.time, coxSurvival(cox, [arm, 0]), end)
              return <Curve key={`cox${arm}`} name={`Cox model: ${label[arm]}`} slot={arm} x={s.x} y={s.y} />
            })}
            {[0, 1].map((arm) => (
              <Curve
                key={`aft${arm}`}
                name={`AFT fit: ${label[arm]}`}
                slot={arm}
                x={grid}
                y={aftSurvival(aft, [arm, 0], grid)}
                thin
              />
            ))}
            {[0, 1].map((arm) => (
              <Curve
                key={`truth${arm}`}
                name={`Truth: ${label[arm]}`}
                x={grid}
                y={weibullPhSurvival(data.truth, [arm, 0], grid)}
                emphasis
                dashed
                thin
              />
            ))}
          </Plot>
        </Plots>
      ) : (
        <Plots cols={2}>
          <Plot x={betaAxis} y={llAxis} title="Partial log-likelihood profile ℓ(β₁)">
            <Curve name="Efron" slot={0} x={likelihoodResult.betaGrid} y={likelihoodResult.profiles[0]} />
            <Curve name="Breslow" slot={1} x={likelihoodResult.betaGrid} y={likelihoodResult.profiles[1]} dashed />
            <Points
              name="Newton iterate"
              x={[pathX[currentStep]]}
              y={[coxPartialLikelihood(data.x, data.time, data.event, [pathX[currentStep], likelihoodResult.b2])]}
              emphasis
              size={8}
            />
          </Plot>

          <Plot x={pathAxisX} y={pathAxisY} title="Newton optimization trajectory in (β₁, β₂)">
            <Curve
              name="Newton path"
              x={pathX.slice(0, currentStep + 1)}
              y={pathY.slice(0, currentStep + 1)}
              slot={0}
              showPoints
            />
            <Points name="True coefficients" x={[treatmentBeta]} y={[0.5]} emphasis size={8} />
          </Plot>
        </Plots>
      )}
    </Figure>
  )
}
