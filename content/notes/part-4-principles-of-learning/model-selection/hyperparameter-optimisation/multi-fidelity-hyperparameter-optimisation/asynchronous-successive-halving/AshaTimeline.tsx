import { useMemo, useState } from 'react'
import { Annotation, Figure, int, Player, Plot, Readout, Segments, slider, useAxis, useFigureState } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { learningCurves } from 'aifn-methods/data/synthetic'
import {
  multiFidelitySteps,
  successiveHalvingSchedule,
  type MultiFidelityState,
} from 'aifn-methods/optimisation/multi-fidelity'

/** Promotions out of a rung that are not among the rung's best ⌊size/η⌋ once every evaluation in it is known. */
function mispromotions(s: MultiFidelityState, eta: number) {
  let count = 0
  const rungs = new Set(s.evaluations.map((e) => e.rung))
  for (const k of rungs) {
    const sorted = s.evaluations.filter((e) => e.rung === k).sort((a, b) => a.loss - b.loss || a.config - b.config)
    const top = new Set(sorted.slice(0, Math.floor(sorted.length / eta)).map((e) => e.config))
    count += s.promotions.filter((p) => p.from === k && !top.has(p.config)).length
  }
  return count
}

/** One run's facts: when the first full-budget result arrived, and how busy the workers were. */
function summary(s: MultiFidelityState, R: number, workers: number) {
  const full = s.evaluations.filter((e) => e.budget === R).map((e) => e.end)
  const busy = s.evaluations.reduce((a, e) => a + (e.end - e.start), 0)
  return { firstFull: full.length ? Math.min(...full) : NaN, busy: busy / (workers * s.time) }
}

/**
 * Synchronous successive halving against ASHA on simulated workers: each bar is one evaluation, from its start to its
 * end, on the worker that ran it, coloured by rung. Training time equals the budget.
 */
export function AshaTimeline() {
  const state = useFigureState({
    n: int(81, { ge: 3, le: 243, suggestions: [27, 81, 243], label: 'configurations n' }),
    maxBudget: int(9, { ge: 3, le: 81, suggestions: [9, 27], label: 'maximum budget R' }),
    eta: int(3, { ge: 2, le: 4, suggestions: [2, 3, 4], label: 'reduction factor η' }),
    workers: int(8, { ge: 1, le: 32, suggestions: [4, 8, 16], label: 'workers' }),
    crossing: slider(0, 1, 0.5, { step: 0.05, label: 'crossing κ' }),
    seed: int(4, { ge: 1, le: 999, label: 'seed' }),
  })
  const { n, maxBudget, eta, workers, crossing, seed } = state

  const runs = useMemo(() => {
    const curves = learningCurves(stream(seed), {
      n,
      budgets: Array.from({ length: maxBudget }, (_, i) => i + 1),
      crossing,
      noise: 0,
    })
    const schedule = successiveHalvingSchedule({ n, minBudget: 1, maxBudget, eta })
    const go = (asynchronous: boolean) =>
      run(multiFidelitySteps(curves, schedule, { asynchronous, workers }), undefined, 100_000)
    const sync = go(false)
    const asha = go(true)
    const final = toFlat(curves.final)
    return {
      schedule,
      sync,
      asha,
      rungs: schedule.rungs.length,
      final: (s: MultiFidelityState) => final[s.incumbent!.config],
      best: final[curves.best],
    }
  }, [n, maxBudget, eta, workers, crossing, seed])

  const horizon = Math.ceil(Math.max(runs.sync.time, runs.asha.time))
  const key = `${n}|${maxBudget}|${eta}|${workers}|${crossing}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const now = Math.min(cursor.key === key ? cursor.step : 0, horizon)

  const sync = summary(runs.sync, maxBudget, workers)
  const asha = summary(runs.asha, maxBudget, workers)
  const time = useAxis({ label: 'simulated time (budget units)', range: [0, horizon] })
  const lane = useAxis({ label: 'worker', range: [-0.7, workers - 0.3], integer: true })

  const panel = (s: MultiFidelityState, firstFull: number, label: string) => (
    <Plot x={time} y={lane} height={Math.min(320, 70 + 22 * workers)}>
      {Array.from({ length: runs.rungs }, (_, k) => (
        <Segments
          key={k}
          name={`rung ${k}`}
          slot={k}
          width={Math.max(3, Math.min(12, 120 / workers))}
          segments={s.evaluations
            .filter((e) => e.rung === k && e.start < now)
            .map((e) => ({
              from: [e.start + 0.04, e.worker] as const,
              to: [Math.max(e.start + 0.08, Math.min(e.end, now) - 0.04), e.worker] as const,
            }))}
        />
      ))}
      {Number.isFinite(firstFull) && firstFull <= now && (
        <Annotation x={firstFull} text={`${label}: first full-budget result`} dashed />
      )}
    </Plot>
  )

  return (
    <Figure
      title="Synchronous successive halving against ASHA"
      state={state}
      caption="Simulated workers run the same successive-halving schedule over the same learning curves; an evaluation takes as long as its budget. Each bar is one evaluation on one worker, coloured by rung. Top: synchronous successive halving, which starts a rung only when the previous rung has finished, so workers idle at the end of each rung and no configuration reaches R until the lower rungs are complete. Bottom: ASHA, which promotes a configuration as soon as it is in the top 1/η of the results its rung has so far, and otherwise starts a new configuration at the bottom rung. Play through time to see the first full-budget result arrive much earlier under ASHA. ASHA promotes a few configurations that a full rung would not have, so it spends somewhat more budget."
      controls={<Player value={now} onChange={(v) => setCursor({ key, step: v })} count={horizon + 1} label="time" />}
      readouts={
        <>
          <Readout label="first result at R: synchronous / ASHA" value={`${sync.firstFull} / ${asha.firstFull}`} />
          <Readout label="total time: synchronous / ASHA" value={`${runs.sync.time} / ${runs.asha.time}`} />
          <Readout label="budget spent: synchronous / ASHA" value={`${runs.sync.spent} / ${runs.asha.spent}`} />
          <Readout
            label="workers busy: synchronous / ASHA"
            value={`${Math.round(100 * sync.busy)}% / ${Math.round(100 * asha.busy)}%`}
          />
          <Readout label="ASHA mispromotions" value={String(mispromotions(runs.asha, eta))} />
          <Readout
            label="final loss returned: synchronous / ASHA / best"
            value={`${runs.final(runs.sync).toFixed(3)} / ${runs.final(runs.asha).toFixed(3)} / ${runs.best.toFixed(3)}`}
          />
        </>
      }
    >
      <div className="grid gap-2">
        <div className="text-xs text-muted-foreground">synchronous successive halving</div>
        {panel(runs.sync, sync.firstFull, 'synchronous')}
        <div className="text-xs text-muted-foreground">asynchronous successive halving (ASHA)</div>
        {panel(runs.asha, asha.firstFull, 'ASHA')}
      </div>
    </Figure>
  )
}
