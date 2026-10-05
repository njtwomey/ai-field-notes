/**
 * Matchbox (`aifn-methods/retrieval/recommenders`): Bayesian ordinal ratings from user and item features, learned
 * online by assumed-density filtering, in the worker: held-out error for known and cold-start users, the item trait
 * map and one pair's predicted rating distribution.
 */
import { useState } from 'react'
import type { MatchboxRun } from 'aifn-methods/retrieval/recommenders'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, float, int, row, setting, slider, useFigureState, type Task } from 'aifn-render/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Bars, Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

type MatchboxSettings = { data: Record<string, number>; seed: number; options: Record<string, unknown> }
const matchboxTask = (s: MatchboxSettings): Task<MatchboxRun> =>
  call<MatchboxRun>(
    'applied/retrieval/recommenders/matchboxRun',
    call('applied/retrieval/recommenders/matchboxRatings', call('foundation/random/stream', s.seed), s.data),
    s.options,
  )

const QUADRANTS = ['true traits (+, +)', 'true traits (−, +)', 'true traits (−, −)', 'true traits (+, −)']
const quadrant = (a: number, b: number) => (a >= 0 ? (b >= 0 ? 0 : 3) : b >= 0 ? 1 : 2)

export function MatchboxSpecimen() {
  const state = useFigureState({
    data: row('1 · ratings', {
      users: int(300, { ge: 10, le: 3000, suggestions: [100, 300, 1000], label: 'users' }),
      items: int(100, { ge: 5, le: 1000, suggestions: [50, 100, 300], label: 'items' }),
      perUser: int(20, { ge: 1, le: 200, suggestions: [5, 20, 50], label: 'ratings per user' }),
      featureShare: slider(0, 1, 0.8, { step: 0.05, label: 'trait share explained by features' }),
      noise: float(0.5, { ge: 0, suggestions: [0.2, 0.5, 1], label: 'rating noise sd' }),
    }),
    model: row('2 · model', {
      traits: int(2, { ge: 0, le: 10, suggestions: [0, 1, 2, 5], label: 'traits K' }),
      useFeatures: setting(true, 'user and item features'),
      beta: float(1, { gt: 0, suggestions: [0.5, 1, 2], label: 'latent noise β' }),
      passes: int(1, { ge: 1, le: 20, suggestions: [1, 3], label: 'passes (1 = online)' }),
      seed: int(2, { ge: 0, le: 9999, label: 'seed' }),
    }),
    pair: row('4 · held-out pair', {
      pair: int(0, { ge: 0, le: 99999, label: 'held-out rating #' }),
    }),
  })
  const { users, items, perUser, featureShare, noise } = state.data
  const { traits, useFeatures, beta, passes, seed } = state.model
  const settings: MatchboxSettings = {
    data: { users, items, perUser, featureShare, noise, traits: Math.max(1, traits) },
    seed,
    options: { traits, useFeatures, beta, passes, seed },
  }
  const trained = useTrainedRun(settings, matchboxTask)
  const run = trained.run.value
  const shots = run?.checkpoints ?? []
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const shot = shots[index]
  const key = JSON.stringify(trained.trained)
  const seenAxis = useAxis({ label: 'ratings seen', range: [0, run?.total ?? 1], key })
  const rmseAxis = useAxis({ label: 'held-out RMSE (levels)', hold: 'union', key })
  const t1 = useAxis({ label: 'item trait 1 (posterior mean)', hold: 'union', key })
  const t2 = useAxis({ label: 'item trait 2', hold: 'union', key, equal: t1 })
  const levelAxis = useAxis({ label: 'rating level', range: [-0.5, (run?.levels ?? 5) - 0.5] })
  const pAxis = useAxis({ label: 'predicted probability', range: [0, 1] })
  const nTest = run?.test.user.length ?? 0
  const q = nTest ? state.pair.pair % nTest : 0
  const K = run?.traits ?? traits
  const groups = run
    ? Array.from({ length: run.trueItemTraits.length / Math.max(1, K) }, (_, i) =>
        quadrant(run.trueItemTraits[i * K], K > 1 ? run.trueItemTraits[i * K + 1] : 0),
      )
    : []
  const L = run?.levels ?? 5
  const probs = run && nTest ? Array.from(run.predictions.subarray(q * L, (q + 1) * L)) : []
  const pickSeen = (v: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.seen - v) < Math.abs(shots[best].seen - v)) best = i
    })
    pick(best)
  }
  return (
    <Figure
      title="Matchbox: Bayesian recommendation with feature traits"
      purpose="Matchbox learns user and item traits as linear functions of their features and scores a pair by the inner product of its traits, through an ordinal likelihood with learned thresholds; every weight has a Gaussian posterior updated once per rating, and features carry what is learned to users it has never seen."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? run.seen / Math.max(1, run.total) : 0}
            progressText={run ? `${run.seen} / ${run.total} ratings` : ''}
          />
          <ControlRow label="3 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `${shots[i]?.seen ?? 0} ratings`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="ratings seen" value={shot ? shot.seen : '—'} />
          <Readout label="RMSE known · cold users" value={shot ? `${fmt(shot.rmse)} · ${fmt(shot.coldRmse)}` : '—'} />
          <Readout label="accuracy (most probable level)" value={shot ? fmt(shot.accuracy) : '—'} />
          <Readout label="mean trait variance" value={shot ? fmt(shot.traitVariance) : '—'} />
          <Readout
            label="held-out pair (user, item, rating)"
            value={run && nTest ? `${run.test.user[q]}, ${run.test.item[q]}, ${run.test.level[q]}` : '—'}
          />
        </>
      }
      caption="aifn matchboxRatings (user traits Aₖ·f + noise, item traits Bₖ·g + noise from 3 standard-normal side features each, affinity Σₖ uₖvₖ plus noise, cut into 5 equal-share levels) and matchboxRun: each user and item is a one-hot id plus, when switched on, its side features; matchboxUpdate is one ADF step per rating; 10% of users are held out entirely (cold start) and 20% of the other ratings are held out. Left: RMSE of the expected level for known users (slot 0) and cold-start users (slot 1), against each item's mean training rating (dashed) and the global mean for cold users (faint dashed). Middle: posterior mean item traits at the checkpoint, coloured by the quadrant of the true traits (recovered up to a rotation and scale). Right: the predicted rating distribution of one held-out pair at the end, its true rating in ink. Some seeds draw feature maps whose two traits are nearly collinear, and the item traits then lie near a line. Press Train; play the checkpoints or drag the marker."
    >
      <Plots cols={3}>
        <Plot x={seenAxis} y={rmseAxis} title={!trained.trained ? 'press Train to start' : 'held-out error'}>
          {run && (
            <Curve name="item mean" x={[0, run.total]} y={[run.itemMeanRmse, run.itemMeanRmse]} emphasis dashed />
          )}
          {run && (
            <Curve
              name="global mean (cold)"
              x={[0, run.total]}
              y={[run.coldGlobalRmse, run.coldGlobalRmse]}
              muted
              dashed
            />
          )}
          {run && (
            <Curve name="known users" slot={0} x={shots.map((c) => c.seen)} y={shots.map((c) => c.rmse)} showPoints />
          )}
          {run && (
            <Curve
              name="cold-start users"
              slot={1}
              x={shots.map((c) => c.seen)}
              y={shots.map((c) => c.coldRmse)}
              showPoints
            />
          )}
          {shot && <Handle kind="x" at={shot.seen} onDrag={pickSeen} label={`${shot.seen}`} />}
        </Plot>
        <Plot x={t1} y={t2} title={K >= 2 ? 'item traits' : 'item traits (needs K ≥ 2)'}>
          {shot && K >= 2 && (
            <Points
              name="items"
              x={Array.from({ length: groups.length }, (_, i) => shot.itemTraits[i * K])}
              y={Array.from({ length: groups.length }, (_, i) => shot.itemTraits[i * K + 1])}
              group={groups}
              groupNames={QUADRANTS}
              size={6}
            />
          )}
        </Plot>
        <Plot x={levelAxis} y={pAxis} title="one held-out pair">
          {run && nTest > 0 && <Bars name="P(rating)" slot={0} x={probs.map((_, l) => l)} y={probs} width={0.6} />}
          {run && nTest > 0 && (
            <Bars name="true rating" emphasis x={[run.test.level[q]]} y={[probs[run.test.level[q]] ?? 0]} width={0.6} />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
