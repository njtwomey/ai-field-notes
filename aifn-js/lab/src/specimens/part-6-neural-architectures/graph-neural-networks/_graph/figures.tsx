/**
 * Showcase: semi-supervised node classification on Zachary's karate club with a two-layer GCN, GAT or GraphSAGE,
 * trained in the worker by aifn-applied `nodeClassificationRun` (layers from `aifn/nn/graph`). The page draws the
 * checkpoints the run streams: predicted clubs on the graph, GAT attention on the edges, the 2-D logits and the curves.
 */
import { useMemo, useState } from 'react'
import { toFlat } from 'aifn/foundation/tensor'
import { karateClub } from 'aifn-applied/data/real'
import type { NodeClassificationOptions, NodeSnapshot } from 'aifn-applied/neural/graph'
import { Player } from '@lab/controls'
import { forceLayout } from '@lab/diagram'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, when, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Plot, Plots, Points, Readout, Segments, useAxis } from '@lab/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const CLUBS = ['Mr. Hi', 'Officer']

type Settings = Omit<NodeClassificationOptions, 'data'>

function runTask(s: Settings): Task<NodeSnapshot> {
  return call<NodeSnapshot>('applied/neural/graph/nodeClassificationRun', {
    ...s,
    data: call('applied/data/real/karateClub'),
  })
}

export function NodeClassificationShowcase() {
  const state = useFigureState({
    model: row('1 · first layer', {
      kind: choice(
        [
          { value: 'gcn', label: 'graph convolution (GCN)' },
          { value: 'gat', label: 'graph attention (GAT)' },
          { value: 'sage', label: 'GraphSAGE' },
        ],
        'gat',
        { label: 'layer' },
      ),
      hidden: int(8, { ge: 1, le: 64, suggestions: [4, 8, 16], label: 'hidden width' }),
      heads: int(2, { ge: 1, le: 8, suggestions: [1, 2, 4], label: 'heads', when: when('kind', 'gat') }),
      aggregate: choice(['mean', 'max', 'pool', 'sum'], 'mean', { label: 'aggregator', when: when('kind', 'sage') }),
    }),
    labels: row('2 · labels', {
      perClass: int(1, { ge: 1, le: 10, suggestions: [1, 2, 4], label: 'labelled members per club' }),
    }),
    run: row('3 · training (Adam, full batch)', {
      steps: int(200, { ge: 1, suggestions: [100, 200, 500], label: 'steps' }),
      stepSize: float(0.02, { gt: 0, le: 1, scale: 'log10', suggestions: [0.005, 0.02, 0.05], label: 'learning rate' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const data = useMemo(() => karateClub(), [])
  const clubs = useMemo(() => Array.from(toFlat(data.y!)), [data])
  // Labelled members: the leaders first (0 is Mr. Hi, 33 the Officer), then the next members of each club by index.
  const train = useMemo(() => {
    const a = clubs.flatMap((c, i) => (c === 0 ? [i] : [])).slice(0, state.labels.perClass)
    const b = clubs
      .flatMap((c, i) => (c === 1 ? [i] : []))
      .reverse()
      .slice(0, state.labels.perClass)
    return [...a, ...b]
  }, [clubs, state.labels.perClass])
  const settings: Settings = {
    kind: state.model.kind,
    hidden: state.model.hidden,
    heads: state.model.heads,
    aggregate: state.model.aggregate,
    train,
    steps: state.run.steps,
    stepSize: state.run.stepSize,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, runTask)
  const snap = trained.run.value
  const shots = snap?.checkpoints ?? []
  const [pick, setPick] = useState(0)
  const index = Math.min(pick, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const shown = trained.trained ?? settings
  const pos = useMemo(() => forceLayout(data.graph.nodes, data.graph.edges, { iterations: 400 }), [data])
  const predicted = shot ? shot.logits.map((r) => (r[1] > r[0] ? 1 : 0)) : null

  // GAT: each friendship's attention, the mean of its two directions, against the uniform share 1/(degree + 1).
  const edgeBins = useMemo(() => {
    const plain = data.graph.edges.map((e) => ({
      from: [pos[e.from].x, pos[e.from].y] as const,
      to: [pos[e.to].x, pos[e.to].y] as const,
    }))
    if (!shot?.attention || !snap?.edges) return { plain, low: [], high: [] }
    const deg = new Array(data.graph.nodes).fill(1)
    snap.edges.destination.forEach((v, k) => snap.edges!.source[k] !== v && deg[v]++)
    const ratio = new Map<string, number[]>()
    snap.edges.source.forEach((u, k) => {
      const v = snap.edges!.destination[k]
      if (u === v) return
      const key = `${Math.min(u, v)}-${Math.max(u, v)}`
      ratio.set(key, [...(ratio.get(key) ?? []), shot.attention![k] * deg[v]])
    })
    const low: typeof plain = []
    const high: typeof plain = []
    const mid: typeof plain = []
    data.graph.edges.forEach((e, k) => {
      const r = ratio.get(`${Math.min(e.from, e.to)}-${Math.max(e.from, e.to)}`) ?? [1]
      const m = r.reduce((s, v) => s + v, 0) / r.length
      ;(m > 1.25 ? high : m < 0.8 ? low : mid).push(plain[k])
    })
    return { plain: mid, low, high }
  }, [data, pos, shot, snap])

  const gx = useAxis({ label: 'layout x', nice: false })
  const gy = useAxis({ label: 'layout y', nice: false, equal: gx })
  const lx = useAxis({ label: 'logit: Mr. Hi', hold: 'union', key: trained.trained })
  const ly = useAxis({ label: 'logit: Officer', hold: 'union', key: trained.trained })
  const sx = useAxis({ label: 'step', range: [0, shown.steps ?? 200], key: shown.steps })
  const accAxis = useAxis({ label: 'accuracy', range: [0, 1.02] })
  const lossAxis = useAxis({ label: 'loss (labelled)', hold: 'union', key: trained.trained })
  const h = snap?.history
  const hi = h && shot ? h.step.indexOf(shot.step) : -1
  const sel = (ids: readonly number[]) => ({ x: ids.map((i) => pos[i].x), y: ids.map((i) => pos[i].y) })
  const all = clubs.map((_, i) => i)
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'training…' : undefined
  const done = snap?.step ?? 0
  return (
    <Figure
      title="Graph neural networks: semi-supervised node classification"
      purpose="Two rounds of message passing carry a handful of labels across the friendship graph: after training on one member per club, the network places most members in the club they joined."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={done / (shown.steps ?? 200)}
            progressText={`${done} / ${shown.steps} steps`}
          />
          <ControlRow label="checkpoints">
            <Player
              value={index}
              onChange={setPick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `step ${shots[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="loss (labelled)" value={hi >= 0 ? f3(h!.loss[hi]) : '—'} />
          <Readout label="accuracy (labelled)" value={hi >= 0 ? f3(h!.trainAccuracy[hi]) : '—'} />
          <Readout label="accuracy (unlabelled)" value={hi >= 0 ? f3(h!.testAccuracy[hi]) : '—'} />
          <Readout label="labelled members" value={shown.train.join(', ')} />
        </>
      }
      caption={
        <>
          aifn-applied <code>nodeClassificationRun</code> on <code>karateClub</code> (34 members, 78 friendships, one
          identity feature per member): a{' '}
          <code>{shown.kind === 'gcn' ? 'GraphConv' : shown.kind === 'gat' ? 'GraphAttention' : 'SageConv'}</code> layer
          of width {shown.hidden}
          {shown.kind === 'gat' ? ` × ${shown.heads} heads` : ''} with tanh, then a layer to two logits, trained by Adam
          on the cross-entropy of the labelled members only (large ink marks). Top left: the friendship graph, each
          member coloured and shaped by the predicted club, with a red halo where it is not the club they joined.{' '}
          {shown.kind === 'gat'
            ? 'Edges: thick where the first layer attends to a friend more than the uniform share 1/(degree + 1), thin where less (averaged over heads and both directions). '
            : ''}
          Top right: every member&apos;s two logits, a 2-D embedding in which the clubs separate. Below: accuracy on the
          labelled and unlabelled members, and the loss. Press Train, then play the checkpoints from step 0.
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[3, 2]}>
        <Plot x={gx} y={gy} title={empty ?? 'the club predicted for every member'}>
          <Segments name="friendships" segments={edgeBins.plain} />
          {edgeBins.low.length > 0 && <Segments name="less attention" segments={edgeBins.low} dashed />}
          {edgeBins.high.length > 0 && <Segments name="more attention" segments={edgeBins.high} emphasis width={2.5} />}
          <Points name="labelled" x={sel(shown.train).x} y={sel(shown.train).y} emphasis size={18} />
          {predicted ? (
            <>
              {/* A red halo under each member placed in the wrong club; the legend's marks then match the points. */}
              <Points
                name="wrong club"
                x={sel(all.filter((i) => predicted[i] !== clubs[i])).x}
                y={sel(all.filter((i) => predicted[i] !== clubs[i])).y}
                tone="destructive"
                size={17}
              />
              <Points
                name="members"
                x={sel(all).x}
                y={sel(all).y}
                group={predicted}
                groupNames={CLUBS.map((c) => `predicted ${c}`)}
                size={10}
              />
            </>
          ) : (
            <Points
              name="members"
              x={sel(all).x}
              y={sel(all).y}
              muted
              shape={clubs}
              shapeNames={CLUBS.map((c) => `joined ${c}`)}
              size={10}
            />
          )}
        </Plot>
        <Plot x={lx} y={ly} title={empty ?? 'the two logits of every member'}>
          {shot && (
            <Points
              name="members"
              x={shot.logits.map((r) => r[0])}
              y={shot.logits.map((r) => r[1])}
              group={clubs}
              groupNames={CLUBS.map((c) => `joined ${c}`)}
            />
          )}
          {shot && (
            <Points
              name="labelled"
              x={shown.train.map((i) => shot.logits[i][0])}
              y={shown.train.map((i) => shot.logits[i][1])}
              emphasis
              size={12}
            />
          )}
        </Plot>
        <Plot x={sx} y={accAxis}>
          {h && <Curve name="accuracy (labelled)" x={h.step} y={h.trainAccuracy} slot={2} />}
          {h && <Curve name="accuracy (unlabelled)" x={h.step} y={h.testAccuracy} slot={3} />}
          {shot && <Curve name="checkpoint" x={[shot.step, shot.step]} y={[0, 1.02]} emphasis dashed width={1} />}
        </Plot>
        <Plot x={sx} y={lossAxis}>
          {h && <Curve name="loss (labelled)" x={h.step} y={h.loss} slot={4} />}
          {shot && hi >= 0 && <Points name="checkpoint" x={[shot.step]} y={[h!.loss[hi]]} emphasis />}
        </Plot>
      </Plots>
    </Figure>
  )
}
