import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { GruCell } from 'aifn-compute/nn/layers'
import { memoryTrace, staticReach, timeRespectingReach, type MemoryUpdate } from 'aifn-methods/neural/graph'
import { averageUpdate, EVENTS, NAMES, NODES, POSITIONS, USERS } from './events'

type Updater = 'average' | 'gru'

/**
 * A temporal graph network's memory module on ten timed interactions (`memoryTrace`), stepped one event at a time,
 * with the nodes that one node's information can have reached along time-respecting paths (`timeRespectingReach`)
 * against those joined to it by any path (`staticReach`).
 */
export function TemporalMemory() {
  const state = useFigureState({
    updater: choice<Updater>(
      [
        { value: 'average', label: 'average (worked example)' },
        { value: 'gru', label: 'GRU cell, random weights' },
      ],
      'average',
      { label: 'memory updater' },
    ),
    origin: choice(NAMES, 'u1', { label: 'follow information from' }),
    seed: int(3, { ge: 0, le: 999, label: 'GRU seed', when: (v) => v.updater === 'gru' }),
  })
  const [step, setStep] = useState(0)

  const update = useMemo((): MemoryUpdate => {
    if (state.updater === 'average') return averageUpdate
    const cell = GruCell(3, 1)
    const params = cell.init(stream(state.seed))
    // Input: the other endpoint's memory, the rating and a one-frequency time encoding cos(0.3 Δt).
    return (self, other, x, dt) =>
      toFlat(cell.step(params, tensor([other[0], x[0], Math.cos(0.3 * dt)]), { h: tensor([...self]) }).h as Tensor)
  }, [state.updater, state.seed])
  const trace = useMemo(() => memoryTrace(NODES, 1, EVENTS, update), [update])
  const origin = NAMES.indexOf(state.origin)
  const reach = useMemo(() => timeRespectingReach(NODES, EVENTS, origin), [origin])
  const connected = useMemo(() => staticReach(NODES, EVENTS, origin), [origin])

  const memory = trace.memory[step].map((m) => m[0])
  const colour = useScaleColor('diverging')
  const bound = Math.max(1e-9, ...trace.memory.flatMap((ms) => ms.map((m) => Math.abs(m[0]))))
  const colours = memory.map((m) => colour((m / bound + 1) / 2))

  const current = step > 0 ? EVENTS[step - 1] : null
  const past = EVENTS.slice(0, Math.max(0, step - 1))
  const held = POSITIONS.filter((_, i) => reach[step][i])
  const linked = POSITIONS.filter((_, i) => connected[step][i] && !reach[step][i])

  const xAxis = useAxis({ label: '', range: [-0.8, 3.8], nice: false })
  const yAxis = useAxis({ label: '', range: [-0.6, 4.6], nice: false, equal: xAxis })

  return (
    <Figure
      title="Node memories over an event stream"
      state={state}
      hoverReadout={false}
      caption={`Ten timed ratings between five users (left) and three items (right); step through them with the player. Each event updates the memories of its two endpoints from their memories just before it: with the average updater, s ← (s + s_other + rating)/2, the worked example's rule; with the GRU updater, a GRU cell with random weights reads the other memory, the rating and a time encoding cos(0.3 Δt). Node colour and label show the memory. The thick ring marks the nodes that information from the chosen node can have reached along paths whose event times increase; the thin ring marks nodes joined to it by earlier events in the wrong order, which a static graph of the same edges would count as reachable. Past events are drawn thin and the current one thick.`}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={EVENTS.length + 1}
          label="event"
          format={(s) => (s === 0 ? 'start' : `event ${s} at t = ${EVENTS[s - 1].time}`)}
        />
      }
      readouts={
        <>
          <Readout
            label="current event"
            value={
              current
                ? `${NAMES[current.source]} rates ${NAMES[current.destination]} ${current.feature[0]} at t = ${current.time}`
                : '·'
            }
          />
          <Readout
            label="time since last update (user, item)"
            value={current ? trace.gaps[step - 1].map((g) => formatNumber(g)).join(', ') : '·'}
          />
          <Readout
            label={`reached from ${state.origin}: time-respecting / any order`}
            value={`${reach[step].filter(Boolean).length} / ${connected[step].filter(Boolean).length}`}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={320} bare>
          {past.length > 0 && (
            <Segments
              name="past events"
              segments={past.map((e) => ({ from: POSITIONS[e.source], to: POSITIONS[e.destination] }))}
              muted
              width={1}
            />
          )}
          {current && (
            <Segments
              name="current event"
              segments={[{ from: POSITIONS[current.source], to: POSITIONS[current.destination] }]}
              emphasis
              width={3}
            />
          )}
          {linked.length > 0 && (
            <Points
              name="joined in the wrong order"
              x={linked.map((p) => p[0])}
              y={linked.map((p) => p[1])}
              muted
              size={34}
            />
          )}
          {held.length > 0 && (
            <Points
              name={`reached from ${state.origin}`}
              x={held.map((p) => p[0])}
              y={held.map((p) => p[1])}
              emphasis
              size={36}
            />
          )}
          <Points
            name="memory"
            x={POSITIONS.map((p) => p[0])}
            y={POSITIONS.map((p) => p[1])}
            colors={colours}
            size={26}
            shape={POSITIONS.map((_, i) => (i < USERS ? 0 : 1))}
            shapeNames={['user', 'item']}
            labels={memory.map((m, i) => `${NAMES[i]}: ${formatNumber(Number(m.toFixed(3)))}`)}
          />
        </Plot>
      </div>
    </Figure>
  )
}
