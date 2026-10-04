/**
 * Markov chains: a transition graph drawn with `Diagram` whose states can be dragged, with a walker and the exact
 * distribution p₀Pᵗ played step by step (`markovChainSteps`); gambler's ruin absorption against first-step analysis
 * and simulated paths; and the distance to stationarity against the spectral gap. The chains come from
 * `aifn-methods/data/synthetic`; every probability, time and distance from `aifn/probability/markov`.
 */
import { useMemo, useState } from 'react'
import {
  ehrenfestChain,
  gamblersRuinChain,
  randomWalkChain,
  weatherChain,
  type NamedChain,
} from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn/foundation/random'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import {
  absorption,
  classifyStates,
  distanceToStationarity,
  markovChainSteps,
  mixingTime,
  simulateChain,
  spectralGap,
  stationaryDistribution,
  type MarkovChainState,
} from 'aifn/probability/markov'
import { Player, usePlayhead } from '@lab/controls'
import { Diagram, type DiagramEdge, type DiagramNode, type DiagramSpec } from '@lab/diagram'
import { ControlRow, Figure } from '@lab/layout'
import { choice, float, int, row, slider, useFigureState, type AnyValues } from '@lab/state'
import { Annotation, Bars, Curve, formatNumber, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) =>
  Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : v > 0 ? '∞' : '—'

const CHAINS = [
  { value: 'weather', label: 'weather (sunny, cloudy, rainy)' },
  { value: 'cycle', label: 'random walk on a cycle' },
  { value: 'ehrenfest', label: 'Ehrenfest urn' },
] as const
const chainOf = (v: AnyValues) => String(v.chain)

/** States on a circle of radius r grid units, the first at the top. */
const circle = (n: number, r: number): [number, number][] =>
  Array.from({ length: n }, (_, i) => [r * Math.sin((2 * Math.PI * i) / n), -r * Math.cos((2 * Math.PI * i) / n)])

const R = 3.2
const FRAME = { x0: -R - 2, y0: -R - 2, x1: R + 2, y1: R + 2 }

/** The transition graph: a node per state, shaded by its probability; an edge per positive transition. */
function chainSpec(chain: NamedChain, at: [number, number][], shade: readonly number[], walker: number): DiagramSpec {
  const P = toRows(chain.P) as number[][]
  const n = P.length
  const nodes: DiagramNode[] = chain.states.map((label, i) => ({
    id: `s${i}`,
    x: at[i][0],
    y: at[i][1],
    shape: 'circle',
    w: 1.3,
    h: 1.3,
    label,
    small: label.length > 3,
    shade: Math.min(1, shade[i]),
    selected: i === walker,
  }))
  const edges: DiagramEdge[] = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (!(P[i][j] > 0)) continue
      const label = P[i][j].toFixed(2).replace(/^0/, '')
      if (i === j) {
        // A self-loop points away from the centre of the drawing.
        const [x, y] = at[i]
        const side = Math.abs(x) > Math.abs(y) ? (x > 0 ? 'e' : 'w') : y > 0 ? 's' : 'n'
        edges.push({ from: `s${i}:${side}`, to: `s${i}`, label, labelRotate: false })
      } else {
        const both = P[j][i] > 0
        edges.push({
          from: `s${i}`,
          to: `s${j}`,
          route: both ? 'curve' : 'straight',
          bend: both ? 1.3 : undefined,
          label,
        })
      }
    }
  return { nodes, edges, frame: FRAME, fitLabels: false, maxScale: 1.6 }
}

// ── 1 · The chain, its walker and its distribution ───────────────────────────────────────────────────────────────────

export function ChainSpecimen() {
  const state = useFigureState({
    chain: row('1 · chain', {
      chain: choice(CHAINS, 'weather', { label: 'chain' }),
      states: int(6, {
        ge: 3,
        le: 12,
        suggestions: [4, 5, 6, 8],
        label: 'states',
        when: (v) => chainOf(v) === 'cycle',
      }),
      hold: slider(0, 0.9, 0, { step: 0.05, label: 'hold probability', when: (v) => chainOf(v) !== 'weather' }),
      drift: slider(-1, 1, 0.3, { step: 0.05, label: 'drift', when: (v) => chainOf(v) === 'cycle' }),
      balls: int(4, { ge: 1, le: 10, suggestions: [2, 4, 8], label: 'balls', when: (v) => chainOf(v) === 'ehrenfest' }),
    }),
    run: row('2 · run', {
      start: int(0, { ge: 0, le: 11, label: 'start state' }),
      steps: int(80, { ge: 1, le: 5000, suggestions: [40, 80, 400, 2000], label: 'steps' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
  })
  const { chain: name, states, hold, drift, balls } = state.chain
  const { start, steps, seed } = state.run
  const chain = useMemo(() => {
    if (name === 'weather') return weatherChain()
    if (name === 'cycle') return randomWalkChain(states, { hold, drift, topology: 'cycle' })
    return ehrenfestChain(balls, hold)
  }, [name, states, hold, drift, balls])
  const n = chain.states.length
  const s0 = Math.min(start, n - 1)
  const [moved, setMoved] = useState<Record<string, [number, number][]>>({})
  const layoutKey = `${name}/${n}`
  const at = moved[layoutKey] ?? circle(n, R)
  const runOf = useMemo(() => {
    const tr = trace(markovChainSteps(chain.P, { start: s0 }), undefined, steps, {
      keep: 'all',
      stream: stream(`markov/walker/${seed}`),
    })
    return tr.steps as MarkovChainState[]
  }, [chain, s0, steps, seed])
  const info = useMemo(() => {
    const classes = classifyStates(chain.P)
    let pi: number[] | null = null
    try {
      pi = Array.from(toFlat(stationaryDistribution(chain.P)))
    } catch {
      pi = null
    }
    return { classes, pi, gap: spectralGap(chain.P) }
  }, [chain])
  const [step, setStep] = usePlayhead(runOf.length)
  const s = runOf[Math.min(step, runOf.length - 1)]
  const p = Array.from(toFlat(s.distribution))
  const occupation = Array.from(toFlat(s.visits)).map((v) => v / (s.t + 1))
  const spec = chainSpec(chain, at, p, s.state)
  const ts = runOf.map((r) => r.t)
  const upto = step + 1
  const lambda = info.gap.secondModulus
  const distAxis = useAxis({ label: 'distance to π (TV)', log: true, range: [1e-4, 1] })
  const tAxis = useAxis({ label: 'step t', range: [0, steps] })
  const stateAxis = useAxis({ label: 'state', categories: [...chain.states] })
  const probAxis = useAxis({ label: 'probability', range: [0, 1] })
  const clip = (v: number) => (Number.isFinite(v) ? Math.max(1e-4, v) : NaN)
  const ref = runOf.map((r) => clip(runOf[0].distance * lambda ** r.t))
  return (
    <Figure
      title="A Markov chain forgets where it started"
      purpose="The distribution of the chain after t steps is p₀Pᵗ; for an irreducible, aperiodic chain it converges to the stationary π = πP from any start, and the distance shrinks roughly like λ⋆ᵗ, the second-largest eigenvalue modulus."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · steps">
          <Player className="col-span-full" value={step} onChange={setStep} count={runOf.length} label="t" />
        </ControlRow>
      }
      readouts={{
        chain: (
          <>
            <Readout label="irreducible" value={info.classes.irreducible ? 'yes' : 'no'} />
            <Readout label="period" value={info.classes.periods.join(', ')} />
            <Readout label="λ⋆" value={fmt(lambda)} />
            <Readout label="relaxation time" value={fmt(info.gap.relaxationTime)} />
            <Readout label="π" value={info.pi ? info.pi.map((v) => fmt(v, 2)).join(', ') : 'not unique'} />
          </>
        ),
        [`step ${s.t}`]: (
          <>
            <Readout label="walker at" value={chain.states[s.state]} />
            <Readout label="‖p₀Pᵗ − π‖" value={fmt(s.distance)} />
            <Readout label="‖occupation − π‖" value={fmt(s.occupationDistance)} />
          </>
        ),
      }}
      caption="Drag the states to rearrange the graph; edges carry the transition probabilities and each state is shaded by p₀Pᵗ, its probability at step t, with a ring around the walker's current state. Play, or step with the arrows. Middle: p₀Pᵗ (bars), the walker's share of visits so far, and π (ink points). Right: both distances to π on a log scale, with λ⋆ᵗ for reference. On a cycle with an even number of states and no holding the chain has period 2: p₀Pᵗ oscillates and never converges, while the visit shares still approach π. Holding with some probability makes it aperiodic."
    >
      <div className="grid h-full grid-cols-[minmax(0,5fr)_minmax(0,7fr)] gap-3">
        <Diagram
          spec={spec}
          ariaLabel="The transition graph; drag a state to move it"
          onNodeDrag={(id, x, y) => {
            const i = Number(id.slice(1))
            const next = at.map((q, k): [number, number] =>
              k === i
                ? [Math.max(FRAME.x0 + 1, Math.min(FRAME.x1 - 1, x)), Math.max(FRAME.y0 + 1, Math.min(FRAME.y1 - 1, y))]
                : q,
            )
            setMoved((m) => ({ ...m, [layoutKey]: next }))
          }}
        />
        <Plots cols={2} widths={[50, 50]}>
          <Plot x={stateAxis} y={probAxis}>
            <Bars name="p₀Pᵗ" x={chain.states.map((_, i) => i)} y={p} slot={0} />
            <Points name="visit share" x={chain.states.map((_, i) => i)} y={occupation} slot={1} size={9} />
            {info.pi && <Points name="π" x={chain.states.map((_, i) => i)} y={info.pi} emphasis size={7} />}
          </Plot>
          <Plot x={tAxis} y={distAxis}>
            <Curve
              name="‖p₀Pᵗ − π‖"
              x={ts.slice(0, upto)}
              y={runOf.slice(0, upto).map((r) => clip(r.distance))}
              slot={0}
            />
            <Curve
              name="‖occupation − π‖"
              x={ts.slice(0, upto)}
              y={runOf.slice(0, upto).map((r) => clip(r.occupationDistance))}
              slot={1}
            />
            <Curve name="λ⋆ᵗ" x={ts} y={ref} dashed muted />
          </Plot>
        </Plots>
      </div>
    </Figure>
  )
}

// ── 2 · Gambler's ruin ───────────────────────────────────────────────────────────────────────────────────────────────

export function GamblersRuinSpecimen() {
  const state = useFigureState({
    game: row('1 · game', {
      target: int(10, { ge: 2, le: 60, suggestions: [5, 10, 20, 50], label: 'target N' }),
      win: slider(0.3, 0.7, 0.47, { step: 0.01, label: 'chance of winning a bet p' }),
      start: int(5, { ge: 1, le: 59, label: 'starting stake i' }),
    }),
    paths: row('2 · simulated paths', {
      paths: int(30, { ge: 1, le: 500, suggestions: [10, 30, 100], label: 'paths' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
  })
  const { target: N, win, start } = state.game
  const { paths, seed } = state.paths
  const i0 = Math.min(Math.max(1, start), N - 1)
  const chain = useMemo(() => gamblersRuinChain(N, win), [N, win])
  const exact = useMemo(() => absorption(chain.P), [chain])
  const sims = useMemo(() => {
    const horizon = Math.max(50, Math.ceil(6 * (Array.from(toFlat(exact.expectedSteps))[i0 - 1] ?? 50)))
    return Array.from({ length: paths }, (_, k) => {
      const path = Array.from(
        toFlat(simulateChain(child(stream(`markov/ruin/${seed}`), 'path', k), chain.P, i0, horizon)),
      )
      const end = path.findIndex((v) => v === 0 || v === N)
      return end < 0 ? path : path.slice(0, end + 1)
    })
  }, [chain, exact, i0, paths, seed, N])
  const reached = sims.filter((p) => p[p.length - 1] === N).length
  const meanLength = sims.reduce((a, p) => a + p.length - 1, 0) / sims.length
  const probs = (toRows(exact.probabilities) as number[][]).map((r) => r[1])
  const times = Array.from(toFlat(exact.expectedSteps))
  const stakes = exact.transient
  const [at, setAt] = usePlayhead(Math.max(...sims.map((p) => p.length)))
  const longest = Math.max(...sims.map((p) => p.length))
  const stakeAxis = useAxis({ label: 'stake i', range: [0, N] })
  const probAxis = useAxis({ label: 'P(reach N before 0)', range: [0, 1] })
  const timeAxis = useAxis({ label: 'expected bets until the end', hold: 'union', key: `${N}/${win}` })
  const tAxis = useAxis({ label: 'bet', range: [0, longest - 1] })
  const pathAxis = useAxis({ label: 'stake', range: [0, N] })
  return (
    <Figure
      title="Gambler's ruin: absorption by first-step analysis"
      purpose="Starting from stake i, the chance of reaching N before ruin and the expected number of bets solve linear equations from one step of the chain: hᵢ = p hᵢ₊₁ + q hᵢ₋₁. In a fair game hᵢ = i/N; a small disadvantage makes a distant target almost unreachable."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · bets">
          <Player className="col-span-full" value={at} onChange={setAt} count={longest} label="bet" />
        </ControlRow>
      }
      readouts={{
        [`from stake ${i0}`]: (
          <>
            <Readout label="P(reach N)" value={fmt(probs[i0 - 1])} />
            <Readout label="simulated" value={`${reached} of ${paths}`} />
            <Readout label="expected bets" value={fmt(times[i0 - 1])} />
            <Readout label="simulated mean bets" value={fmt(meanLength)} />
            <Readout label="sd of bets" value={fmt(Math.sqrt(Array.from(toFlat(exact.varianceSteps))[i0 - 1]))} />
          </>
        ),
      }}
      caption="Left: the absorption probability and the expected duration from every stake, from the fundamental matrix N = (I − Q)⁻¹ (B = NR, t = N1); the ink points are the starting stake. Right: simulated games from that stake, played bet by bet, each stopping at 0 or N; a dot marks each game's stake now, coloured by how it ends. At p = 0.5 the probability is the straight line i/N and the duration the parabola i(N − i); at p = 0.47 with N = 10 the chance from 5 is already down to 0.35."
    >
      <Plots cols={3} widths={[30, 30, 40]}>
        <Plot x={stakeAxis} y={probAxis}>
          <Curve name="P(reach N)" x={[0, ...stakes, N]} y={[0, ...probs, 1]} slot={0} showPoints />
          <Points name="start" x={[i0]} y={[probs[i0 - 1]]} emphasis size={10} />
        </Plot>
        <Plot x={stakeAxis} y={timeAxis}>
          <Curve name="expected bets" x={[0, ...stakes, N]} y={[0, ...times, 0]} slot={1} showPoints />
          <Points name="start" x={[i0]} y={[times[i0 - 1]]} emphasis size={10} />
        </Plot>
        <Plot x={tAxis} y={pathAxis}>
          {sims.map((p, k) => (
            <Curve
              key={k}
              id={`path-${k}`}
              name={p[p.length - 1] === N ? 'reaches N' : 'ruined'}
              x={p.slice(0, at + 1).map((_, t) => t)}
              y={p.slice(0, at + 1)}
              slot={p[p.length - 1] === N ? 2 : 3}
              thin
              silent
            />
          ))}
          {/* Each game's stake at the current bet (its last stake once it has ended), so step 0 shows the start. */}
          {[2, 3].map((slot) => {
            const games = sims.filter((p) => (p[p.length - 1] === N ? 2 : 3) === slot)
            return (
              <Points
                key={slot}
                name={slot === 2 ? 'reaches N' : 'ruined'}
                x={games.map((p) => Math.min(at, p.length - 1))}
                y={games.map((p) => p[Math.min(at, p.length - 1)])}
                slot={slot}
                size={7}
              />
            )
          })}
          <Annotation y={N} dashed />
          <Annotation y={0} text="ruin" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3 · Mixing ───────────────────────────────────────────────────────────────────────────────────────────────────────

export function MixingSpecimen() {
  const state = useFigureState({
    chain: row('1 · lazy random walks', {
      states: int(10, { ge: 3, le: 60, suggestions: [5, 10, 20, 40], label: 'states n' }),
      hold: slider(0.05, 0.9, 0.5, { step: 0.05, label: 'hold probability' }),
      epsilon: float(0.25, { gt: 0, lt: 1, suggestions: [0.1, 0.25], label: 'ε of t_mix(ε)' }),
    }),
  })
  const { states: n, hold, epsilon } = state.chain
  const r = useMemo(() => {
    const chains = (['cycle', 'path'] as const).map((topology) => {
      const P = randomWalkChain(n, { hold, topology }).P
      return { topology, P, m: mixingTime(P, { epsilon, maxSteps: 20000 }), gap: spectralGap(P) }
    })
    // One horizon for both curves: 2.5 times the slower mixing time.
    const slowest = Math.max(...chains.map((c) => (Number.isFinite(c.m.time) ? c.m.time : 160)))
    const T = Math.min(Math.ceil(2.5 * slowest), 20000)
    return chains.map((c) => ({ ...c, d: Array.from(toFlat(distanceToStationarity(c.P, T).worst)) }))
  }, [n, hold, epsilon])
  const longest = Math.max(...r.map((c) => c.d.length))
  const tAxis = useAxis({ label: 'step t', range: [0, longest - 1] })
  const dAxis = useAxis({ label: 'worst-case distance d(t)', range: [0, 1] })
  return (
    <Figure
      title="Mixing time and the spectral gap"
      purpose="The mixing time is the first t at which the chain is within ε of π from every start; for a reversible chain the relaxation time 1/(1 − λ⋆) bounds it on both sides, and on a cycle or a path of n states it grows like n²."
      state={state}
      readouts={Object.fromEntries(
        r.map((c) => [
          `${c.topology} of ${n}`,
          <>
            <Readout label="t_mix(ε)" value={fmt(c.m.time)} />
            <Readout label="bounds" value={`${fmt(c.m.lower ?? NaN)} … ${fmt(c.m.upper ?? NaN)}`} />
            <Readout label="1 − λ⋆" value={fmt(c.gap.absoluteGap)} />
          </>,
        ]),
      )}
      caption="d(t) = maxₓ ‖Pᵗ(x, ·) − π‖ for a lazy random walk on a cycle and on a path of n states, computed from matrix powers; it never increases. The dashed line is ε and the points mark each chain's mixing time. The path mixes more slowly than the cycle, because its ends are twice as far apart. Double n and the mixing times roughly quadruple; raise the hold probability and they grow as 1/(1 − hold)."
    >
      <Plot x={tAxis} y={dAxis}>
        {r.map((c, k) => (
          <Curve key={c.topology} name={`${c.topology}`} x={c.d.map((_, t) => t)} y={c.d} slot={k} />
        ))}
        {r.map((c, k) =>
          Number.isFinite(c.m.time) ? (
            <Points
              key={`${c.topology}-t`}
              name={`t_mix (${c.topology})`}
              x={[c.m.time]}
              y={[c.d[c.m.time]]}
              slot={k}
              size={10}
            />
          ) : null,
        )}
        <Annotation y={epsilon} text="ε" dashed />
      </Plot>
    </Figure>
  )
}
