import { useMemo, useState } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import {
  bayesianRuleListTrace,
  mineAntecedents,
  ruleListLogPosterior,
  ruleProbabilities,
  type Antecedent,
  nominalConditions,
  ruleText,
} from 'aifn-compute/learning/rules'
import { titanicRows, TITANIC_LEVELS, TITANIC_NAMES } from '../_shared/rule-figures'

const LENGTHS = [1, 2] as const

const text = (a: Antecedent) => ruleText(a.conditions, { names: TITANIC_NAMES, levels: TITANIC_LEVELS })

/** Metropolis-Hastings over decision lists for Titanic survival: the chain, the current list and its rule probabilities. */
export function BrlSampler() {
  const fig = useFigureState({
    lambda: float(3, { min: 0.5, max: 12, step: 0.5, label: 'prior mean length λ', suggestions: [1, 3, 8] }),
    eta: float(1, { min: 0.5, max: 3, step: 0.5, label: 'prior mean conditions η', suggestions: [1, 2] }),
    maxLength: choice(LENGTHS, 2, { label: 'conditions per antecedent, at most' }),
    steps: int(2000, { min: 100, max: 20000, label: 'MCMC steps', suggestions: [500, 2000, 10000] }),
    seed: int(1, { min: 1, max: 999, label: 'chain seed' }),
  })
  const { lambda, eta, maxLength, steps, seed } = fig

  const data = useMemo(() => titanicRows(), [])
  const ants = useMemo(
    () => mineAntecedents(data.rows, nominalConditions([4, 2, 2]), { minSupport: 0.01, maxLength }),
    [data, maxLength],
  )
  const run = useComputed(
    () => bayesianRuleListTrace(ants, data.y, 2, stream(`brl-${seed}`), { lambda, eta, alpha: 1, steps }),
    [ants, data, lambda, eta, steps, seed],
    { mode: 'release' },
  )
  const chain = run.value
  const lengths = useMemo(() => {
    const kept = chain.slice(Math.floor(chain.length / 2))
    const counts = new Map<number, number>()
    for (const s of kept) counts.set(s.list.length, (counts.get(s.list.length) ?? 0) + 1)
    const xs = [...counts.keys()].sort((a, b) => a - b)
    return { x: xs, y: xs.map((k) => (counts.get(k) ?? 0) / kept.length) }
  }, [chain])

  const settings = `${lambda}/${eta}/${maxLength}/${steps}/${seed}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const at = pos.settings === settings ? Math.min(pos.at, chain.length - 1) : 0
  const s = chain[at]
  const describe = (list: number[]) => {
    const score = ruleListLogPosterior(list, ants, data.y, 2, { lambda, eta, alpha: 1 })
    const probs = ruleProbabilities(score.counts)
    return {
      score,
      lines: [
        ...list.map((j, k) => `${k === 0 ? 'if' : 'else if'} ${text(ants[j])} then ${formatNumber(probs[k][1])}`),
        `else ${formatNumber(probs[list.length][1])}`,
      ],
      counts: score.counts,
    }
  }
  const current = useMemo(() => describe(s?.list ?? []), [s, ants, data, lambda, eta]) // eslint-disable-line react-hooks/exhaustive-deps
  const best = useMemo(() => describe(s?.best ?? []), [s, ants, data, lambda, eta]) // eslint-disable-line react-hooks/exhaustive-deps

  // Thin the trace for drawing: at most 600 points.
  const stride = Math.max(1, Math.ceil(chain.length / 600))
  const shown = chain.filter((_, t) => t % stride === 0 || t === chain.length - 1)
  const upTo = shown.filter((c) => c.t <= at)

  const xa = useAxis({ label: 'MCMC step', range: [0, chain.length - 1], integer: true })
  const ya = useAxis({ label: 'log posterior (nats)', hold: 'union' })
  const xb = useAxis({ label: 'list length m', integer: true })
  const yb = useAxis({ label: 'share of the second half', range: [0, 1] })
  if (!s) return null

  return (
    <Figure
      title="Sampling decision lists for Titanic survival"
      state={fig}
      caption={`Metropolis-Hastings over decision lists built from the ${ants.length} antecedents of at most ${maxLength} conditions with support of at least 1% (class, sex and age of the 2201 people aboard). Each step proposes to move, add or remove one antecedent and accepts with the Metropolis-Hastings probability; the chain starts from the empty list. Left: the log posterior of the current list (accepted moves only change it) and of the best list so far. Right: the length of the sampled lists over the second half of the chain. The readouts give the current and the best list with each rule's posterior mean survival probability. λ moves the sampled lengths: with λ = 1 the best list found is two rules (adult men, then 3rd class), with λ = 8 and η = 2 the chain wanders among lists of about ten rules, a third of which capture nobody because earlier rules took their rows. With antecedents of one condition the lists must chain several rules to reach the interaction of sex and age that one two-condition antecedent states directly, and the best posterior is lower. Step through with the player.`}
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={chain.length} label="step" />}
      readouts={{
        [`step ${s.t}: proposal ${s.proposal}, ${s.accepted ? 'accepted' : 'rejected'}`]: (
          <>
            <Readout
              label="log prior + log likelihood"
              value={`${formatNumber(current.score.logPrior)} + ${formatNumber(current.score.logLikelihood)} = ${formatNumber(current.score.logPosterior)}`}
            />
            {current.lines.map((l, k) => (
              <Readout
                key={k}
                label={`${k + 1}`}
                value={`${l}  (${current.counts[k][0]} died : ${current.counts[k][1]} survived)`}
              />
            ))}
          </>
        ),
        [`best list so far, log posterior ${formatNumber(s.bestLogPosterior)}`]: (
          <>
            {best.lines.map((l, k) => (
              <Readout key={k} label={`${k + 1}`} value={l} />
            ))}
          </>
        ),
      }}
    >
      <Plots cols={2} widths={[62, 38]}>
        <Plot x={xa} y={ya}>
          <Curve
            name="current list"
            x={upTo.map((c) => c.t)}
            y={upTo.map((c) => c.logPosterior)}
            slot={0}
            stale={run.stale}
          />
          <Curve name="best so far" x={upTo.map((c) => c.t)} y={upTo.map((c) => c.bestLogPosterior)} emphasis dashed />
        </Plot>
        <Plot x={xb} y={yb}>
          <Bars name="list length" x={lengths.x} y={lengths.y} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
