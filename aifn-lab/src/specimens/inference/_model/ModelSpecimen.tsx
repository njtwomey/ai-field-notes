import { modelMarkovBlanket, toFactorGraph } from 'aifn/inference/model'
import { useMemo, useState } from 'react'
import { Select } from '@lab/controls'
import { Figure } from '@lab/layout'
import { FactorGraphView, PlateView } from '@lab/views'
import { Readout } from '@lab/viz'
import { MODELS } from './models'

type Key = keyof typeof MODELS
const OPTIONS = Object.entries(MODELS).map(([value, m]) => ({ value: value as Key, label: m.label }))

/** One description, two drawings: plate notation, and the factor graph of a small expansion with a Markov blanket. */
export function ModelSpecimen() {
  const [which, setWhich] = useState<Key>('lda')
  const entry = MODELS[which]
  const [focus, setFocus] = useState<Record<string, string>>({})
  const current = focus[which] ?? entry.focus
  const fg = useMemo(() => toFactorGraph(entry.model, entry.bindings), [entry])
  const latent = fg.attributes.filter((n) => n.role === 'latent').map((n) => n.name)
  const blanket = useMemo(() => modelMarkovBlanket(entry.model, current, entry.bindings), [entry, current])
  const list = (xs: readonly string[]) => (xs.length ? xs.join(', ') : '—')
  const choose = <Select label="model" value={which} onChange={setWhich} options={OPTIONS} />
  return (
    <>
      <Figure
        title="A model in plate notation"
        description="The description language's model, drawn by toDiagram: plates are groups labelled with their sizes, observed nodes are shaded."
        defaultSize="S"
        hoverReadout={false}
        controls={choose}
        caption="Circles are latent variables, shaded circles observed ones, small circles constants and dashed circles deterministic nodes. Arrows run from each node's parents. LDA's plates nest: words inside documents, topics on their own."
      >
        <PlateView graph={entry.model} positions={entry.positions} />
      </Figure>
      <Figure
        title="The expanded factor graph and a Markov blanket"
        description="toFactorGraph unrolls the plates against small data; each stochastic instance brings one factor, its conditional given its parents."
        defaultSize="M"
        hoverReadout={false}
        controls={
          <>
            {choose}
            <Select
              label="variable"
              value={current}
              onChange={(v) => setFocus({ ...focus, [which]: v })}
              options={latent}
            />
          </>
        }
        readouts={
          <>
            <Readout label="parents" value={list(blanket.parents)} />
            <Readout label="children" value={list(blanket.children)} />
            <Readout label="co-parents" value={list(blanket.coParents)} />
          </>
        }
        caption="Click a latent variable (or pick one): it becomes active, the variables of its Markov blanket and the factors joining them stay drawn, everything else dims. Its Gibbs conditional needs only these. In LDA the word w[d,n] depends on every topic φ[k] through z[d,n], so all topics are co-parents of z[d,n]."
      >
        <FactorGraphView
          graph={entry.model}
          bindings={entry.bindings}
          highlight={current}
          onVariableClick={(key) => latent.includes(key) && setFocus({ ...focus, [which]: key })}
          ariaLabel={`${entry.label} as a factor graph`}
        />
      </Figure>
    </>
  )
}
