import { modelMarkovBlanket, toFactorGraph } from 'aifn/inference/model'
import { useMemo } from 'react'
import { Select } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, row, useFigureState } from '@lab/state'
import { FactorGraphView, PlateView } from '@lab/views'
import { Readout } from '@lab/viz'
import { MODELS } from './models'

type Key = keyof typeof MODELS
const OPTIONS = Object.entries(MODELS).map(([value, m]) => ({ value: value as Key, label: m.label }))

/** One description, two drawings: plate notation, and the factor graph of a small expansion with a Markov blanket. */
export function ModelSpecimen() {
  const plate = useFigureState({ model: row('model', { which: choice(OPTIONS, 'lda', { label: 'model' }) }) })
  const which: Key = plate.model.which
  const entry = MODELS[which]
  const fg = useMemo(() => toFactorGraph(entry.model, entry.bindings), [entry])
  const latent = useMemo(() => fg.attributes.filter((n) => n.role === 'latent').map((n) => n.name), [fg])
  // The variable's choices depend on the model: the schema is rebuilt each render and the value re-validated.
  const graph = useFigureState({
    focus: row('variable', { variable: choice(latent, entry.focus, { label: 'variable' }) }),
  })
  const current = latent.includes(graph.focus.variable) ? graph.focus.variable : entry.focus
  const blanket = useMemo(() => modelMarkovBlanket(entry.model, current, entry.bindings), [entry, current])
  const list = (xs: readonly string[]) => (xs.length ? xs.join(', ') : '—')
  // The same model choice above the second figure, writing the first figure's state.
  const choose = (
    <Select label="model" value={which} onChange={(v: Key) => plate.set('model.which', v)} options={OPTIONS} />
  )
  return (
    <>
      <Figure
        title="A model in plate notation"
        purpose="One model description drawn in plate notation: each plate is a group labelled with its size, so a repeated structure is drawn once; observed nodes are shaded."
        defaultSize="S"
        hoverReadout={false}
        state={plate}
        caption="Circles are latent variables, shaded circles observed ones, small circles constants and dashed circles deterministic nodes. Arrows run from each node's parents. LDA's plates nest: words inside documents, topics on their own."
      >
        <PlateView graph={entry.model} positions={entry.positions} />
      </Figure>
      <Figure
        title="The expanded factor graph and a Markov blanket"
        purpose="Unrolling the plates against small data gives a factor graph with one factor per stochastic instance; a variable's Markov blanket (parents, children, co-parents) is all its Gibbs conditional needs."
        defaultSize="M"
        hoverReadout={false}
        state={graph}
        controls={choose}
        readouts={{
          'Markov blanket': (
            <>
              <Readout label="parents" value={list(blanket.parents)} />
              <Readout label="children" value={list(blanket.children)} />
              <Readout label="co-parents" value={list(blanket.coParents)} />
            </>
          ),
        }}
        caption="Click a latent variable (or pick one): it becomes active, the variables of its Markov blanket and the factors joining them stay drawn, everything else dims. Its Gibbs conditional needs only these. In LDA the word w[d,n] depends on every topic φ[k] through z[d,n], so all topics are co-parents of z[d,n]."
      >
        <FactorGraphView
          graph={entry.model}
          bindings={entry.bindings}
          highlight={current}
          onVariableClick={(key) => latent.includes(key) && graph.set('focus.variable', key)}
          ariaLabel={`${entry.label} as a factor graph`}
        />
      </Figure>
    </>
  )
}
