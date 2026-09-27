import { useState, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { PageContainer } from '@/components/layout/AppShell'
import { Diagram } from '@/components/diagram/Diagram'
import { gru, lstm } from '@/components/diagram/specs/recurrent'
import { transformer } from '@/components/diagram/specs/transformer'
import { autoencoder, latentDiffusion, vae } from '@/components/diagram/specs/generative'
import { lda } from '@/components/diagram/specs/graphical'
import type { DiagramSpec } from '@/components/diagram/types'

const factorGraph: DiagramSpec = {
  nodes: [
    { id: 'a', x: 0, y: 2, shape: 'circle', label: '$x_1$', tone: 'ink' },
    { id: 'b', x: 2, y: 0, shape: 'circle', label: '$x_2$', tone: 'ink' },
    { id: 'c', x: 4, y: 2, shape: 'circle', label: '$x_3$', tone: 'ink', filled: true },
    { id: 'fab', x: 1, y: 1, shape: 'factor', label: '$\\psi_{12}$', labelSide: 'w' },
    { id: 'fbc', x: 3, y: 1, shape: 'factor', label: '$\\psi_{23}$', labelSide: 'e' },
    { id: 'fac', x: 2, y: 2, shape: 'factor', label: '$\\psi_{13}$', labelSide: 's' },
  ],
  edges: [
    { from: 'a', to: 'fab', route: 'straight', arrow: 'none' },
    { from: 'fab', to: 'b', route: 'straight', arrow: 'none' },
    { from: 'b', to: 'fbc', route: 'straight', arrow: 'none' },
    { from: 'fbc', to: 'c', route: 'straight', arrow: 'none' },
    { from: 'a', to: 'fac', route: 'straight', arrow: 'none' },
    { from: 'fac', to: 'c', route: 'straight', arrow: 'none' },
    { from: 'b', to: 'c', route: 'curve', bend: 1.1, dashed: true },
  ],
}

const ALARM: Record<string, { x: number; y: number; label: string }> = {
  B: { x: 0, y: 0, label: '$B$' },
  E: { x: 3, y: 0, label: '$E$' },
  A: { x: 1.5, y: 1.6, label: '$A$' },
  J: { x: 0, y: 3.2, label: '$J$' },
  M: { x: 3, y: 3.2, label: '$M$' },
}
const ALARM_EDGES: [string, string][] = [
  ['B', 'A'],
  ['E', 'A'],
  ['A', 'J'],
  ['A', 'M'],
]

/** Parents, children and the children's other parents. */
function markovBlanket(v: string): Set<string> {
  const parents = ALARM_EDGES.filter(([, t]) => t === v).map(([s]) => s)
  const children = ALARM_EDGES.filter(([s]) => s === v).map(([, t]) => t)
  const coParents = ALARM_EDGES.filter(([s, t]) => children.includes(t) && s !== v).map(([s]) => s)
  return new Set([...parents, ...children, ...coParents])
}

function MarkovBlanketDemo() {
  const [focus, setFocus] = useState('A')
  const blanket = markovBlanket(focus)
  const spec: DiagramSpec = {
    nodes: Object.entries(ALARM).map(([id, n]) => ({
      id,
      ...n,
      shape: 'circle' as const,
      tone: 'ink' as const,
      filled: blanket.has(id),
      highlight: id === focus,
    })),
    edges: ALARM_EDGES.map(([s, t]) => ({
      from: s,
      to: t,
      route: 'straight' as const,
      highlight: s === focus || t === focus,
    })),
  }
  return (
    <>
      <Diagram
        spec={spec}
        ariaLabel="Alarm network with the Markov blanket of the chosen node shaded"
        onNodeClick={setFocus}
      />
      <p className="mt-2 text-xs text-muted-foreground">
        Click a node. Its Markov blanket is shaded: {[...blanket].join(', ') || 'none'}.
      </p>
    </>
  )
}

const CHAIN_P = [
  [0.7, 0.2, 0.1],
  [0.3, 0.4, 0.3],
  [0.2, 0.3, 0.5],
]
const CHAIN_AT: [number, number, string][] = [
  [0, 0, 'w'],
  [4.4, 0, 'e'],
  [2.2, 3.4, 's'],
]

/** Self-loops and shading: a three-state chain whose nodes are shaded by the distribution after t steps. */
function MarkovChainDemo() {
  const [t, setT] = useState(0)
  let d = [1, 0, 0]
  for (let k = 0; k < t; k++) d = [0, 1, 2].map((j) => d.reduce((s, v, i) => s + v * CHAIN_P[i][j], 0))
  const spec: DiagramSpec = {
    nodes: CHAIN_AT.map(([x, y], i) => ({
      id: `s${i}`,
      x,
      y,
      shape: 'circle' as const,
      tone: i,
      label: `$${d[i].toFixed(2)}$`,
      shade: d[i],
    })),
    edges: CHAIN_P.flatMap((row, i) =>
      row.map((p, j) => ({
        from: i === j ? `s${i}:${CHAIN_AT[i][2]}` : `s${i}`,
        to: `s${j}`,
        route: 'curve' as const,
        bend: 0.55,
        labelSide: 'right' as const,
        label: p.toFixed(1),
      })),
    ),
  }
  return (
    <>
      <Diagram spec={spec} ariaLabel="Three-state Markov chain with self-loops, shaded by probability" />
      <Button size="sm" variant="outline" className="mt-2" onClick={() => setT((v) => (v + 1) % 8)}>
        step (t = {t})
      </Button>
    </>
  )
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border p-5">
      <h2 className="mb-4 text-sm font-medium">{title}</h2>
      {children}
    </section>
  )
}

/** Test bench for the diagram library: architectures, projectors, the reparameterisation block and graphical models. */
export function DiagramLabPage() {
  return (
    <PageContainer>
      <h1 className="font-prose text-3xl font-bold">Diagram lab</h1>
      <p className="mt-2 mb-8 text-sm text-muted-foreground">
        Hand-specified diagrams: every node placed on a grid, groups drawn around nodes, edges routed through ports and
        waypoints, labels in KaTeX. Not linked from the site.
      </p>
      <div className="grid gap-6">
        <Panel title="LSTM cell">
          <Diagram spec={lstm} ariaLabel="LSTM cell" />
        </Panel>
        <Panel title="GRU cell">
          <Diagram spec={gru} ariaLabel="GRU cell" />
        </Panel>
        <Panel title="Transformer (Vaswani et al. 2017)">
          <Diagram spec={transformer} ariaLabel="Transformer encoder and decoder" />
        </Panel>
        <Panel title="Latent diffusion">
          <Diagram spec={latentDiffusion} ariaLabel="Latent diffusion model" />
        </Panel>
        <Panel title="Autoencoder">
          <Diagram spec={autoencoder} ariaLabel="Autoencoder" />
        </Panel>
        <Panel title="Variational autoencoder with the reparameterisation block">
          <Diagram spec={vae} ariaLabel="Variational autoencoder" />
        </Panel>
        <div className="grid gap-6 md:grid-cols-2">
          <Panel title="LDA (plates)">
            <Diagram spec={lda} ariaLabel="Latent Dirichlet allocation plate diagram" />
          </Panel>
          <Panel title="Factor graph">
            <Diagram spec={factorGraph} ariaLabel="Factor graph on three variables" />
          </Panel>
          <Panel title="Markov blanket (clickable)">
            <MarkovBlanketDemo />
          </Panel>
          <Panel title="Markov chain (self-loops, shade)">
            <MarkovChainDemo />
          </Panel>
        </div>
      </div>
    </PageContainer>
  )
}
