import { useMemo, useState } from 'react'
import { Player, Select, Slider, Switch } from 'aifn-render/controls'
import { Diagram, factor, link, variable, type DiagramSpec, type ElementState } from 'aifn-render/diagram'
import { Figure } from 'aifn-render/layout'
import { TreeView } from '@lab/views'
import { binaryTree, pathToRoot, treeFromNested, treeFromParents, type NestedBinaryTree } from 'aifn-compute/graph'
import { Section } from './Section'

/** Latent Dirichlet allocation in plate notation: documents d hold word positions n; topics k have their own plate. */
const lda: DiagramSpec = {
  nodes: [
    variable('alpha', 0, 2.4, '$\\alpha$', { w: 0.7, h: 0.7, small: true }),
    variable('theta', 1.7, 2.4, '$\\boldsymbol{\\theta}_d$'),
    variable('z', 3.5, 2.4, '$z_{dn}$'),
    variable('w', 5.3, 2.4, '$w_{dn}$', { filled: true }),
    variable('phi', 5.3, -0.7, '$\\boldsymbol{\\phi}_k$'),
    variable('beta', 7.3, -0.7, '$\\beta$', { w: 0.7, h: 0.7, small: true }),
  ],
  edges: [link('alpha', 'theta'), link('theta', 'z'), link('z', 'w'), link('phi', 'w'), link('beta', 'phi')],
  groups: [
    { id: 'N', label: '$N_d$', tone: 'ink', around: ['z', 'w'], pad: 0.3, labelAt: 'bottom-right' },
    { id: 'D', label: '$D$', tone: 'ink', around: ['theta', 'z', 'w'], pad: 0.75, labelAt: 'bottom-right' },
    { id: 'K', label: '$K$', tone: 'ink', around: ['phi'], pad: 0.3, labelAt: 'bottom-right' },
  ],
}

/**
 * A chain of three variables as a factor graph, p(x₁, x₂, x₃) ∝ φ₁(x₁) ψ₁₂(x₁, x₂) ψ₂₃(x₂, x₃): unary factors above,
 * pair factors between the variables, undirected links.
 */
const chainFactorGraph: DiagramSpec = {
  nodes: [
    factor('phi1', 0, 0, '$\\phi_1$', 'n'),
    variable('x1', 0, 1.8, '$x_1$'),
    factor('psi12', 2, 1.8, '$\\psi_{12}$', 's'),
    variable('x2', 4, 1.8, '$x_2$', { filled: true }),
    factor('psi23', 6, 1.8, '$\\psi_{23}$', 's'),
    variable('x3', 8, 1.8, '$x_3$'),
  ],
  edges: [
    link('phi1', 'x1', false),
    link('x1', 'psi12', false),
    link('psi12', 'x2', false),
    link('x2', 'psi23', false),
    link('psi23', 'x3', false),
  ],
}

/** The plate model, as specified by hand. */
function PlateFigure() {
  return (
    <Figure
      // TODO(5c): state the purpose (placeholder: the title)
      purpose="Plate notation: latent Dirichlet allocation"
      title="Plate notation: latent Dirichlet allocation"
      defaultSize="S"
      hoverReadout={false}
      caption="A hand-placed spec: variables on a grid, straight directed links, plates drawn around nodes with their counts at the bottom right. The shaded node is observed."
    >
      <Diagram spec={lda} ariaLabel="LDA in plate notation" />
    </Figure>
  )
}

/** The factors each variable of the chain touches. */
const TOUCHING: Record<string, string[]> = { x1: ['phi1', 'psi12'], x2: ['psi12', 'psi23'], x3: ['psi23'] }

/** A factor graph with a variable to condition on: its neighbouring factors take the active state. */
function FactorGraphFigure() {
  const [target, setTarget] = useState<'x1' | 'x2' | 'x3'>('x2')
  const spec = useMemo((): DiagramSpec => {
    const on = new Set([target, ...TOUCHING[target]])
    return {
      ...chainFactorGraph,
      nodes: chainFactorGraph.nodes.map((n) => ({ ...n, filled: false, state: on.has(n.id) ? 'active' : 'done' })),
      edges: chainFactorGraph.edges?.map((e) => ({
        ...e,
        state: on.has(e.from) && on.has(e.to) ? 'active' : 'done',
      })),
    }
  }, [target])
  return (
    <Figure
      // TODO(5c): state the purpose (placeholder: the title)
      purpose="A factor graph: the factors touching a variable"
      title="A factor graph: the factors touching a variable"
      defaultSize="S"
      hoverReadout={false}
      controls={<Select label="variable" value={target} onChange={setTarget} options={['x1', 'x2', 'x3']} />}
      caption="Pick a variable, or click one: it and the factors it touches take the active state (its full conditional is their product). The change of state fades in."
    >
      <Diagram
        spec={spec}
        ariaLabel="A chain factor graph of three variables"
        onNodeClick={(id) => id in TOUCHING && setTarget(id as 'x1' | 'x2' | 'x3')}
      />
    </Figure>
  )
}

/** A small DAG given as data, in a deliberately scrambled order, placed by the layered layout. */
const DAG: [string, string][] = [
  ['x', 'h1'],
  ['y', 'loss'],
  ['h2', 'out'],
  ['x', 'h2'],
  ['w1', 'h1'],
  ['h1', 'h3'],
  ['w2', 'h2'],
  ['h2', 'h3'],
  ['h3', 'out'],
  ['out', 'loss'],
  ['w3', 'h3'],
  ['h1', 'out'],
]
const DAG_NODES = ['loss', 'h3', 'x', 'out', 'w2', 'h1', 'y', 'w3', 'h2', 'w1']

/** The DAG's nodes in topological order (for stepping through it). */
function topological(): string[] {
  const indegree = new Map(DAG_NODES.map((n) => [n, 0]))
  for (const [, b] of DAG) indegree.set(b, indegree.get(b)! + 1)
  const queue = DAG_NODES.filter((n) => indegree.get(n) === 0)
  const order: string[] = []
  while (queue.length) {
    const n = queue.shift()!
    order.push(n)
    for (const [a, b] of DAG) {
      if (a !== n) continue
      indegree.set(b, indegree.get(b)! - 1)
      if (indegree.get(b) === 0) queue.push(b)
    }
  }
  return order
}
const ORDER = topological()

function LayeredFigure() {
  const [direction, setDirection] = useState<'right' | 'down'>('right')
  const [sweeps, setSweeps] = useState(8)
  const [compact, setCompact] = useState(true)
  const [reverse, setReverse] = useState(false)
  const [step, setStep] = useState(ORDER.length)
  const spec = useMemo((): DiagramSpec => {
    const rank = new Map(ORDER.map((n, i) => [n, i]))
    const state = (id: string): ElementState =>
      rank.get(id)! < step - 1 ? 'done' : rank.get(id)! === step - 1 ? 'active' : 'idle'
    return {
      layout: 'layered',
      layered: { direction, sweeps, compactSources: compact },
      nodes: DAG_NODES.map((id) => ({
        id,
        shape: id.startsWith('w') ? 'box' : 'circle',
        w: id.startsWith('w') ? 0.9 : undefined,
        h: id.startsWith('w') ? 0.6 : undefined,
        label: `$${id.replace(/(\d)/, '_$1')}$`,
        tone: id.startsWith('w') ? 1 : 0,
        state: state(id),
        notes: { [direction === 'right' ? 'n' : 'e']: `$\\#${rank.get(id)! + 1}$` },
      })),
      edges: DAG.map(([from, to]) => ({
        from,
        to,
        route: 'straight',
        arrow: 'mid',
        reverse,
        state: state(to) === 'idle' ? 'idle' : state(to) === 'active' ? 'active' : 'done',
      })),
    }
  }, [direction, sweeps, compact, reverse, step])
  return (
    <Figure
      // TODO(5c): state the purpose (placeholder: the title)
      purpose="Layered layout of a graph given as data"
      title="Layered layout of a graph given as data"
      defaultSize="M"
      hoverReadout={false}
      controls={
        <>
          <Select label="direction" value={direction} onChange={setDirection} options={['right', 'down']} />
          <Slider label="ordering sweeps" value={sweeps} min={0} max={8} step={1} onChange={setSweeps} />
          <Switch label="sources next to their consumers" checked={compact} onChange={setCompact} />
          <Switch label="reverse the arrows" checked={reverse} onChange={setReverse} />
          <div className="col-span-full">
            <Player label="visit in topological order" value={step} onChange={setStep} count={ORDER.length + 1} />
          </div>
        </>
      }
      caption="The nodes are listed in a scrambled order and given no positions. Each goes into the column of its topological depth; barycentre sweeps order each column to cut crossings (set sweeps to 0 to see the listed order). Edges spanning several columns bend through waypoints. The player steps through the nodes in topological order: visited nodes are done, the current one active, the rest idle. Arrows sit mid-edge and can be reversed without rerouting."
    >
      <Diagram spec={spec} ariaLabel="A layered layout of a small computation DAG" />
    </Figure>
  )
}

/** A binary search tree of keys inserted as 50, 30, 20, 10, 25, 40, 35, 45, 70, 80, 90, 85, 5, 2: lopsided on purpose. */
const bst: NestedBinaryTree = (() => {
  const keys = [50, 30, 20, 10, 25, 40, 35, 45, 70, 80, 90, 85, 5, 2]
  type Node = { label: string; key: number; left?: Node; right?: Node }
  const root: Node = { label: String(keys[0]), key: keys[0] }
  for (const k of keys.slice(1)) {
    let n = root
    for (;;) {
      const side = k < n.key ? 'left' : 'right'
      if (!n[side]) {
        n[side] = { label: String(k), key: k }
        break
      }
      n = n[side]!
    }
  }
  return root
})()
const BST = binaryTree(bst)

/** A wide n-ary tree: 3 to 6 children per node for three levels (fixed, not random). */
const WIDE = treeFromParents(
  (() => {
    const parents = [-1]
    let frontier = [0]
    for (let level = 0; level < 3; level++) {
      const next: number[] = []
      frontier.forEach((p, i) => {
        const kids = level === 2 ? (i % 3 === 0 ? 3 : 0) : 3 + ((p + i) % 4)
        for (let k = 0; k < kids; k++) {
          next.push(parents.length)
          parents.push(p)
        }
      })
      frontier = next
    }
    return parents
  })(),
)

/** A small dendrogram: six points merged at increasing distances (hand-set heights). */
const DENDROGRAM = treeFromNested({
  label: '',
  height: 4.2,
  children: [
    {
      label: '',
      height: 1.6,
      children: [
        { label: '$x_1$', height: 0 },
        {
          label: '',
          height: 0.7,
          children: [
            { label: '$x_2$', height: 0 },
            { label: '$x_3$', height: 0 },
          ],
        },
      ],
    },
    {
      label: '',
      height: 2.5,
      children: [
        {
          label: '',
          height: 0.4,
          children: [
            { label: '$x_4$', height: 0 },
            { label: '$x_5$', height: 0 },
          ],
        },
        { label: '$x_6$', height: 0 },
      ],
    },
  ],
})

/** The tidy tree layout on one of three trees: lopsided binary, wide n-ary (collapsible) or a dendrogram by height. */
type TreeKind = 'unbalanced binary' | 'wide n-ary' | 'dendrogram'
const TREE_TITLE: Record<TreeKind, string> = {
  'unbalanced binary': 'Tidy layout of an unbalanced binary search tree',
  'wide n-ary': 'Tidy layout of a wide tree, with collapsing',
  dendrogram: 'A dendrogram: nodes placed by merge height',
}
const TREE_CAPTION: Record<TreeKind, string> = {
  'unbalanced binary':
    'Keys inserted as 50, 30, 20, 10, 25, 40, 35, 45, 70, 80, 90, 85, 5, 2. treeLayout (Buchheim, Jünger and Leipert’s linear-time Walker algorithm) centres each parent over its children and pushes subtrees apart only as far as their contours need; a lone child keeps its side (5 left of 10, 80 right of 70). Hover a node to light its path to the root.',
  'wide n-ary':
    'Three to six children per node. Small subtrees between two large ones are spaced evenly. Click an internal node to collapse or expand it: a collapsed node is dashed and notes how many nodes it hides.',
  dendrogram:
    'heightAxis places each node by its height (a merge distance) instead of its depth, with elbow edges, as a hierarchical clustering draws its merges.',
}

function TreeLayoutFigure({ which }: { which: TreeKind }) {
  const [orientation, setOrientation] = useState<'down' | 'right'>('down')
  const [gap, setGap] = useState(0.35)
  const [hovered, setHovered] = useState<number | null>(null)
  const tree = which === 'unbalanced binary' ? BST : which === 'wide n-ary' ? WIDE : DENDROGRAM
  const path = hovered === null || hovered >= tree.nodes.length ? [] : pathToRoot(tree, hovered)
  return (
    <Figure
      // TODO(5c): state the purpose (placeholder: the title)
      purpose={TREE_TITLE[which]}
      title={TREE_TITLE[which]}
      defaultSize="M"
      hoverReadout={false}
      controls={
        <>
          <Select label="orientation" value={orientation} onChange={setOrientation} options={['down', 'right']} />
          <Slider label="sibling gap" value={gap} min={0.1} max={1.5} onChange={setGap} />
        </>
      }
      caption={TREE_CAPTION[which]}
    >
      <TreeView
        tree={tree}
        orientation={orientation}
        siblingGap={gap}
        heightAxis={which === 'dendrogram'}
        shape={which === 'dendrogram' ? (v) => (DENDROGRAM.nodes[v].children.length ? 'dot' : 'circle') : undefined}
        nodeLabels={which === 'wide n-ary' ? () => '' : undefined}
        collapsible={which === 'wide n-ary'}
        defaultCollapsed={which === 'wide n-ary' ? [4] : undefined}
        highlight={path}
        onNodeHover={setHovered}
        ariaLabel={`A ${which} tree`}
      />
    </Figure>
  )
}

/** The Diagrams page: the lab's own diagram system, exercised. */
export function DiagramsKit() {
  return (
    <div className="flex flex-col gap-10">
      <Section
        title="Hand-specified diagrams"
        description="Diagrams are not charts: nodes on a grid, ports, orthogonal, straight or curved edges, KaTeX labels. They scale to the figure frame."
      >
        <PlateFigure />
        <FactorGraphFigure />
      </Section>
      <Section
        title="Automatic layered layout"
        description="layout: 'layered' places graphs generated from data: columns by topological depth, crossings reduced by barycentre sweeps."
      >
        <LayeredFigure />
      </Section>
      <Section
        title="Tidy tree layout"
        description="treeLayout and TreeView draw any aifn/graph Tree: parents centred over children, binary trees keeping left and right, dendrograms by height."
      >
        <TreeLayoutFigure which="unbalanced binary" />
        <TreeLayoutFigure which="wide n-ary" />
        <TreeLayoutFigure which="dendrogram" />
      </Section>
    </div>
  )
}
