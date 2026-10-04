import { Diagram, Figure } from 'aifn-render'
import type { DiagramEdge, DiagramNode, DiagramSpec, Tone } from 'aifn-render'

const user = (id: string, y: number, label: string): DiagramNode => ({
  id,
  x: 1,
  y,
  shape: 'circle',
  w: 0.9,
  h: 0.9,
  label,
  tone: 0,
})
const dish = (id: string, y: number, label: string): DiagramNode => ({
  id,
  x: 7,
  y,
  w: 2.4,
  h: 0.7,
  label,
  tone: 1,
})
const order = (from: string, to: string, n: number): DiagramEdge => ({
  from,
  to,
  route: 'straight',
  arrow: 'none',
  label: String(n),
})

const bipartite: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'ut', x: 1, y: 0.3, shape: 'text', small: true, label: 'users' },
    { id: 'dt', x: 7, y: 0.3, shape: 'text', small: true, label: 'dishes' },
    user('u1', 1.4, '$u_1$'),
    user('u2', 3, '$u_2$'),
    user('u3', 4.6, '$u_3$'),
    dish('d1', 1.2, 'chicken tandoori'),
    dish('d2', 2.4, 'garlic naan'),
    dish('d3', 3.6, 'ma po tofu'),
    dish('d4', 4.8, 'cobb salad'),
    {
      id: 'note',
      x: 4,
      y: 5.9,
      shape: 'text',
      small: true,
      label: 'edge weight = number of orders; one graph per city',
    },
  ],
  edges: [
    order('u1', 'd1', 10),
    order('u1', 'd2', 3),
    order('u2', 'd2', 1),
    order('u2', 'd3', 4),
    order('u3', 'd3', 2),
    order('u3', 'd4', 6),
  ],
}

/** The user–dish bipartite graph with order-count edge weights. */
export function BipartiteDiagram() {
  return (
    <Figure
      title="The user–dish graph"
      caption="Users and dishes are the two node types; an edge joins a user to every dish they ordered, weighted by how many times they ordered it (ratings also inform the weights). A second graph does the same for users and restaurants, and each city gets its own graphs because different cities are only loosely connected. Redrawn from Liu et al. (2019), Uber Engineering."
    >
      <Diagram
        spec={bipartite}
        ariaLabel="Bipartite graph of three users and four dishes with order counts on the edges"
      />
    </Figure>
  )
}

const box = (id: string, x: number, y: number, label: string, tone: number | 'neutral', w = 1.6): DiagramNode => ({
  id,
  x,
  y,
  w,
  h: 0.6,
  label,
  tone,
  small: true,
})

const tree: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'x', x: 3.5, y: 7.4, w: 6.6, h: 0.6, label: 'features $\\xvec$ of nodes two hops from A', tone: 'neutral' },
    box('proj0', 3.5, 6.3, 'type projection $\\Pmat_{\\text{type}}$', 2, 3.6),
    box('agg1', 2, 5.1, 'weighted pool', 'neutral'),
    box('w1', 2, 4.1, 'PROJ $\\Wmat_1$', 0),
    box('b1', 5.2, 4.1, 'PROJ $\\Bmat_1$', 1),
    { id: 'self1', x: 5.2, y: 5.1, shape: 'text', small: true, label: 'the hop-1 node itself' },
    { id: 'cat1', x: 3.5, y: 3, shape: 'op', label: '$\\Vert$' },
    { id: 'h1', x: 3.5, y: 2.2, shape: 'text', label: "$\\hvec^{(1)}$ of A's neighbours" },
    box('agg2', 2, 1.2, 'weighted pool', 'neutral'),
    box('w2', 2, 0.2, 'PROJ $\\Wmat_2$', 0),
    box('b2', 5.2, 0.2, 'PROJ $\\Bmat_2$', 1),
    { id: 'self2', x: 5.2, y: 1.2, shape: 'text', small: true, label: 'node A itself' },
    { id: 'cat2', x: 3.5, y: -0.9, shape: 'op', label: '$\\Vert$' },
    { id: 'hA', x: 3.5, y: -1.8, shape: 'text', label: '$\\zvec_A = \\hvec^{(2)}_A$' },
    { id: 'l1', x: -0.4, y: 4.6, shape: 'text', small: true, label: 'layer 1' },
    { id: 'l2', x: -0.4, y: 0.7, shape: 'text', small: true, label: 'layer 2' },
  ],
  edges: [
    { from: 'x', to: 'proj0' },
    { from: 'proj0:s', to: 'agg1:n', via: [[2, 5.7]] },
    { from: 'proj0:s', to: 'self1:n', via: [[5.2, 5.7]] },
    { from: 'agg1', to: 'w1' },
    { from: 'self1', to: 'b1' },
    { from: 'w1:s', to: 'cat1:w', via: [[2, 3]] },
    { from: 'b1:s', to: 'cat1:e', via: [[5.2, 3]] },
    { from: 'cat1', to: 'h1' },
    { from: 'h1:s', to: 'agg2:n', via: [[2, 1.8]] },
    { from: 'h1:s', to: 'self2:n', via: [[5.2, 1.8]] },
    { from: 'agg2', to: 'w2' },
    { from: 'self2', to: 'b2' },
    { from: 'w2:s', to: 'cat2:w', via: [[2, -0.9]] },
    { from: 'b2:s', to: 'cat2:e', via: [[5.2, -0.9]] },
    { from: 'cat2', to: 'hA' },
  ],
}

/** The two-layer computation that embeds node A. */
export function ComputationTree() {
  return (
    <Figure
      title="Embedding one node with two layers"
      caption="To embed node A, collect nodes one and two hops away. A type-specific projection maps users, dishes and restaurants to vectors of one size. Each layer pools a node's neighbours (weighted by the edges), projects the pooled vector with W, projects the node's own vector with B, and concatenates the two; layer 2 repeats this with new matrices to produce A's embedding. Redrawn from Liu et al. (2019), Uber Engineering."
    >
      <Diagram
        spec={tree}
        ariaLabel="Two-layer GraphSAGE computation: projected features, weighted pooling, W and B projections, concatenation, twice"
      />
    </Figure>
  )
}

const step = (id: string, x: number, y: number, label: string, tone: Tone, w = 2.8): DiagramNode => ({
  id,
  x,
  y,
  w,
  h: 1,
  label,
  tone,
  small: true,
})

const pipeline: DiagramSpec = {
  unit: 36,
  nodes: [
    step('hive', 1.5, 1, 'Hive tables\n(orders)', 'neutral', 2.2),
    step('parquet', 4.7, 1, 'Parquet on HDFS\nnodes, edges, timestamps', 0),
    step('cypher', 8.2, 1, 'latest properties at a date\nCypher format on HDFS', 0, 3.2),
    step('spark', 11.9, 1, 'Spark runs Cypher\ngraphs per city', 0),
    step('nx', 11.9, 3.2, 'NetworkX graphs', 1),
    step('tf', 8.2, 3.2, 'TensorFlow on GPUs\ntrain GNN, embed nodes', 1, 3.2),
    step('lookup', 4.7, 3.2, 'embedding\nlookup table', 2),
    step('ranker', 1.5, 3.2, 'personalised ranker\n(online request)', 'ink', 2.2),
    { id: 'offline', x: 6.7, y: 0, shape: 'text', small: true, label: 'data pipeline (offline)' },
    { id: 'online', x: 6.7, y: 4.3, shape: 'text', small: true, label: 'training pipeline and serving' },
  ],
  edges: [
    { from: 'hive', to: 'parquet' },
    { from: 'parquet', to: 'cypher' },
    { from: 'cypher', to: 'spark' },
    { from: 'spark', to: 'nx' },
    { from: 'nx', to: 'tf' },
    { from: 'tf', to: 'lookup' },
    { from: 'lookup', to: 'ranker', label: 'dot, cosine' },
  ],
}

/** From order tables to embeddings served to the ranker. */
export function PipelineDiagram() {
  return (
    <Figure
      title="Data and training pipeline"
      caption="Jobs pull order data from Hive into Parquet files on HDFS, with every node and edge property versioned by timestamp so that back-dated graphs can be built. The latest properties at a chosen date are stored in Cypher format; Spark runs Cypher queries to produce one graph per city; the graphs are converted to NetworkX and consumed by TensorFlow on GPUs, which trains the model and writes embeddings to a lookup table read by the ranker at request time. Redrawn from Liu et al. (2019), Uber Engineering."
    >
      <Diagram
        spec={pipeline}
        ariaLabel="Pipeline: Hive to Parquet on HDFS to Cypher format to Spark per-city graphs to NetworkX to TensorFlow to an embedding lookup table used by the ranker"
      />
    </Figure>
  )
}
