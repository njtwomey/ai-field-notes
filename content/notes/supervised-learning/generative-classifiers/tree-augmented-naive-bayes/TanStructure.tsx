import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import type { DiagramEdge, DiagramSpec } from '@/components/diagram/types'
import { Interactive, ParamChoice, Readout, StepControls } from '@/components/viz'

/** The worked example's 16 training rows: class c, then features x1..x4. */
const DATA = [
  [1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 0, 1, 1, 0],
  [1, 0, 0, 0, 0],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [0, 1, 1, 1, 1],
  [0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1],
  [0, 0, 0, 1, 1],
  [0, 1, 1, 1, 0],
]
const D = 4
const N = DATA.length
const TEST = [1, 1, 1, 0]
const id = (i: number) => `x${i + 1}`

type Row = number[]
const count = (pred: (r: Row) => boolean) => DATA.filter(pred).length

/** Empirical conditional mutual information I(x_i; x_j | c) in bits. */
function cmi(i: number, j: number) {
  let s = 0
  for (const c of [0, 1])
    for (const a of [0, 1])
      for (const b of [0, 1]) {
        const nab = count((r) => r[0] === c && r[i + 1] === a && r[j + 1] === b)
        if (nab === 0) continue
        const nc = count((r) => r[0] === c)
        const na = count((r) => r[0] === c && r[i + 1] === a)
        const nb = count((r) => r[0] === c && r[j + 1] === b)
        s += (nab / N) * Math.log2((nab * nc) / (na * nb))
      }
  return s
}

/** Maximum-likelihood p(x_i = v | c, x_parent = pv). */
function theta(i: number, v: number, c: number, parent: number | null, pv: number) {
  const match = (r: Row) => r[0] === c && (parent === null || r[parent + 1] === pv)
  return count((r) => match(r) && r[i + 1] === v) / count(match)
}

/** Training log-likelihood (bits) and p(c = 1 | TEST) for a structure given as each feature's feature parent. */
function evaluate(parents: (number | null)[]) {
  const logJoint = (x: number[], c: number) =>
    Math.log2(count((r) => r[0] === c) / N) +
    parents.reduce<number>((s, p, i) => s + Math.log2(theta(i, x[i], c, p, p === null ? 0 : x[p])), 0)
  const ll = DATA.reduce((s, r) => s + logJoint(r.slice(1), r[0]), 0)
  const [l0, l1] = [0, 1].map((c) => 2 ** logJoint(TEST, c))
  return { ll, posterior: l1 / (l0 + l1) }
}

const EDGES = (() => {
  const out: { i: number; j: number; w: number }[] = []
  for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++) out.push({ i, j, w: cmi(i, j) })
  return out.sort((a, b) => b.w - a.w)
})()

/** Kruskal's algorithm on EDGES: whether each of the first `steps` edges was accepted. */
function kruskal(steps: number) {
  const root = Array.from({ length: D }, (_, k) => k)
  const find = (a: number): number => (root[a] === a ? a : (root[a] = find(root[a])))
  return EDGES.slice(0, steps).map(({ i, j }) => {
    const [a, b] = [find(i), find(j)]
    if (a !== b) root[a] = b
    return a !== b
  })
}

const TREE = EDGES.filter((_, k) => kruskal(EDGES.length)[k])

/** Each feature's parent once the tree is directed away from `root`. */
function directed(root: number) {
  const parents: (number | null)[] = Array(D).fill(null)
  const queue = [root]
  const seen = new Set(queue)
  while (queue.length) {
    const u = queue.shift()!
    for (const { i, j } of TREE) {
      const v = i === u ? j : j === u ? i : -1
      if (v >= 0 && !seen.has(v)) {
        parents[v] = u
        seen.add(v)
        queue.push(v)
      }
    }
  }
  return parents
}

// K4 drawn without crossings: x2 sits inside the triangle x1, x3, x4, so every weight label has its own place.
const COMPLETE: [number, number][] = [
  [0, 3],
  [2.4, 1.9],
  [4.8, 3],
  [2.4, 0],
]

type View = 'kruskal' | 'nb' | 'tan'

/** The complete graph of conditional mutual informations, Kruskal's algorithm on it, and the resulting TAN. */
export function TanStructure() {
  const [view, setView] = useState<View>('kruskal')
  const [steps, setSteps] = useState(0)
  const [root, setRoot] = useState(0)
  const accepted = kruskal(steps)
  const next = EDGES[steps]
  const done = steps >= EDGES.length
  const nb = useMemo(() => evaluate(Array(D).fill(null)), [])
  const tan = useMemo(() => evaluate(directed(root)), [root])

  let spec: DiagramSpec
  if (view === 'kruskal') {
    const ends = next ? [id(next.i), id(next.j)] : []
    spec = {
      unit: 48,
      nodes: COMPLETE.map(([x, y], i) => variable(id(i), x, y, `$x_${i + 1}$`, { highlight: ends.includes(id(i)) })),
      edges: EDGES.map(({ i, j, w }, k) =>
        link(id(i), id(j), false, {
          label: `$${w.toFixed(3)}$`,
          // The vertical x2–x4 edge has its weight low on its right, clear of the x1–x4 and x3–x4 edges.
          ...(i === 1 && j === 3 && { labelSide: 'right' as const, labelPos: 0.3 }),
          highlight: k < steps && accepted[k],
          dashed: k < steps && !accepted[k],
        }),
      ),
    }
  } else {
    const parents = directed(root)
    const edges: DiagramEdge[] = Array.from({ length: D }, (_, i) => link('c', id(i)))
    // Tree edges run along the feature row, bowing below it so that they clear the features in between.
    if (view === 'tan')
      parents.forEach((p, i) => {
        if (p !== null) edges.push(link(id(p), id(i), true, { highlight: true, route: 'curve', bend: 0.35 * (i - p) }))
      })
    spec = {
      unit: 48,
      nodes: [
        variable('c', 2.25, 0, '$c$'),
        ...Array.from({ length: D }, (_, i) =>
          variable(id(i), 1.5 * i, 1.8, `$x_${i + 1}$`, { highlight: view === 'tan' && i === root }),
        ),
      ],
      edges,
    }
  }

  const verdict = (k: number) => {
    const probe = kruskal(k + 1)
    return probe[k] ? 'joins two trees: accept' : 'closes a cycle: reject'
  }
  const treeWeight = EDGES.reduce((s, e, k) => s + (k < steps && accepted[k] ? e.w : 0), 0)

  return (
    <Interactive
      title="Learning the TAN structure"
      caption={
        <MathText
          text={
            view === 'kruskal'
              ? 'Edge weights are the conditional mutual informations $I(x_i; x_j \\mid c)$ in bits, from the worked example. Step considers edges from heaviest to lightest: a coloured edge is accepted into the tree, a dashed edge is rejected because it would close a cycle. The coloured nodes are the ends of the next edge.'
              : 'The class $c$ is a parent of every feature. In the TAN graph the coloured arrows are the maximum spanning tree, directed away from the coloured root; click a feature to make it the root. The training log-likelihood does not change with the root.'
          }
        />
      }
      controls={
        <>
          <ParamChoice
            label="view"
            value={view}
            onChange={setView}
            options={[
              { value: 'kruskal', label: "Kruskal's algorithm" },
              { value: 'nb', label: 'naive Bayes' },
              { value: 'tan', label: 'TAN' },
            ]}
          />
          {view === 'kruskal' && (
            <StepControls
              onStep={() => setSteps((s) => s + 1)}
              onRun={() => setSteps(EDGES.length)}
              onReset={() => setSteps(0)}
              done={done}
            />
          )}
        </>
      }
      readout={
        view === 'kruskal' ? (
          <>
            <Readout label="step" value={`${steps} / ${EDGES.length}`} />
            <Readout
              label="next"
              value={
                done ? (
                  'done'
                ) : (
                  <MathText
                    text={`$x_${next.i + 1}$–$x_${next.j + 1}$ (${next.w.toFixed(3)} bits) ${verdict(steps)}`}
                  />
                )
              }
            />
            <Readout label="tree weight (bits)" value={treeWeight.toFixed(3)} />
          </>
        ) : (
          <>
            <Readout label="training log-likelihood (bits)" value={(view === 'nb' ? nb : tan).ll.toFixed(2)} />
            <Readout
              label={<MathText text="$p(c = 1 \mid \xvec = (1, 1, 1, 0))$" />}
              value={(view === 'nb' ? nb : tan).posterior.toFixed(3)}
            />
          </>
        )
      }
    >
      <Diagram
        spec={spec}
        ariaLabel={
          view === 'kruskal'
            ? 'Complete graph over four features weighted by conditional mutual information'
            : 'Class node with arrows to four features, plus tree edges between features in the TAN view'
        }
        onNodeClick={view === 'tan' ? (n) => n.startsWith('x') && setRoot(Number(n.slice(1)) - 1) : undefined}
      />
    </Interactive>
  )
}
