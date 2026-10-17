import { useMemo, useState } from 'react'
import {
  cn,
  Diagram,
  Figure,
  int,
  Readout,
  slider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
  type DiagramEdge,
  type DiagramNode,
  type DiagramSpec,
} from 'aifn-render'
import { mineClosedRules, predictLinks } from 'aifn-compute/logic/induction'
import { PEOPLE_AND_PLACES } from './people-and-places'

const POSITION: Record<string, { x: number; y: number }> = {
  ann: { x: 0, y: 0 },
  ben: { x: 0, y: 1.8 },
  gus: { x: 2.2, y: 0.9 },
  cal: { x: 0, y: 3.8 },
  dia: { x: 0, y: 5.6 },
  hal: { x: 2.2, y: 4.7 },
  eli: { x: 0, y: 7.6 },
  fay: { x: 0, y: 9.4 },
  paris: { x: 5.2, y: 1.2 },
  lyon: { x: 5.2, y: 8.2 },
  rome: { x: 5.2, y: 4.2 },
  milan: { x: 5.2, y: 6.2 },
  france: { x: 8.4, y: 3.2 },
  italy: { x: 8.4, y: 5.6 },
}
const PLACES = new Set(['paris', 'lyon', 'rome', 'milan', 'france', 'italy'])

const schema = {
  minSupport: int(2, { label: 'least support', min: 1, max: 6, suggestions: [1, 2, 3, 4] }),
  minPca: slider(0, 1, 0.1, { label: 'least PCA confidence', step: 0.05 }),
}

const pct = (x: number) => x.toFixed(2)

/** Closed rules mined from a small knowledge graph, their scores, and the facts they predict. */
export function RuleMiningExplorer() {
  const state = useFigureState(schema)
  const rules = useMemo(
    () => mineClosedRules(PEOPLE_AND_PLACES, { minSupport: state.minSupport, minPcaConfidence: state.minPca }),
    [state.minSupport, state.minPca],
  )
  const predictions = useMemo(() => predictLinks(rules), [rules])
  const [picked, setPicked] = useState<string | null>(null)
  const rule = rules.find((r) => r.text === picked) ?? rules[0] ?? null

  const spec: DiagramSpec = useMemo(() => {
    const nodes: DiagramNode[] = Object.entries(POSITION).map(([id, p]) => ({
      id,
      ...p,
      shape: PLACES.has(id) ? ('box' as const) : ('pill' as const),
      w: 1.5,
      h: 0.7,
      label: id,
      filled: PLACES.has(id),
    }))
    const support = new Set(rule ? rule.supporting.map(([x, y]) => `${x} ${rule.head} ${y}`) : [])
    const edges: DiagramEdge[] = PEOPLE_AND_PLACES.map(([s, r, o]) => {
      const on = support.has(`${s} ${r} ${o}`)
      const reverse = PEOPLE_AND_PLACES.some(([a, b, c]) => a === o && b === r && c === s)
      return {
        from: s,
        to: o,
        route: reverse ? ('curve' as const) : ('straight' as const),
        bend: reverse ? 0.4 : undefined,
        arrow: 'end' as const,
        tone: on ? 2 : ('neutral' as const),
        highlight: on,
        label: on ? r : undefined,
        state: on || !rule ? undefined : ('idle' as const),
      }
    })
    if (rule)
      for (const [x, y] of rule.predictions)
        edges.push({
          from: x,
          to: y,
          route: 'curve',
          bend: -0.6,
          arrow: 'end',
          tone: 1,
          dashed: true,
          highlight: true,
          label: `${rule.head}?`,
        })
    return { unit: 40, nodes, edges }
  }, [rule])

  return (
    <Figure
      title="Mining rules from an incomplete knowledge graph"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      readouts={
        rule
          ? {
              'selected rule': (
                <>
                  <Readout label="rule" value={rule.text} />
                  <Readout label="support" value={`${rule.support} of ${rule.headSize} ${rule.head} facts`} />
                  <Readout
                    label="standard confidence"
                    value={`${rule.support} / ${rule.bodySize} = ${pct(rule.confidence)}`}
                  />
                  <Readout
                    label="PCA confidence"
                    value={`${rule.support} / ${rule.pcaBodySize} = ${pct(rule.pcaConfidence)}`}
                  />
                </>
              ),
            }
          : undefined
      }
      caption="The graph has 14 entities and 26 facts; people are rounded, places are shaded boxes. Fay has no home or citizenship, Gus no citizenship, and only Eli's side of their marriage is recorded. Set the least support and the least PCA confidence; every closed rule of up to two body atoms that passes is listed, best first. Click a rule: its supporting facts (body true and head a fact) are drawn in the third colour with their relation, and the facts it predicts (body true, head missing) as dashed curves in the second colour. Standard confidence counts every prediction as wrong; PCA confidence ignores predictions about a subject with no fact of the head relation at all, which are unknown. The predictions table scores each missing fact by its best rule."
    >
      <div className="grid h-full gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className="min-h-80">
          <Diagram
            spec={spec}
            ariaLabel="Knowledge graph with the facts supporting the selected rule and the facts it predicts"
          />
        </div>
        <div className="grid content-start gap-3 overflow-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>rule</TableHead>
                <TableHead className="text-right">support</TableHead>
                <TableHead className="text-right">head cov.</TableHead>
                <TableHead className="text-right">std conf.</TableHead>
                <TableHead className="text-right">PCA conf.</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rules.map((r) => (
                <TableRow
                  key={r.text}
                  className={cn('cursor-pointer', r === rule && 'bg-muted font-semibold')}
                  onClick={() => setPicked(r.text)}
                  aria-selected={r === rule}
                >
                  <TableCell className="font-mono">{r.text}</TableCell>
                  <TableCell className="text-right">{r.support}</TableCell>
                  <TableCell className="text-right">{pct(r.headCoverage)}</TableCell>
                  <TableCell className="text-right">{pct(r.confidence)}</TableCell>
                  <TableCell className="text-right">{pct(r.pcaConfidence)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>predicted fact</TableHead>
                <TableHead className="text-right">score</TableHead>
                <TableHead className="text-right">rules</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predictions.slice(0, 10).map((p) => (
                <TableRow key={`${p.subject} ${p.relation} ${p.object}`}>
                  <TableCell className="font-mono">{`${p.relation}(${p.subject}, ${p.object})`}</TableCell>
                  <TableCell className="text-right">{pct(p.score)}</TableCell>
                  <TableCell className="text-right">{p.rules.length}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {predictions.length > 10 && (
            <p className="text-xs text-muted-foreground">
              {predictions.length - 10} lower-scored predictions not shown.
            </p>
          )}
        </div>
      </div>
    </Figure>
  )
}
