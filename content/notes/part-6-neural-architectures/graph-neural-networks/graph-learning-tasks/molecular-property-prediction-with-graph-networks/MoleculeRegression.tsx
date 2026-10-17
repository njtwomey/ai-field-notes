import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { FixedHeight, drawingRange, useSlicedRun } from '../_shared/figure-helpers'
import {
  fingerprintModel,
  gnnRun,
  MOLECULES,
  splitOf,
  type Checkpoint,
  type Setup,
  type SplitKind,
} from './molecule-run'

const PANEL = 250
const SPLITS: { value: SplitKind; label: string }[] = [
  { value: 'scaffold', label: 'scaffold: test on five-rings' },
  { value: 'random', label: 'random, same test size' },
]
/** Contributions beyond ±LIMIT take the ends of the colour scale. */
const LIMIT = 1

/** Message passing against fingerprints on synthetic molecules with an additive property. */
export function MoleculeRegression() {
  const state = useFigureState({
    split: choice(SPLITS, 'scaffold', { label: 'split' }),
    radius: int(2, { ge: 0, le: 3, label: 'fingerprint radius' }),
    steps: int(150, { ge: 10, le: 3000, suggestions: [50, 150, 1000], label: 'Adam steps' }),
    rate: float(0.02, { gt: 0, le: 0.5, suggestions: [0.005, 0.02, 0.05], label: 'learning rate' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    molecule: int(1, { ge: 1, le: 50, label: 'test molecule' }),
  })
  const { split, radius, steps, rate, seed } = state
  const setup = useMemo<Setup>(() => ({ split, steps, rate, seed }), [split, steps, rate, seed])
  const run = useSlicedRun(setup, gnnRun)
  const last: Checkpoint | undefined = run.checkpoints.at(-1)
  const fp = useMemo(() => fingerprintModel(split, radius), [split, radius])
  const test = useMemo(() => splitOf(split).test, [split])
  const truth = test.map((i) => MOLECULES[i].property)
  const pick = Math.min(state.molecule, test.length) - 1
  const mol = MOLECULES[test[pick]]

  const colour = useScaleColor('diverging')
  const tone = (v: number) => colour(0.5 + Math.max(-1, Math.min(1, v / LIMIT)) / 2)
  const drawing = { pos: mol.pos, edges: mol.bonds.map(([a, b]) => [a, b] as const) }
  const range = drawingRange([drawing], 0.7)
  const singles = mol.bonds.filter((b) => b[2] === 1).map(([a, b]) => ({ from: mol.pos[a], to: mol.pos[b] }))
  const doubles = mol.bonds.filter((b) => b[2] === 2).map(([a, b]) => ({ from: mol.pos[a], to: mol.pos[b] }))
  const atomLabel = (v: number, value: number) =>
    `${mol.atoms[v]}${mol.hydrogens[v] ? `H${mol.hydrogens[v] > 1 ? mol.hydrogens[v] : ''}` : ''} ${formatNumber(value)}`

  const px = useAxis({ label: 'true property', range: [-1.5, 4.5], nice: false })
  const py = useAxis({ label: 'predicted', range: [-1.5, 4.5], nice: false, equal: px })
  const sx = useAxis({ label: 'Adam step', range: [0, steps], nice: false })
  const ey = useAxis({ label: 'test RMSE', range: [0.03, 3], log: true, nice: false })
  const mx = useAxis({ range: range.x, nice: false, zoom: false })
  const my = useAxis({ range: range.y, nice: false, zoom: false, equal: mx })
  const mx2 = useAxis({ range: range.x, nice: false, zoom: false })
  const my2 = useAxis({ range: range.y, nice: false, zoom: false, equal: mx2 })

  const molecule = (values: readonly number[], title: string, x: typeof mx, y: typeof my) => (
    <Plot x={x} y={y} bare title={title} legend={false}>
      <Segments id="single" name="single bond" segments={singles} width={1.5} />
      {doubles.length > 0 && <Segments id="double" name="double bond" segments={doubles} width={4} />}
      <Points
        name="atoms"
        x={mol.pos.map((p) => p[0])}
        y={mol.pos.map((p) => p[1])}
        colors={values.map(tone)}
        labels={values.map((v, i) => atomLabel(i, v))}
        size={16}
      />
    </Plot>
  )
  const gnnAtoms = last?.atomsTest[pick]

  return (
    <Figure
      title="A message passing network against fingerprints"
      state={state}
      defaultSize="XL"
      caption={`${MOLECULES.length} synthetic molecules: a six-ring, five-ring or four-carbon chain of carbons with one to four groups (methyl, ethyl, hydroxyl, methoxy, amine, carbonyl). The property is a sum of per-atom contributions fixed by element, hydrogen count and double bond, plus 0.5 for every pair of N or O atoms three bonds apart, plus noise of standard deviation 0.05. The scaffold split tests on the five-ring molecules, a scaffold never seen in training. The message passing network has three layers of width 12 with one weight matrix per bond order, a linear output per atom and a sum readout, trained by full-batch Adam. The fingerprint model is ridge regression on unhashed Morgan-style counts of every atom environment up to the chosen radius. Top left: predicted against true property on the test molecules. Top right: the network's test error during training against the fingerprint model's (dashed). Bottom: one test molecule, coloured by each atom's true contribution (left) and by the network's per-atom output (right), which sum to the true and the predicted property. Bond lines are thick for double bonds.`}
      readouts={
        <>
          <Readout label="GNN test RMSE" value={last ? formatNumber(last.testRmse) : '…'} />
          <Readout label={`fingerprint (radius ${radius}) test RMSE`} value={formatNumber(fp.testRmse)} />
          <Readout label="fingerprint identifiers" value={String(fp.vocabulary)} />
          <Readout
            label={`molecule ${pick + 1}: true, GNN, fingerprint`}
            value={`${formatNumber(mol.property)}, ${last ? formatNumber(last.predictTest[pick]) : '…'}, ${formatNumber(fp.predictTest[pick])}`}
          />
          <Readout label="step" value={last ? `${last.step}${run.done ? '' : ', training…'}` : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={px} y={py} title="test molecules">
            <Curve name="perfect prediction" x={[-1.5, 4.5]} y={[-1.5, 4.5]} dashed muted />
            <Points name={`fingerprint, radius ${radius}`} x={truth} y={fp.predictTest} slot={1} thin />
            {last && <Points name="message passing network" x={truth} y={last.predictTest} slot={0} thin />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sx} y={ey} title="error on the test molecules">
            <Curve
              name="message passing network"
              slot={0}
              x={run.checkpoints.map((c) => c.step)}
              y={run.checkpoints.map((c) => Math.min(3, Math.max(0.03, c.testRmse)))}
            />
            <Curve
              name={`fingerprint, radius ${radius}`}
              slot={1}
              x={[0, steps]}
              y={[fp.testRmse, fp.testRmse]}
              dashed
            />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>{molecule(mol.contributions, 'true contribution per atom', mx, my)}</FixedHeight>
        <FixedHeight height={PANEL}>
          {gnnAtoms ? (
            molecule(gnnAtoms, 'network output per atom', mx2, my2)
          ) : (
            <div className="p-4 text-sm text-muted-foreground">Training…</div>
          )}
        </FixedHeight>
      </div>
    </Figure>
  )
}
