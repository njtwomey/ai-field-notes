import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  choice,
  formatNumber,
  int,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { CONTESTED, UNASSIGNED, decide, expectedAccuracy, fit, makeData, type Method, type Vec2 } from './reductions'

const LO = -4
const HI = 4
const GRID = toFlat(linspace(LO, HI, 61))
const PER_CLASS = 40
const CLASS_NAMES = ['class 1', 'class 2', 'class 3', 'class 4']
/** Class 2 sits between classes 1 and 3, so no line separates it from the rest. */
const START: Vec2[] = [
  [-2.3, -1],
  [0, -0.8],
  [2.3, -1],
  [0, 1.9],
]

const METHODS: { value: Method; label: string }[] = [
  { value: 'ovr-argmax', label: 'OvR argmax' },
  { value: 'ovr-claimed', label: 'OvR claims' },
  { value: 'ovo', label: 'OvO vote' },
  { value: 'ecoc', label: 'ECOC Hamming' },
  { value: 'ecoc-loss', label: 'ECOC loss' },
  { value: 'multinomial', label: 'multinomial' },
]

const clamp = (v: number) => Math.min(Math.max(v, LO + 0.3), HI - 0.3)

/** The part of the line w0 + w1 x1 + w2 x2 = 0 inside the plot, as two end points (none if it misses the plot). */
function boundary([w0, w1, w2]: [number, number, number]): { x: number[]; y: number[] } {
  if (Math.abs(w2) >= Math.abs(w1)) {
    if (w2 === 0) return { x: [], y: [] }
    return { x: [LO, HI], y: [LO, HI].map((a) => -(w0 + w1 * a) / w2) }
  }
  return { x: [LO, HI].map((b) => -(w0 + w2 * b) / w1), y: [LO, HI] }
}

/**
 * Four classes on the plane and five ways to classify them with linear logistic classifiers: one-versus-rest (argmax
 * of scores, or only where exactly one classifier claims the point), one-versus-one majority vote, an exhaustive
 * error-correcting output code with Hamming decoding, and multinomial logistic regression.
 */
export function MulticlassReductions({ initial = 'ovr-argmax' }: { initial?: Method }) {
  const [centres, setCentres] = useState<Vec2[]>(START)
  const state = useFigureState({
    method: choice<Method>(METHODS, initial, { label: 'method' }),
    spread: slider(0.3, 1.6, 0.8, { step: 0.05, label: 'class spread (standard deviation)' }),
    seed: int(3, { min: 1, max: 20, label: 'data seed' }),
  })
  const { method, spread, seed } = state

  const data = useMemo(() => makeData(seed, centres, PER_CLASS, spread), [seed, centres, spread])
  const fitted = useMemo(() => fit(data, method), [data, method])

  const { z, unassigned, contested } = useMemo(() => {
    const z = GRID.map((b) => GRID.map((a) => decide(fitted, [a, b], data.k).value))
    const flat = z.flat()
    const share = (v: number) => flat.filter((c) => c === v).length / flat.length
    return { z, unassigned: share(UNASSIGNED), contested: share(CONTESTED) }
  }, [fitted, data.k])
  const accuracy = useMemo(() => expectedAccuracy(fitted, data), [fitted, data])

  const points = useMemo(() => ({ x: data.x.map((p) => p[0]), y: data.x.map((p) => p[1]) }), [data])
  // One-versus-rest draws each binary boundary in its class's colour: the claimed side of line k is class k's.
  const lines = useMemo(
    () => (method === 'ovr-argmax' || method === 'ovr-claimed' ? fitted.weights.map((w) => boundary(w)) : []),
    [fitted, method],
  )

  const xAxis = useAxis({ label: 'x₁', range: [LO, HI] })
  const yAxis = useAxis({ label: 'x₂', range: [LO, HI] })
  const pct = (v: number) => `${formatNumber(100 * v)}%`
  return (
    <Figure
      title="Four classes, five multiclass rules"
      state={state}
      caption={
        <>
          Four Gaussian classes of 40 points each; every binary classifier is a ridge-penalised linear logistic
          regression. Cells show the predicted class. Light grey cells get no single class: no one-versus-rest
          classifier claims them, or the one-versus-one votes or the Hamming distances to the codewords tie. Dark grey
          cells are claimed by several one-versus-rest classifiers. The output code is the exhaustive code for four
          classes (7 columns, any two codewords 4 bits apart), decoded by Hamming distance or by the summed logistic
          loss. The coloured lines are the one-versus-rest boundaries. Class 2 lies between classes 1 and 3, so no line
          separates class 2 from the rest. Training accuracy breaks ties at random. Drag a class mean to move its
          cluster.
        </>
      }
      readouts={
        <>
          <Readout label="training accuracy" value={pct(accuracy)} />
          <Readout label="plane unassigned" value={pct(unassigned)} />
          <Readout label="plane contested" value={pct(contested)} />
          <Readout
            label="binary problems × examples"
            value={fitted.problems === 0 ? 'one joint fit' : `${fitted.problems} × ${fitted.problemSize}`}
          />
        </>
      }
    >
      <Plot
        x={xAxis}
        y={yAxis}
        height={440}
        ariaLabel="Predicted class over the plane for the chosen multiclass rule, with the training points"
      >
        <Raster
          x={GRID}
          y={GRID}
          z={z}
          scale="categorical"
          categoryNames={CLASS_NAMES}
          fillOpacity={0.45}
          valueLabel="predicted"
        />
        <Points name="training point" x={points.x} y={points.y} group={data.y} groupNames={CLASS_NAMES} live />
        {lines.map((l, k) => (
          <Curve key={k} name={`${k + 1} vs rest, score 0`} x={l.x} y={l.y} slot={k} live />
        ))}
        {centres.map((c, k) => (
          <Handle
            key={k}
            kind="point"
            at={c}
            label={`mean ${k + 1}`}
            onDrag={([a, b]) => setCentres((cs) => cs.map((old, j): Vec2 => (j === k ? [clamp(a), clamp(b)] : old)))}
          />
        ))}
      </Plot>
    </Figure>
  )
}
