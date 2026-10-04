import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  exhaustive,
  minimumDistance,
  oneVersusOne,
  oneVersusRest,
  randomCode,
  successProbability,
  type Code,
  type CodeName,
} from './codes'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const P_GRID = toFlat(linspace(0, 0.5, 26))
const P_RANGE: [number, number] = [0, 0.5]
const Y_RANGE: [number | undefined, number | undefined] = [0, 1]
const ENTRY_RANGE: [number, number] = [-1, 1]
const ENTRY_TICKS = [-1, 0, 1]

/** Fixed order, so each code keeps its colour slot. */
const CODES: { value: CodeName; label: string }[] = [
  { value: 'ovr', label: 'one-versus-rest' },
  { value: 'ovo', label: 'one-versus-one' },
  { value: 'exhaustive', label: 'exhaustive' },
  { value: 'random', label: 'random' },
]

const pct = (v: number) => `${(100 * v).toFixed(1)}%`

/**
 * Code explorer for error-correcting output codes: the code matrix, its minimum distance, and the probability of
 * decoding the true class when each binary classifier errs independently with probability p.
 */
export function CodeExplorer() {
  const state = useFigureState({
    k: int(4, { min: 3, max: 7, step: 1, label: 'classes, K' }),
    chosen: choice<CodeName>(CODES, 'exhaustive', { label: 'code shown' }),
    p: float(0.1, { min: 0, max: 0.5, step: 0.01, label: 'bit error rate, p' }),
    length: int(10, { min: 3, max: 40, step: 1, label: 'random code length, L' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const codes = useMemo(
    (): Record<Exclude<CodeName, 'random'>, Code> => ({
      ovr: oneVersusRest(state.k),
      ovo: oneVersusOne(state.k),
      exhaustive: exhaustive(state.k),
    }),
    [state.k],
  )
  const random = useMemo(() => randomCode(state.k, state.length, state.seed), [state.k, state.length, state.seed])
  const all = useMemo((): Record<CodeName, Code> => ({ ...codes, random }), [codes, random])

  const fixedEvaluators = useMemo(
    () => ({
      ovr: successProbability(codes.ovr, 11),
      ovo: successProbability(codes.ovo, 12),
      exhaustive: successProbability(codes.exhaustive, 13),
    }),
    [codes],
  )
  const randomEvaluator = useMemo(() => successProbability(random, state.seed + 1000), [random, state.seed])
  const evaluators = useMemo(
    () => ({ ...fixedEvaluators, random: randomEvaluator }),
    [fixedEvaluators, randomEvaluator],
  )

  const fixedCurves = useMemo(
    () => (['ovr', 'ovo', 'exhaustive'] as const).map((name) => P_GRID.map((q) => fixedEvaluators[name].at(q))),
    [fixedEvaluators],
  )
  const randomCurve = useMemo(() => P_GRID.map((q) => randomEvaluator.at(q)), [randomEvaluator])

  const series = useMemo((): SeriesSpec[] => {
    const curves = [...fixedCurves, randomCurve]
    return CODES.map((c, slot) => ({
      name: `${c.label} (L = ${all[c.value][0].length})`,
      type: 'line',
      x: P_GRID,
      y: curves[slot],
      slot,
    }))
  }, [fixedCurves, randomCurve, all])

  const code = all[state.chosen]
  const heat = useMemo(
    () => ({
      x: code[0].map((_, s) => s + 1),
      y: code.map((_, r) => r + 1),
    }),
    [code],
  )
  const d = useMemo(() => minimumDistance(code), [code])
  const correctable = Math.max(0, Math.floor((d - 1) / 2))
  const success = useMemo(() => evaluators[state.chosen].at(state.p), [evaluators, state.chosen, state.p])

  const xAxis = useAxis({ label: 'column (binary classifier)' })
  const yAxis = useAxis({ label: 'class' })
  const xAxis2 = useAxis({ label: 'bit error rate p', range: P_RANGE })
  const yAxis2 = useAxis({ label: 'P(correct class)', range: Y_RANGE })
  return (
    <Figure
      title="Redundant codes buy robustness to bit errors"
      state={state}
      caption={
        <>
          Each row of the matrix is a class's codeword and each column one binary classifier (red +1, blue −1, grey 0:
          the classifier never sees that class). The chart assumes every classifier is wrong independently with
          probability p and decodes to the nearest codeword, zeros counting 1/2 and ties broken at random; a classifier
          asked about a class it never saw answers with a fair coin. Real classifiers share features and training data,
          so their errors are correlated and the curves are optimistic. Codes of up to 14 columns are enumerated
          exactly; longer ones use 300 simulated inputs per class. Drag the vertical line to move p.
        </>
      }

      readouts={
        <>
          <Readout label="columns L" value={code[0].length} />
          <Readout label="minimum distance d" value={d} />
          <Readout label="guaranteed corrections ⌊(d − 1)/2⌋" value={correctable} />
          <Readout label="rate log₂K / L" value={(Math.log2(state.k) / code[0].length).toFixed(3)} />
          <Readout
            label={`P(correct class) at p = ${state.p.toFixed(2)}${evaluators[state.chosen].exact ? '' : ' (simulated)'}`}
            value={pct(success)}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm font-medium">
            Code matrix: {CODES.find((c) => c.value === state.chosen)?.label}
          </p>
          <Plot x={xAxis} y={yAxis} height={300}>
            <Raster
              x={heat.x}
              y={heat.y}
              z={code}
              scale={'diverging'}
              range={ENTRY_RANGE}
              scaleTicks={ENTRY_TICKS}
              valueLabel={'entry'}
            />
          </Plot>
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Probability of decoding the true class</p>
          <Plot x={xAxis2} y={yAxis2} height={300}>
            {seriesLayers(series)}
            <Handle {...state.handle('p', { label: 'bit error rate p' })} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
