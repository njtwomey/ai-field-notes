import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  useParam,
  XYChart,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
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

const P_GRID = linspace(0, 0.5, 26)
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
  const [k, setK] = useState(4)
  const [chosen, setChosen] = useState<CodeName>('exhaustive')
  const [length, setLength] = useState(10)
  const [seed, setSeed] = useState(1)
  const p = useParam(0.1, { min: 0, max: 0.5, step: 0.01 })

  const codes = useMemo(
    (): Record<Exclude<CodeName, 'random'>, Code> => ({
      ovr: oneVersusRest(k),
      ovo: oneVersusOne(k),
      exhaustive: exhaustive(k),
    }),
    [k],
  )
  const random = useMemo(() => randomCode(k, length, seed), [k, length, seed])
  const all = useMemo((): Record<CodeName, Code> => ({ ...codes, random }), [codes, random])

  const fixedEvaluators = useMemo(
    () => ({
      ovr: successProbability(codes.ovr, 11),
      ovo: successProbability(codes.ovo, 12),
      exhaustive: successProbability(codes.exhaustive, 13),
    }),
    [codes],
  )
  const randomEvaluator = useMemo(() => successProbability(random, seed + 1000), [random, seed])
  const evaluators = useMemo(
    () => ({ ...fixedEvaluators, random: randomEvaluator }),
    [fixedEvaluators, randomEvaluator],
  )

  const fixedCurves = useMemo(
    () => (['ovr', 'ovo', 'exhaustive'] as const).map((name) => P_GRID.map((q) => fixedEvaluators[name].at(q))),
    [fixedEvaluators],
  )
  const randomCurve = useMemo(() => P_GRID.map((q) => randomEvaluator.at(q)), [randomEvaluator])

  const series = useMemo((): XYSeries[] => {
    const curves = [...fixedCurves, randomCurve]
    return CODES.map((c, slot) => ({
      name: `${c.label} (L = ${all[c.value][0].length})`,
      type: 'line',
      x: P_GRID,
      y: curves[slot],
      slot,
    }))
  }, [fixedCurves, randomCurve, all])

  const code = all[chosen]
  const heat = useMemo(
    () => ({
      x: code[0].map((_, s) => s + 1),
      y: code.map((_, r) => r + 1),
    }),
    [code],
  )
  const d = useMemo(() => minimumDistance(code), [code])
  const correctable = Math.max(0, Math.floor((d - 1) / 2))
  const success = useMemo(() => evaluators[chosen].at(p.value), [evaluators, chosen, p.value])

  const handles: Handle[] = [{ kind: 'x', at: p.value, label: 'bit error rate p', onDrag: p.set }]

  return (
    <Interactive
      title="Redundant codes buy robustness to bit errors"
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
      controls={
        <>
          <ParamSlider label="classes, K" value={k} onChange={setK} min={3} max={7} step={1} withArrows />
          <ParamChoice label="code shown" value={chosen} onChange={setChosen} options={CODES} />
          <ParamSlider param={p} label="bit error rate, p" />
          <ParamSlider
            label="random code length, L"
            value={length}
            onChange={setLength}
            min={3}
            max={40}
            step={1}
            withArrows
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New random code</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="columns L" value={code[0].length} />
          <Readout label="minimum distance d" value={d} />
          <Readout label="guaranteed corrections ⌊(d − 1)/2⌋" value={correctable} />
          <Readout label="rate log₂K / L" value={(Math.log2(k) / code[0].length).toFixed(3)} />
          <Readout
            label={`P(correct class) at p = ${p.value.toFixed(2)}${evaluators[chosen].exact ? '' : ' (simulated)'}`}
            value={pct(success)}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm font-medium">
            Code matrix: {CODES.find((c) => c.value === chosen)?.label}
          </p>
          <Heatmap
            x={heat.x}
            y={heat.y}
            z={code}
            xLabel="column (binary classifier)"
            yLabel="class"
            valueLabel="entry"
            scale="diverging"
            range={ENTRY_RANGE}
            scaleTicks={ENTRY_TICKS}
            height={300}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Probability of decoding the true class</p>
          <XYChart
            series={series}
            xLabel="bit error rate p"
            yLabel="P(correct class)"
            xRange={P_RANGE}
            yRange={Y_RANGE}
            handles={handles}
            height={300}
          />
        </div>
      </div>
    </Interactive>
  )
}
