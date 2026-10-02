/**
 * Showcase: from noisy labelling functions to labels. `labellingFunctions` draws votes from functions with known
 * accuracies and coverages; aifn-applied `labelModelReport` runs majority vote, Dawid–Skene EM (every step) and the
 * data-programming label model on them in the worker. The page compares each method's estimate of every function's
 * accuracy with the truth, and the accuracy of the labels each method assigns.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { labellingFunctions } from 'aifn-applied/data/synthetic'
import type { LabelModelReport } from 'aifn-applied/learning/weak-supervision'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, float, int, row, useComputed, useFigureState } from '@lab/state'
import { formatValue } from '@lab/views'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const SHOWN = 60
const EM_STEPS = 30
const CLASS_NAMES = ['class 1', 'class 2', 'class 3', 'class 4']

export function LabelModelShowcase() {
  const state = useFigureState({
    data: row('1 · examples', {
      n: int(600, { ge: 20, le: 5000, suggestions: [200, 600, 2000], label: 'examples' }),
      classes: int(2, { ge: 2, le: 4, label: 'classes' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    functions: row('2 · labelling functions', {
      functions: int(10, { ge: 2, le: 30, suggestions: [4, 10, 20], label: 'functions' }),
      low: float(0.55, { ge: 0, le: 1, suggestions: [0.4, 0.55, 0.7], label: 'accuracy from' }),
      high: float(0.95, { ge: 0, le: 1, suggestions: [0.8, 0.95, 0.99], label: 'accuracy to' }),
      copies: int(0, { ge: 0, le: 10, suggestions: [0, 2, 4], label: 'copies of function 1' }),
    }),
  })
  const { n, classes, seed } = state.data
  const { functions, low, high, copies } = state.functions
  const data = useMemo(
    () =>
      labellingFunctions(stream(seed), {
        n,
        classes,
        functions,
        copies,
        accuracy: [Math.min(low, high), Math.max(low, high)],
      }),
    [n, classes, seed, functions, copies, low, high],
  )
  const truth = useMemo(() => Array.from(toFlat(data.y)), [data])
  const m = data.votes.shape[1]
  const key = JSON.stringify([n, classes, seed, functions, copies, low, high])
  const report = useComputed(
    () =>
      call<LabelModelReport>('applied/learning/weak-supervision/labelModelReport', data.votes, classes, truth, {
        steps: EM_STEPS,
      }),
    [key],
    {
      mode: 'worker',
      initial: null as { key: string; r: LabelModelReport } | null,
      then: (r) => ({ key, r }),
      cancelAfter: 2000,
    },
  )
  const r = report.value?.key === key ? report.value.r : null
  const [picked, setPicked] = useState<{ key: string; step: number } | null>(null)
  const step = r ? Math.min(picked?.key === key ? picked.step : 0, r.frames.length - 1) : 0
  const pick = (s: number) => setPicked({ key, step: Math.max(0, Math.round(s)) })
  const frame = r?.frames[step]

  // The first examples' votes as a raster (example × function), abstentions empty.
  const votes = useMemo(() => {
    const v = toFlat(data.votes)
    return Array.from({ length: Math.min(SHOWN, n) }, (_, i) => Array.from({ length: m }, (_, j) => v[i * m + j]))
  }, [data, n, m])
  const argmax = (p: Float64Array) =>
    Array.from({ length: n }, (_, i) => {
      let b = 0
      for (let k = 1; k < classes; k++) if (p[i * classes + k] > p[i * classes + b]) b = k
      return b
    })
  const lmLabels = useMemo(() => (r ? argmax(r.labelModel) : null), [r]) // oxlint-disable-line react-hooks/exhaustive-deps
  const xy = useMemo(
    () => ({
      x: Array.from({ length: n }, (_, i) => toFlat(data.x)[2 * i]),
      y: Array.from({ length: n }, (_, i) => toFlat(data.x)[2 * i + 1]),
    }),
    [data, n],
  )
  const trueAcc = r ? r.empiricalAccuracy : []

  const ax = useAxis({ label: 'true accuracy of the function', range: [0, 1.02] })
  const ay = useAxis({ label: 'estimated accuracy', range: [0, 1.02], equal: ax })
  const fx = useAxis({ label: 'labelling function', range: [-0.5, m - 0.5], key: m, integer: true })
  const ey = useAxis({ label: 'example', range: [-0.5, Math.min(SHOWN, n) - 0.5], key: n, integer: true })
  const sx = useAxis({ label: 'Dawid–Skene EM step', range: [0, EM_STEPS], integer: true })
  const sy = useAxis({ label: 'accuracy of the labels', hold: 'union', key })
  const px = useAxis({ label: 'x₁', hold: 'union', key })
  const py = useAxis({ label: 'x₂', hold: 'union', key, equal: px })
  const pending = r ? undefined : 'fitting…'
  return (
    <Figure
      title="Weak supervision: from noisy labelling functions to labels"
      purpose="No example has a label, yet the agreements between labelling functions reveal how accurate each one is; a label model that weighs each function by its estimated accuracy labels the data better than a majority vote."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="EM steps">
          <Player
            value={step}
            onChange={pick}
            count={r ? r.frames.length : 1}
            label="step"
            format={(i) => `step ${i}`}
          />
        </ControlRow>
      }
      readouts={{
        labels: (
          <>
            <Readout label="majority vote" value={f3(r?.accuracy.majority)} />
            <Readout label="Dawid–Skene (final)" value={f3(r?.accuracy.dawidSkene)} />
            <Readout label="label model" value={f3(r?.accuracy.labelModel)} />
            <Readout label={`Dawid–Skene at step ${step}`} value={f3(frame?.labelAccuracy)} />
            <Readout label="log-likelihood" value={f3(frame?.logLikelihood)} />
            {!r && <Readout label="computing" value={<span aria-busy="true">in a worker…</span>} />}
          </>
        ),
      }}
      caption={
        <>
          aifn-applied <code>labellingFunctions</code>: {n} examples in {classes} Gaussian classes and {functions}{' '}
          functions{copies ? ` plus ${copies} copies of function 1` : ''}, each voting with a coverage in [0.2, 0.8]
          and, when it votes, right with an accuracy drawn from [{Math.min(low, high)}, {Math.max(low, high)}];{' '}
          <code>labelModelReport</code> fits the models to the votes alone. Top left: each function&apos;s accuracy
          estimated by the label model (fixed) and by Dawid–Skene at the EM step on the player, against its true
          accuracy on the data; the diagonal is a perfect estimate. Top right: the votes of the first {SHOWN} examples
          (colour: the class voted, empty: abstained). Bottom left: the accuracy of each method&apos;s labels against
          the hidden truth, Dawid–Skene step by step; drag the step marker or play from step 0. Bottom right: the
          examples coloured by the label model&apos;s label and shaped by their true class.
          {copies
            ? ' Copies break the label models’ assumption that functions err independently: they count one function several times.'
            : ''}
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[1, 1]}>
        <Plot x={ax} y={ay} title={pending ?? 'estimated against true accuracy'}>
          <Curve name="perfect estimate" x={[0, 1]} y={[0, 1]} emphasis dashed width={1} />
          {r && <Points name="label model" x={trueAcc} y={r.labelModelAccuracy} slot={2} size={9} />}
          {frame && <Points name={`Dawid–Skene, step ${step}`} x={trueAcc} y={frame.accuracy} slot={3} size={9} />}
        </Plot>
        <Plot x={fx} y={ey} title={`votes of the first ${Math.min(SHOWN, n)} examples`}>
          <Raster
            x={Array.from({ length: m }, (_, j) => j)}
            y={Array.from({ length: Math.min(SHOWN, n) }, (_, i) => i)}
            z={votes}
            scale="categorical"
            categoryNames={CLASS_NAMES.slice(0, classes)}
          />
        </Plot>
        <Plot x={sx} y={sy} title={pending ?? 'how good are the labels'}>
          {r && (
            <Curve
              name="Dawid–Skene"
              x={r.frames.map((f) => f.step)}
              y={r.frames.map((f) => f.labelAccuracy)}
              slot={3}
            />
          )}
          {r && (
            <Curve
              name="majority vote"
              x={[0, EM_STEPS]}
              y={[r.accuracy.majority, r.accuracy.majority]}
              slot={0}
              dashed
            />
          )}
          {r && (
            <Curve
              name="label model"
              x={[0, EM_STEPS]}
              y={[r.accuracy.labelModel, r.accuracy.labelModel]}
              slot={2}
              dashed
            />
          )}
          {r && <Handle kind="x" at={step} onDrag={pick} label={`step ${step}`} />}
        </Plot>
        <Plot x={px} y={py} title={pending ?? 'the label model’s labels'}>
          {lmLabels && (
            <Points
              name="examples"
              x={xy.x}
              y={xy.y}
              group={lmLabels}
              groupNames={CLASS_NAMES.slice(0, classes).map((c) => `labelled ${c}`)}
              shape={truth}
              shapeNames={CLASS_NAMES.slice(0, classes).map((c) => `truly ${c}`)}
              size={6}
            />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
