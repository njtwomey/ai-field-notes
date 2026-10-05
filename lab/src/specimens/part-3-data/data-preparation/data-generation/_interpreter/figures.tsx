import { useMemo, useState } from 'react'
import { fromData, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { members, signature, type Prelude } from 'aifn-compute/interpreter'
import { dataset } from 'aifn-compute/learning/estimators'
import { prelude } from 'aifn-methods/interpreter'
import { linearRegression } from 'aifn-methods/learning/linear'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import {
  Button,
  CodeEditor,
  EntryControls,
  NumberField,
  ProgramStatus,
  Select,
  Switch,
  useEntryArgs,
  useProgram,
} from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { call } from 'aifn-render/state'
import { Curve, Plot, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const REGRESSION = `seed(7)

/**
 * @param {int} n [20, 1000] number of points
 * @param {real} noise [0, 2] noise standard deviation
 */
function make(n = 200, noise = 0.3) {
  const x = array.linspace(0, 6, n)
  const e = random.normal(n)
  const y = x.map((v, i) => 1.5 * v + 2 * math.sin(v) + noise * e[i])
  return [x, y]
}
`

const CLASSIFICATION = `seed(3)

/**
 * Two Gaussian clouds in the plane, labelled 0 and 1.
 * @param {int} n [20, 1000] number of points
 * @param {real} gap [0, 5] distance between the class means along x₁
 */
function make(n = 160, gap = 2.5) {
  const labels = random.bernoulli(0.5, n)
  const x1 = random.normal(n).map((z, i) => z + gap * labels[i])
  const x2 = random.normal(n).map((z, i) => z + 0.6 * gap * labels[i])
  const X = array.stack([x1, x2], 1)
  return [X, labels]
}
`

const EXAMPLES: Record<'regression' | 'classification', { label: string; code: string }> = {
  regression: { label: 'regression: return [x, y]', code: REGRESSION },
  classification: { label: 'two classes: return [X, labels]', code: CLASSIFICATION },
}
type ExampleKey = keyof typeof EXAMPLES

type Fit =
  | {
      kind: 'regression'
      x: number[]
      y: number[]
      line: { x: number[]; y: number[] }
      slope: number
      intercept: number
      r2: number
      sd: number
    }
  | {
      kind: 'classification'
      x1: number[]
      x2: number[]
      labels: number[]
      boundary: { x: number[]; y: number[] }
      weights: number[]
      intercept: number
      accuracy: number
    }

const isNumbers = (v: unknown): v is number[] => Array.isArray(v) && v.every((x) => typeof x === 'number')
const isRows = (v: unknown): v is number[][] => Array.isArray(v) && v.every((r) => isNumbers(r) && r.length === 2)

/** The program's result checked against the example's contract, and the model fitted to it. */
function fitResult(kind: ExampleKey, value: unknown): Fit {
  if (!Array.isArray(value) || value.length !== 2)
    throw new Error(`make() should return a pair: ${kind === 'regression' ? '[x, y]' : '[X, labels]'}`)
  const [a, b] = value as unknown[]
  if (kind === 'regression') {
    if (!isNumbers(a) || !isNumbers(b) || a.length !== b.length || a.length < 3)
      throw new Error('[x, y] should be two arrays of numbers of the same length (at least 3)')
    const model = linearRegression().fit(dataset(fromData(Float64Array.from(a), [a.length, 1]), tensor(b)))
    const slope = toFlat(model.weights)[0]
    const lo = Math.min(...a)
    const hi = Math.max(...a)
    const ybar = b.reduce((s, v) => s + v, 0) / b.length
    const tss = b.reduce((s, v) => s + (v - ybar) ** 2, 0)
    return {
      kind,
      x: a,
      y: b,
      line: { x: [lo, hi], y: [model.intercept + slope * lo, model.intercept + slope * hi] },
      slope,
      intercept: model.intercept,
      r2: 1 - model.rss / tss,
      sd: model.noiseSd,
    }
  }
  if (!isRows(a) || !isNumbers(b) || a.length !== b.length || !b.every((v) => v === 0 || v === 1))
    throw new Error('[X, labels] should be rows [x1, x2] and labels 0 or 1, one per row')
  if (new Set(b).size < 2) throw new Error('labels should contain both classes')
  const model = logisticRegression().fit(dataset(tensor(a), tensor(b)))
  const [w1, w2] = toFlat(model.weights)
  const intercept = toFlat(model.intercept)[0]
  const x1 = a.map((r) => r[0])
  const x2 = a.map((r) => r[1])
  const p = toFlat(model.expect(tensor(a)))
  const accuracy = b.reduce((s: number, v, i) => s + ((p[i] > 0.5 ? 1 : 0) === v ? 1 : 0), 0) / b.length
  // The boundary w·x + b = 0, drawn across the data and clipped to its vertical extent.
  const [lo, hi] = [Math.min(...x1), Math.max(...x1)]
  const [ylo, yhi] = [Math.min(...x2), Math.max(...x2)]
  const boundary: { x: number[]; y: number[] } = { x: [], y: [] }
  for (let k = 0; k <= 100; k++) {
    const u = lo + ((hi - lo) * k) / 100
    const v = -(intercept + w1 * u) / w2
    if (v >= ylo && v <= yhi) {
      boundary.x.push(u)
      boundary.y.push(v)
    }
  }
  return { kind, x1, x2, labels: b, boundary, weights: [w1, w2], intercept, accuracy }
}

/** The prelude's namespaces and members, with each one's signature, doc and source module (from the entries). */
function PreludeReference({ prelude }: { prelude: Prelude }) {
  const top = members(prelude, null)
  return (
    <details className="group rounded-md border bg-card text-xs">
      <summary className="cursor-pointer px-3 py-2 text-muted-foreground select-none">
        Prelude reference: {prelude.entries.length} functions in {prelude.namespaces.length} namespaces, plus{' '}
        {top.map((e) => e.name).join(', ')} and Math
      </summary>
      <div className="flex flex-col gap-1 border-t px-3 py-2">
        {[...prelude.namespaces]
          .sort((a, b) => a.name.localeCompare(b.name))
          .map((n) => {
            const list = members(prelude, n.name)
            return (
              <details key={n.name}>
                <summary className="cursor-pointer py-0.5 select-none">
                  <span className="font-mono font-medium">{n.name}</span>{' '}
                  <span className="text-muted-foreground">
                    · {list.length} · {n.doc} · <span className="font-mono">{n.source}</span>
                  </span>
                </summary>
                <ul className="my-1 ml-4 flex flex-col gap-0.5">
                  {list.map((e) => (
                    <li key={e.name} className="grid grid-cols-[minmax(14rem,auto)_1fr] gap-3">
                      <span className="font-mono">{signature(e)}</span>
                      <span className="text-muted-foreground">
                        {e.doc} <span className="font-mono opacity-75">{e.source}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )
          })}
      </div>
    </details>
  )
}

const PRELUDE_TASK = call('applied/interpreter/prelude')

/** The editor, the program's run in the compute worker, and the fitted model. */
export function DatasetFromCode() {
  const [example, setExample] = useState<ExampleKey>('regression')
  const [code, setCode] = useState<string>(EXAMPLES.regression.code)
  const [seed, setSeed] = useState(0)
  const [auto, setAuto] = useState(true)
  // With auto-run off, the program runs only on the Run button (or Mod-Enter): `submitted` is what was run.
  const [submitted, setSubmitted] = useState({ code: EXAMPLES.regression.code, at: 0 })
  const source = auto ? code : submitted.code

  // The entry's parameters as controls, and the run in the worker (aifn-render's program helpers).
  const entry = useEntryArgs(source)
  const shape = entry.controls
  const run = useProgram(source, { seed, args: entry.args, prelude: PRELUDE_TASK, runKey: submitted.at })
  const { result } = run

  const fitted = useMemo((): { fit: Fit | null; error: string | null } => {
    if (!result || !result.ok) return { fit: null, error: null }
    try {
      return { fit: fitResult(example, result.value), error: null }
    } catch (err) {
      return { fit: null, error: err instanceof Error ? err.message : String(err) }
    }
  }, [result, example])

  // The last good fit stays drawn while the program has an error (state adjusted during render, React's pattern).
  const [lastGood, setLastGood] = useState<Fit | null>(null)
  if (fitted.fit && fitted.fit !== lastGood) setLastGood(fitted.fit)
  const fit = lastGood?.kind === example ? lastGood : null

  const pick = (k: ExampleKey) => {
    setExample(k)
    setCode(EXAMPLES[k].code)
    setSubmitted({ code: EXAMPLES[k].code, at: submitted.at + 1 })
  }
  const runNow = () => setSubmitted({ code, at: submitted.at + 1 })

  const xa = useAxis({ label: example === 'regression' ? 'x' : 'x₁' })
  const ya = useAxis({ label: example === 'regression' ? 'y' : 'x₂' })

  return (
    <Figure
      title="Interpreter: a dataset from code"
      purpose="A short JavaScript program makes the data; the same program and seed always give the same draws, and the model is refitted on every run."
      defaultSize="L"
      controls={
        <>
          <Select
            label="example"
            value={example}
            onChange={pick}
            options={(Object.keys(EXAMPLES) as ExampleKey[]).map((k) => ({ value: k, label: EXAMPLES[k].label }))}
          />
          <NumberField label="run seed" value={seed} onChange={setSeed} min={0} max={9999} step={1} />
          <Switch label="run as you type" checked={auto} onChange={setAuto} />
          <div className="flex items-end">
            <Button size="sm" variant="outline" onClick={runNow}>
              Run (⌘↵)
            </Button>
          </div>
        </>
      }
      equation={
        <div className="flex flex-col gap-2 text-left">
          <CodeEditor
            value={code}
            onChange={setCode}
            prelude={prelude}
            errors={run.errors}
            onRun={runNow}
            label="Program"
          />
          {Object.keys(shape.space.dims).length > 0 && (
            <div className="flex flex-col gap-1.5">
              <div className="text-xs text-muted-foreground">
                parameters of <span className="font-mono">make()</span>, from its signature and JSDoc
              </div>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] items-end gap-x-6 gap-y-3">
                <EntryControls key={entry.controlsKey} {...shape} />
              </div>
            </div>
          )}
          <ProgramStatus run={run} problem={fitted.error} />
          <PreludeReference prelude={prelude} />
        </div>
      }
      readouts={
        fit?.kind === 'regression' ? (
          <>
            <Readout label="slope" value={formatValue(fit.slope)} />
            <Readout label="intercept" value={formatValue(fit.intercept)} />
            <Readout label="R²" value={formatValue(fit.r2)} />
            <Readout label="σ̂" value={formatValue(fit.sd)} />
            <Readout label="n" value={fit.x.length} />
          </>
        ) : fit?.kind === 'classification' ? (
          <>
            <Readout label="w" value={`[${fit.weights.map(formatValue).join(', ')}]`} />
            <Readout label="b" value={formatValue(fit.intercept)} />
            <Readout label="training accuracy" value={formatValue(fit.accuracy)} />
            <Readout label="n" value={fit.labels.length} />
          </>
        ) : null
      }
      caption="Edit the program: completion lists the namespaces (math, array, random, stats, linalg, signal, learn) and, after a dot, their functions with signatures, docs and source modules; hovering a name shows its doc, and errors are underlined. The parameters of make() get controls from its signature: an integer default is an int, a decimal a real, and a JSDoc @param {int} n [20, 1000] line sets the type and range. Changing the run seed changes every draw; seed(…) inside the program keys it further. The program runs in a worker, so a runaway loop is stopped by the next edit."
    >
      <Plot x={xa} y={ya}>
        {fit?.kind === 'regression' && (
          <>
            <Points name="data" x={fit.x} y={fit.y} muted />
            <Curve name="least-squares line" x={fit.line.x} y={fit.line.y} slot={0} />
          </>
        )}
        {fit?.kind === 'classification' && (
          <>
            <Points name="data" x={fit.x1} y={fit.x2} group={fit.labels} groupNames={['label 0', 'label 1']} />
            <Curve name="decision boundary p = ½" x={fit.boundary.x} y={fit.boundary.y} emphasis />
          </>
        )}
      </Plot>
    </Figure>
  )
}
