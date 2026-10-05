import { useMemo, useState, type ReactNode } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { precisionRecallCurve } from 'aifn-compute/learning/metrics'
import {
  hyphenationPatterns,
  liangHyphenate,
  liangResult,
  liangSteps,
  markHyphens,
  type HyphenationPatterns,
  type LiangState,
  type PatgenPass,
} from 'aifn-compute/text/hyphenation'
import { mobyHyphenation, type HyphenationData } from 'aifn-methods/data/real/hyphenation'
import {
  BiRnnTagger,
  crfHyphenate,
  gapLabels,
  hyphenScores,
  rnnProbabilities,
  rnnSaliency,
  thresholdScores,
  WindowTagger,
  windowProbabilities,
  windowSaliency,
  type BiRnnTaggerParams,
  type CrfHyphenationSnapshot,
  type HyphenScores,
  type LiangSnapshot,
  type TaggerSnapshot,
  type WindowTaggerParams,
} from 'aifn-methods/text/hyphenation'
import { Button, Player, Slider } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { seriesColor } from 'aifn-render/design'
import { useTheme } from 'aifn-render/design'
import { cn } from 'aifn-render/lib/utils'
import { choice, int, row, useFigureState, type FigureState } from 'aifn-render/state'
import { Input } from 'aifn-render/ui/input'
import { TrainControls, useTrainedRun } from '@lab/views'
import { crfTask, DEFAULT_CRF } from './crf-tasks'
import { liangTask, MARGINS, SPLIT_SEED, taggerTask, type LiangSettings, type TaggerSettings } from './tasks'
import { Bars, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'

const LONGEST = 15

const pct = (v: number) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '—')
const LIANG_SLOT = 0
const WINDOW_SLOT = 1
const RNN_SLOT = 2
const CRF_SLOT = 3
const EXAMPLES = ['hyphenation', 'computer', 'information', 'probability', 'beautiful', 'understanding', 'monkey']

/** The patterns after `step` passes: the union of every pass's additions so far, merged. */
function patternsAt(passes: readonly PatgenPass[], step: number): HyphenationPatterns {
  return hyphenationPatterns(passes.slice(0, step).flatMap((p) => p.added))
}

const fHalfOf = (c: { tp: number; fp: number; fn: number }) => (1.25 * c.tp) / (1.25 * c.tp + 0.25 * c.fn + c.fp)
const precisionOf = (c: { tp: number; fp: number }) => (c.tp + c.fp > 0 ? c.tp / (c.tp + c.fp) : NaN)
const recallOf = (c: { tp: number; fn: number }) => (c.tp + c.fn > 0 ? c.tp / (c.tp + c.fn) : NaN)

// ── Small views ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** A word with its hyphens marked against the dictionary: right (green), wrong (red), missed (a muted dot). */
export function Marked({
  word,
  hyphens,
  truth,
}: {
  word: string
  hyphens: readonly number[]
  truth: readonly number[] | null
}) {
  const on = new Set(hyphens)
  const want = new Set(truth ?? [])
  return (
    <span className="font-mono text-base tracking-wide">
      {[...word].map((c, i) => {
        const last = i === word.length - 1
        let mark: ReactNode = null
        if (!last && on.has(i))
          mark = (
            <span
              className={cn(
                truth === null
                  ? 'text-foreground'
                  : want.has(i)
                    ? 'text-success'
                    : 'rounded-sm bg-destructive/15 text-destructive',
              )}
            >
              -
            </span>
          )
        else if (!last && want.has(i)) mark = <span className="text-muted-foreground/60">·</span>
        return (
          <span key={i}>
            {c}
            {mark}
          </span>
        )
      })}
    </span>
  )
}

/**
 * The classic layout of Liang's matching (Liang 1983; the TeXbook, appendix H): the dotted word, each pattern that
 * matched on its own line under the letters it covers with its digits in the gaps, and the gap values (the largest
 * digit per gap) at the bottom. Odd values inside the margins are hyphens. Patterns that fired at the last step are
 * outlined.
 */
function LiangLayout({ state, hyphens }: { state: LiangState; hyphens: readonly number[] }) {
  const chars = [...state.dotted]
  const cols = 2 * chars.length + 1
  const fresh = new Set(state.fired.map((m) => `${m.pattern}@${m.at}`))
  const hyphenSlots = new Set(hyphens.map((i) => i + 2))
  // Every cell is placed explicitly (row and column): auto-placement would wrap a digit left of a letter onto a new row.
  const cell = (key: string, row: number, col: number, content: ReactNode, className?: string) => (
    <span key={key} style={{ gridRow: row, gridColumn: col + 1 }} className={cn('text-center leading-5', className)}>
      {content}
    </span>
  )
  const charRow = (prefix: string, row: number, from: number, to: number, className?: string) =>
    chars.slice(from, to).map((c, j) => cell(`${prefix}c${from + j}`, row, 2 * (from + j) + 1, c, className))
  const last = state.matches.length + 2
  return (
    <div className="max-h-80 overflow-auto rounded border border-border p-2">
      <div
        className="grid w-max gap-x-0 font-mono text-sm"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0.75rem, auto))` }}
      >
        <div className="contents">{charRow('w', 1, 0, chars.length, 'font-semibold')}</div>
        {state.matches.map((m, r) => (
          <div
            key={`${m.pattern}@${m.at}`}
            className={cn('contents', fresh.has(`${m.pattern}@${m.at}`) ? 'text-foreground' : 'text-muted-foreground')}
          >
            {charRow(
              `m${r}`,
              r + 2,
              m.at,
              m.at + m.letters.length,
              fresh.has(`${m.pattern}@${m.at}`) ? 'bg-muted' : undefined,
            )}
            {m.values.map((v, j) =>
              v > 0
                ? cell(
                    `m${r}d${j}`,
                    r + 2,
                    2 * (m.at + j),
                    v,
                    cn('text-xs', v % 2 === 1 ? 'text-primary' : '', fresh.has(`${m.pattern}@${m.at}`) && 'bg-muted'),
                  )
                : null,
            )}
          </div>
        ))}
        <div className="contents border-t">
          {chars.map((c, k) => cell(`fc${k}`, last, 2 * k + 1, c, 'border-t border-border font-semibold'))}
          {state.slots.map((v, k) =>
            k >= 2 && k <= chars.length - 2
              ? cell(
                  `fs${k}`,
                  last,
                  2 * k,
                  hyphenSlots.has(k) ? '-' : v,
                  cn(
                    'border-t border-border text-xs',
                    hyphenSlots.has(k)
                      ? 'font-bold text-success'
                      : v % 2 === 1
                        ? 'text-primary'
                        : 'text-muted-foreground',
                  ),
                )
              : cell(`fs${k}`, last, 2 * k, '', 'border-t border-border'),
          )}
        </div>
      </div>
    </div>
  )
}

export function ScoreTable({
  rows,
}: {
  rows: { name: string; slot: number; scores: HyphenScores | null; note?: string }[]
}) {
  const mode = useTheme().resolved
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs text-muted-foreground">
          <th className="py-1 pr-3 font-medium">held-out words</th>
          <th className="py-1 pr-3 font-medium">precision</th>
          <th className="py-1 pr-3 font-medium">recall</th>
          <th className="py-1 pr-3 font-medium">F₀.₅</th>
          <th className="py-1 pr-3 font-medium">F₁</th>
          <th className="py-1 pr-3 font-medium">wrong hyphens</th>
          <th className="py-1 font-medium">missed</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.name} className="border-t border-border">
            <td className="py-0.5 pr-3">
              <span
                className="mr-1.5 inline-block size-2 rounded-full"
                style={{ background: seriesColor(mode, r.slot) }}
              />
              {r.name}
              {r.note && <span className="ml-1 text-xs text-muted-foreground">{r.note}</span>}
            </td>
            <td className="py-0.5 pr-3 font-semibold">{r.scores ? pct(r.scores.precision) : '—'}</td>
            <td className="py-0.5 pr-3">{r.scores ? pct(r.scores.recall) : '—'}</td>
            <td className="py-0.5 pr-3 font-semibold">{r.scores ? pct(r.scores.fHalf) : '—'}</td>
            <td className="py-0.5 pr-3">{r.scores ? pct(r.scores.f1) : '—'}</td>
            <td className="py-0.5 pr-3">{r.scores ? r.scores.fp : '—'}</td>
            <td className="py-0.5">{r.scores ? r.scores.fn : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Picks the checkpoint of a run: the one picked on this run, else step 0. */
function usePicked<R>(run: R | null) {
  const [picked, setPicked] = useState<{ run: R | null; index: number } | null>(null)
  const index = picked && picked.run === run ? picked.index : 0
  return [index, (i: number) => setPicked({ run, index: i })] as const
}

// ── The page ─────────────────────────────────────────────────────────────────────────────────────────────────────────

export function HyphenationShowcase() {
  const data = useMemo(() => mobyHyphenation(stream(SPLIT_SEED)), [])
  const testLabels = useMemo(() => gapLabels(data.test.words), [data])

  // Liang: PATGEN pass by pass.
  const liangState = useFigureState({
    run: row('1 · pattern learning', {
      budget: choice(
        [
          { value: 250, label: '250' },
          { value: 500, label: '500' },
          { value: 1000, label: '1000' },
          { value: 0, label: 'no limit' },
        ],
        1000,
        { label: 'pattern budget' },
      ),
    }),
  })
  const liangSettings: LiangSettings = { budget: Number(liangState.run.budget) }
  const liang = useTrainedRun(liangSettings, liangTask)
  const liangRun = liang.run.value
  const [liangIndex, pickPass] = usePicked(liang.trained)
  const passIndex = Math.min(liangIndex, liangRun?.step ?? 0)
  const patterns = useMemo(() => (liangRun ? patternsAt(liangRun.passes, passIndex) : null), [liangRun, passIndex])

  // Neural taggers: both trained side by side.
  const taggerState = useFigureState({
    run: row('1 · training run', {
      steps: int(600, { ge: 1, suggestions: [200, 400, 600, 900], label: 'Adam steps (each model)' }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const taggerSettings: TaggerSettings = { steps: Number(taggerState.run.steps), seed: taggerState.run.seed }
  const tagger = useTrainedRun(taggerSettings, taggerTask)
  const taggerRun = tagger.run.value
  const shots = useMemo(() => taggerRun?.checkpoints ?? [], [taggerRun])
  const [shotPicked, pickShot] = usePicked(tagger.trained)
  const shotIndex = Math.min(shotPicked, Math.max(0, shots.length - 1))
  const shot = shots[shotIndex]
  const [threshold, setThreshold] = useState(0.5)
  const windowModel = useMemo(() => (taggerRun ? WindowTagger(taggerRun.windowConfig) : null), [taggerRun])
  const rnnModel = useMemo(() => (taggerRun ? BiRnnTagger(taggerRun.rnnConfig) : null), [taggerRun])

  // Held-out scores of all three.
  const liangScores = useMemo(
    () =>
      patterns
        ? hyphenScores(
            data.test.words,
            data.test.words.map((w) => liangHyphenate(patterns, w.word, MARGINS).hyphens),
          )
        : null,
    [patterns, data],
  )
  const windowScores = useMemo(
    () => (shot ? thresholdScores(testLabels, shot.windowTest, threshold) : null),
    [shot, testLabels, threshold],
  )
  const rnnScores = useMemo(
    () => (shot ? thresholdScores(testLabels, shot.rnnTest, threshold) : null),
    [shot, testLabels, threshold],
  )
  // The linear-chain CRF (classic CRF++ templates, L-BFGS), the fourth method; its own page varies the settings.
  const crf = useTrainedRun(DEFAULT_CRF, crfTask)
  const crfSnap = crf.run.value ?? null
  const scoreRows = [
    {
      name: "Liang's patterns",
      slot: LIANG_SLOT,
      scores: liangScores,
      note: liangRun ? `(pass ${passIndex}, ${patterns?.patterns.length ?? 0} patterns)` : '(press Train above)',
    },
    {
      name: 'window MLP',
      slot: WINDOW_SLOT,
      scores: windowScores,
      note: shot ? `(step ${shot.step}, threshold ${threshold.toFixed(2)})` : '(press Train)',
    },
    {
      name: 'BiLSTM',
      slot: RNN_SLOT,
      scores: rnnScores,
      note: shot ? `(step ${shot.step}, threshold ${threshold.toFixed(2)})` : '(press Train)',
    },
    {
      name: 'linear-chain CRF',
      slot: CRF_SLOT,
      scores: crfSnap?.testScores ?? null,
      note: crfSnap ? `(Viterbi, iteration ${crfSnap.scoredAt})` : '(press Train in row 4)',
    },
  ]

  return (
    <>
      <LiangFigure
        state={liangState}
        liang={liang}
        run={liangRun}
        passIndex={passIndex}
        pickPass={pickPass}
        patterns={patterns}
        budget={liangSettings.budget}
        data={data}
      />
      <TaggerFigure
        state={taggerState}
        tagger={tagger}
        run={taggerRun}
        shotIndex={shotIndex}
        pickShot={pickShot}
        threshold={threshold}
        setThreshold={setThreshold}
        testLabels={testLabels}
        liangScores={liangScores}
        scoreRows={scoreRows}
        steps={taggerSettings.steps}
        crf={crf}
      />
      <WordFigure
        data={data}
        patterns={patterns}
        passIndex={passIndex}
        windowModel={windowModel}
        windowParams={(shot?.window as WindowTaggerParams | undefined) ?? null}
        rnnModel={rnnModel}
        rnnParams={(shot?.rnn as BiRnnTaggerParams | undefined) ?? null}
        step={shot?.step ?? null}
        threshold={threshold}
        setThreshold={setThreshold}
        crf={crfSnap}
      />
    </>
  )
}

// ── Figure 1: PATGEN ─────────────────────────────────────────────────────────────────────────────────────────────────

function LiangFigure({
  state,
  liang,
  run,
  passIndex,
  pickPass,
  patterns,
  budget,
  data,
}: {
  state: FigureState<any> // oxlint-disable-line typescript/no-explicit-any -- the figure only passes it on
  liang: ReturnType<typeof useTrainedRun<LiangSettings, LiangSnapshot>>
  run: LiangSnapshot | null | undefined
  passIndex: number
  pickPass: (i: number) => void
  patterns: HyphenationPatterns | null
  budget: number
  data: HyphenationData
}) {
  const steps = run?.steps ?? 20
  const curves = useMemo(() => {
    if (!run) return null
    const x = run.history.map((_, k) => k)
    return {
      x,
      testP: run.history.map((h) => precisionOf(h.test)),
      testR: run.history.map((h) => recallOf(h.test)),
      trainP: run.history.map((h) => precisionOf(h.train)),
      trainR: run.history.map((h) => recallOf(h.train)),
      oddX: run.passes.flatMap((p, k) => (p.level % 2 === 1 ? [k + 1] : [])),
      oddY: run.passes.flatMap((p) => (p.level % 2 === 1 ? [p.added.length] : [])),
      evenX: run.passes.flatMap((p, k) => (p.level % 2 === 0 ? [k + 1] : [])),
      evenY: run.passes.flatMap((p) => (p.level % 2 === 0 ? [p.added.length] : [])),
    }
  }, [run])
  const passAxis = useAxis({ label: 'pass (level × pattern length)', range: [0, steps], key: steps, integer: true })
  const rateAxis = useAxis({ label: 'precision / recall', range: [0, 1] })
  const countAxis = useAxis({ label: 'patterns added', hold: 'union', key: liang.trained, range: [0, undefined] })
  const pass = run && passIndex > 0 ? run.passes[passIndex - 1] : null
  const counts = run?.history[passIndex]
  const marker = run ? (
    <Handle
      kind="x"
      at={passIndex}
      onDrag={(x) => pickPass(Math.max(0, Math.min(run.step, Math.round(x))))}
      label={`pass ${passIndex}`}
    />
  ) : null
  const done = run?.step ?? 0
  return (
    <Figure
      title="Liang's patterns, learned by PATGEN"
      purpose="TeX hyphenates by patterns with priorities, learned from a hyphenated word list level by level: odd levels add hyphens, even levels inhibit the wrong ones. Watch precision and recall on held-out words move pass by pass."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls run={liang as never} progress={done / steps} progressText={`${done} / ${steps} passes`} />
          {run && (
            <ControlRow label="2 · pass">
              <Player
                label="pass"
                value={passIndex}
                onChange={pickPass}
                count={run.step + 1}
                format={(k) => (k === 0 ? 'no patterns' : `pass ${k}`)}
              />
            </ControlRow>
          )}
        </>
      }
      readouts={{
        pass: (
          <>
            <Readout label="pass" value={run ? `${passIndex} / ${run.step}` : '—'} />
            <Readout
              label="level"
              value={
                pass
                  ? `${pass.level} (${pass.level % 2 === 1 ? 'hyphenating' : 'inhibiting'}), length ${pass.length}`
                  : '—'
              }
            />
            <Readout label="patterns" value={patterns ? patterns.patterns.length : '—'} />
          </>
        ),
        'held-out words': (
          <>
            <Readout label="precision" value={counts ? pct(precisionOf(counts.test)) : '—'} />
            <Readout label="recall" value={counts ? pct(recallOf(counts.test)) : '—'} />
            <Readout label="F₀.₅" value={counts ? pct(fHalfOf(counts.test)) : '—'} />
            <Readout
              label="train precision / recall"
              value={counts ? `${pct(precisionOf(counts.train))} / ${pct(recallOf(counts.train))}` : '—'}
            />
          </>
        ),
      }}
      caption={
        <>
          aifn <code>patgenSteps</code> (in <code>aifn/text/hyphenation</code>) learns from the{' '}
          {data.train.words.length} training words of <code>mobyHyphenation</code>, one pass per step: at level k and
          pattern length L, every substring of a dotted training word with a digit in one gap is a candidate, counted
          good where digit k would fix that gap and bad where it would break it, and kept when good·w₊ − bad·w₋ ≥ the
          level&apos;s threshold. Five levels, lengths 2–7, about 20 passes in all (a second in the worker). The budget
          caps the pattern count: a pass keeps its best candidates until the set is full. Precision and recall are
          counted over every gap of the {data.test.words.length} held-out words (solid) and the training words (thin).
          Drag the pass marker on either chart or play the passes; the word figure below hyphenates with the patterns of
          the pass shown.
          {budget > 0 && ` Budget: ${budget} patterns.`}
        </>
      }
    >
      <Plots cols={2} scale={0.5}>
        <Plot x={passAxis} y={rateAxis} title="precision and recall by pass">
          {curves && <Curve name="train precision" x={curves.x} y={curves.trainP} muted thin />}
          {curves && <Curve name="train recall" x={curves.x} y={curves.trainR} muted thin dashed />}
          {curves && <Curve name="test precision" x={curves.x} y={curves.testP} slot={LIANG_SLOT} showPoints />}
          {curves && <Curve name="test recall" x={curves.x} y={curves.testR} slot={LIANG_SLOT} dashed showPoints />}
          {marker}
        </Plot>
        <Plot x={passAxis} y={countAxis} title="patterns added per pass">
          {curves && <Bars name="hyphenating (odd level)" x={curves.oddX} y={curves.oddY} slot={3} />}
          {curves && <Bars name="inhibiting (even level)" x={curves.evenX} y={curves.evenY} slot={4} />}
          {marker}
        </Plot>
      </Plots>
      {run ? (
        <div className="mt-2 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="max-h-64 overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-2 font-medium">pass</th>
                  <th className="py-1 pr-2 font-medium">level</th>
                  <th className="py-1 pr-2 font-medium">length</th>
                  <th className="py-1 pr-2 font-medium">candidates</th>
                  <th className="py-1 pr-2 font-medium">kept</th>
                  <th className="py-1 pr-2 font-medium">good</th>
                  <th className="py-1 font-medium">bad</th>
                </tr>
              </thead>
              <tbody>
                {run.passes.map((p, k) => (
                  <tr
                    key={k}
                    onClick={() => pickPass(k + 1)}
                    className={cn('cursor-pointer border-t border-border', k + 1 === passIndex && 'bg-muted')}
                  >
                    <td className="py-0.5 pr-2">{k + 1}</td>
                    <td className="py-0.5 pr-2">
                      {p.level} {p.level % 2 === 1 ? 'hyphenate' : 'inhibit'}
                    </td>
                    <td className="py-0.5 pr-2">{p.length}</td>
                    <td className="py-0.5 pr-2">{p.candidates.toLocaleString()}</td>
                    <td className="py-0.5 pr-2">{p.added.length}</td>
                    <td className="py-0.5 pr-2">{p.good}</td>
                    <td className="py-0.5">{p.bad}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-xs">
            <div className="mb-1 text-muted-foreground">
              {pass
                ? `patterns kept at pass ${passIndex} (best first${pass.added.length > 60 ? ', first 60' : ''}):`
                : 'Pick a pass (or click a row) to list the patterns it kept.'}
            </div>
            {pass && (
              <div className="flex flex-wrap gap-1 font-mono">
                {pass.added.slice(0, 60).map((p) => (
                  <span key={p} className="rounded border border-border px-1">
                    {p}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : null}
    </Figure>
  )
}

// ── Figure 2: neural taggers ─────────────────────────────────────────────────────────────────────────────────────────

function TaggerFigure({
  state,
  tagger,
  run,
  shotIndex,
  pickShot,
  threshold,
  setThreshold,
  testLabels,
  liangScores,
  scoreRows,
  steps,
  crf,
}: {
  state: FigureState<any> // oxlint-disable-line typescript/no-explicit-any -- the figure only passes it on
  tagger: ReturnType<typeof useTrainedRun<TaggerSettings, TaggerSnapshot>>
  run: TaggerSnapshot | null | undefined
  shotIndex: number
  pickShot: (i: number) => void
  threshold: number
  setThreshold: (t: number) => void
  testLabels: readonly number[]
  liangScores: HyphenScores | null
  scoreRows: { name: string; slot: number; scores: HyphenScores | null; note?: string }[]
  steps: number
  crf: ReturnType<typeof useTrainedRun<typeof DEFAULT_CRF, CrfHyphenationSnapshot>>
}) {
  const shots = run?.checkpoints ?? []
  const shot = shots[shotIndex]
  const losses = useMemo(() => {
    if (!run) return null
    const n = run.windowLosses.length
    const stride = Math.max(1, Math.floor(n / 300))
    const x: number[] = []
    const w: number[] = []
    const r: number[] = []
    for (let i = 0; i < n; i += stride) {
      x.push(i)
      w.push(run.windowLosses[i])
      r.push(run.rnnLosses[i])
    }
    return { x, w, r }
  }, [run])
  // Precision and recall against the threshold, and the precision–recall curve, at the checkpoint shown.
  const sweeps = useMemo(() => {
    if (!shot) return null
    const of = (p: Float64Array) => {
      const c = precisionRecallCurve([...testLabels], p, { positive: 1 })
      const t = Array.from(c.thresholds.data as Float64Array)
      const P = Array.from(c.y.data as Float64Array)
      const R = Array.from(c.x.data as Float64Array)
      // Thin to about 300 points, keeping the order (thresholds decrease).
      const stride = Math.max(1, Math.floor(t.length / 300))
      const keep = t.map((_, i) => i).filter((i) => i % stride === 0 && Number.isFinite(t[i]))
      return { t: keep.map((i) => t[i]), P: keep.map((i) => P[i]), R: keep.map((i) => R[i]) }
    }
    return { window: of(shot.windowTest), rnn: of(shot.rnnTest) }
  }, [shot, testLabels])
  const stepAxis = useAxis({ label: 'step', range: [0, run?.steps ?? steps], key: run?.steps })
  const lossAxis = useAxis({
    label: 'loss (nats per letter)',
    hold: 'union',
    key: tagger.trained,
    range: [0, undefined],
  })
  const thresholdAxis = useAxis({ label: 'threshold on P(hyphen)', range: [0, 1] })
  const rateAxis = useAxis({ label: 'precision / recall', range: [0, 1] })
  const recallAxis = useAxis({ label: 'recall', range: [0, 1] })
  const precisionAxis = useAxis({ label: 'precision', range: [0, 1] })
  const stepMarker = shot ? (
    <Handle
      kind="x"
      at={shot.step}
      onDrag={(s) => {
        let best = 0
        shots.forEach((c, i) => {
          if (Math.abs(c.step - s) < Math.abs(shots[best].step - s)) best = i
        })
        pickShot(best)
      }}
      label={`step ${shot.step}`}
    />
  ) : null
  const done = run?.step ?? 0
  const total = tagger.trained?.steps ?? steps
  const [ws, rs] = [scoreRows[1].scores, scoreRows[2].scores]
  return (
    <Figure
      title="Neural taggers: a window MLP and a BiLSTM"
      purpose="Two networks learn P(hyphen after this letter) from the same training words: a NETtalk-style MLP that sees seven letters, and a bidirectional LSTM that reads the whole word. A threshold turns probabilities into hyphens; raising it trades recall for precision."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls run={tagger as never} progress={done / total} progressText={`${done} / ${total} steps`} />
          {shots.length > 0 && (
            <ControlRow label="2 · checkpoint">
              <Player
                label="checkpoint"
                value={shotIndex}
                onChange={pickShot}
                count={shots.length}
                format={(k) => `step ${shots[k]?.step ?? 0}`}
              />
            </ControlRow>
          )}
          <ControlRow label="3 · threshold">
            <Slider label="hyphen if P ≥" value={threshold} onChange={setThreshold} min={0.05} max={0.95} step={0.01} />
          </ControlRow>
          <TrainControls
            label="4 · a linear-chain CRF (classic CRF++ templates, L-BFGS, -c 1)"
            run={crf as never}
            progress={crf.run.value ? crf.run.value.step / crf.run.value.maxSteps : 0}
            progressText={`${crf.run.value?.step ?? 0} / ${crf.run.value?.maxSteps ?? DEFAULT_CRF.maxSteps} iterations`}
          />
        </>
      }
      readouts={{
        'window MLP': (
          <>
            <Readout label="precision" value={ws ? pct(ws.precision) : '—'} />
            <Readout label="recall" value={ws ? pct(ws.recall) : '—'} />
            <Readout label="F₀.₅" value={ws ? pct(ws.fHalf) : '—'} />
          </>
        ),
        BiLSTM: (
          <>
            <Readout label="precision" value={rs ? pct(rs.precision) : '—'} />
            <Readout label="recall" value={rs ? pct(rs.recall) : '—'} />
            <Readout label="F₀.₅" value={rs ? pct(rs.fHalf) : '—'} />
          </>
        ),
      }}
      caption={
        <>
          aifn-methods <code>taggerTrainingRun</code> trains both models in the worker, one Adam step of each per step:
          the window MLP (7 letters × 8-dimensional embeddings → 32 tanh units → 1 logit; 256 windows a step) and a
          bidirectional LSTM (12-dimensional embeddings, 24 units each way, forget-gate bias 1; 32 words a step) over{' '}
          <code>.word.</code>, both by binary cross-entropy on the dictionary labels, using <code>aifn/nn/layers</code>{' '}
          and <code>trainingLoop</code>. Scores count every gap of the held-out words. A wrong hyphen is worse than a
          missed one in typesetting, so precision and F₀.₅ (precision weighted four times as heavily as recall) lead the
          table. Drag the threshold line on the middle chart, or the step marker on the loss chart, or play the
          checkpoints. On the precision–recall chart, the dots are each model at the threshold and Liang&apos;s patterns
          at their pass. The default run (600 steps) takes about 30–60 s in the browser. Row 4 trains a fourth method, a
          linear-chain CRF over CRF++ templates (letters −2 … +3 around the gap, n-grams across it, the vowel/consonant
          pattern) by L-BFGS, about 15–30 s; its hyphens are its Viterbi path, so it is one point on the
          precision–recall chart. The CRF page next to this one varies its templates, regularisation and optimiser.
        </>
      }
    >
      <Plots cols={3} scale={0.5}>
        <Plot x={stepAxis} y={lossAxis} title="training loss">
          {losses && <Curve name="window MLP" x={losses.x} y={losses.w} slot={WINDOW_SLOT} thin />}
          {losses && <Curve name="BiLSTM" x={losses.x} y={losses.r} slot={RNN_SLOT} thin />}
          {stepMarker}
        </Plot>
        <Plot x={thresholdAxis} y={rateAxis} title="precision and recall against the threshold">
          {sweeps && <Curve name="window precision" x={sweeps.window.t} y={sweeps.window.P} slot={WINDOW_SLOT} />}
          {sweeps && <Curve name="window recall" x={sweeps.window.t} y={sweeps.window.R} slot={WINDOW_SLOT} dashed />}
          {sweeps && <Curve name="BiLSTM precision" x={sweeps.rnn.t} y={sweeps.rnn.P} slot={RNN_SLOT} />}
          {sweeps && <Curve name="BiLSTM recall" x={sweeps.rnn.t} y={sweeps.rnn.R} slot={RNN_SLOT} dashed />}
          <Handle
            kind="x"
            at={threshold}
            onDrag={(t) => setThreshold(Math.round(100 * Math.max(0.05, Math.min(0.95, t))) / 100)}
            label={`threshold ${threshold.toFixed(2)}`}
          />
        </Plot>
        <Plot x={recallAxis} y={precisionAxis} title="precision–recall on held-out words">
          {sweeps && <Curve name="window MLP" x={sweeps.window.R} y={sweeps.window.P} slot={WINDOW_SLOT} />}
          {sweeps && <Curve name="BiLSTM" x={sweeps.rnn.R} y={sweeps.rnn.P} slot={RNN_SLOT} />}
          {ws && rs && (
            <Points
              name="at the threshold"
              x={[ws.recall, rs.recall]}
              y={[ws.precision, rs.precision]}
              group={[WINDOW_SLOT, RNN_SLOT]}
              size={9}
            />
          )}
          {scoreRows[3].scores && (
            <Points
              name="linear-chain CRF (Viterbi)"
              x={[scoreRows[3].scores.recall]}
              y={[scoreRows[3].scores.precision]}
              slot={CRF_SLOT}
              size={10}
            />
          )}
          {liangScores && (
            <Points
              name="Liang's patterns"
              x={[liangScores.recall]}
              y={[liangScores.precision]}
              slot={LIANG_SLOT}
              size={10}
            />
          )}
        </Plot>
      </Plots>
      <div className="mt-2 overflow-x-auto">
        <ScoreTable rows={scoreRows} />
      </div>
    </Figure>
  )
}

// ── Figure 3: any word ───────────────────────────────────────────────────────────────────────────────────────────────

function WordFigure({
  data,
  patterns,
  passIndex,
  windowModel,
  windowParams,
  rnnModel,
  rnnParams,
  step,
  threshold,
  setThreshold,
  crf,
}: {
  data: HyphenationData
  patterns: HyphenationPatterns | null
  passIndex: number
  windowModel: WindowTagger | null
  windowParams: WindowTaggerParams | null
  rnnModel: BiRnnTagger | null
  rnnParams: BiRnnTaggerParams | null
  step: number | null
  threshold: number
  setThreshold: (t: number) => void
  crf: CrfHyphenationSnapshot | null
}) {
  const mode = useTheme().resolved
  const [typed, setTyped] = useState('hyphenation')
  // A memo, so the React compiler can treat the word as a stable value in the memos below.
  const word = useMemo(
    () =>
      typed
        .toLowerCase()
        .replace(/[^a-z]/g, '')
        .slice(0, LONGEST),
    [typed],
  )
  const valid = word.length >= 2
  const truth = valid ? data.truth.hyphens(word) : null
  const split = !truth
    ? 'not in the list'
    : data.test.words.some((w) => w.word === word)
      ? 'a held-out word'
      : 'a training word'

  // Liang's matching, step by step through the start positions of .word.
  const liangStates = useMemo(() => {
    if (!patterns || !valid) return null
    const alg = liangSteps(patterns, word)
    const states: LiangState[] = [alg.init(undefined, stream(0))]
    while (!states.at(-1)!.done) states.push(alg.step(states.at(-1)!, { t: states.length - 1, stream: stream(0) }))
    return states
  }, [patterns, word, valid])
  const [liangStep, setLiangStep] = useState<{ word: string; step: number } | null>(null)
  const lastStep = (liangStates?.length ?? 1) - 1
  const at = liangStep && liangStep.word === word ? Math.min(liangStep.step, lastStep) : lastStep
  const liangState = liangStates?.[at] ?? null
  const liangFinal = liangStates ? liangResult(liangStates[lastStep], MARGINS) : null
  const liangNow = liangState ? liangResult(liangState, MARGINS) : null

  const probs = useMemo(() => {
    if (!valid || !windowModel || !windowParams || !rnnModel || !rnnParams) return null
    return {
      window: windowProbabilities(windowModel, windowParams, [word])[0],
      rnn: rnnProbabilities(rnnModel, rnnParams, [word])[0],
      windowSaliency: windowSaliency(windowModel, windowParams, word),
      rnnSaliency: rnnSaliency(rnnModel, rnnParams, word),
    }
  }, [valid, word, windowModel, windowParams, rnnModel, rnnParams])
  const cut = (p: readonly number[]) => p.flatMap((v, i) => (v >= threshold ? [i] : []))

  // Saliency as letters (across) × gaps (down): the window's slots mapped onto the word's letters.
  const [gap, setGap] = useState<{ word: string; gap: number } | null>(null)
  const n = word.length
  // By default the gap the BiLSTM is surest of.
  const surest = probs && probs.rnn.length > 0 ? probs.rnn.indexOf(Math.max(...probs.rnn)) : 0
  const focus = gap && gap.word === word ? Math.min(gap.gap, n - 2) : surest
  const radius = windowModel?.config.radius ?? 3
  const windowRow = probs
    ? Array.from({ length: n }, (_, j) => {
        const slot = j - focus + radius
        return slot >= 0 && slot < 2 * radius + 1 ? probs.windowSaliency[focus][slot] : NaN
      })
    : null
  const rnnRow = probs ? probs.rnnSaliency[focus] : null

  const gapLabelsOf = Array.from({ length: Math.max(0, n - 1) }, (_, i) => `${word[i]}|${word[i + 1]}`)
  const gapAxis = useAxis({ label: 'gap', categories: gapLabelsOf })
  const probAxis = useAxis({ label: 'P(hyphen)', range: [0, 1] })
  const letterAxis = useAxis({ label: 'letter', categories: [...word].map((c, j) => `${j} ${c}`) })
  const rowsAxis = useAxis({ label: 'model', categories: ['BiLSTM', 'MLP'] })

  const methods: { name: string; slot: number; hyphens: readonly number[] | null; note: string }[] = [
    {
      name: "Liang's patterns",
      slot: LIANG_SLOT,
      hyphens: liangFinal?.hyphens ?? null,
      note: patterns ? `pass ${passIndex}, ${patterns.patterns.length} patterns` : 'train the patterns above',
    },
    {
      name: 'window MLP',
      slot: WINDOW_SLOT,
      hyphens: probs ? cut(probs.window) : null,
      note: probs ? `step ${step}, P ≥ ${threshold.toFixed(2)}` : 'train the taggers above',
    },
    {
      name: 'BiLSTM',
      slot: RNN_SLOT,
      hyphens: probs ? cut(probs.rnn) : null,
      note: probs ? `step ${step}, P ≥ ${threshold.toFixed(2)}` : 'train the taggers above',
    },
    {
      name: 'linear-chain CRF',
      slot: CRF_SLOT,
      hyphens: crf && valid ? crfHyphenate(crf.crf, word).hyphens : null,
      note: crf ? `Viterbi, iteration ${crf.step}` : 'train the CRF above (row 4)',
    },
  ]

  return (
    <Figure
      title="Type any word"
      purpose="Each method's hyphenation of a word you type, side by side with the dictionary's when the word is listed; the patterns that fire at every gap, in the classic layout; and each network's probability and which letters it relied on."
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          <ControlRow label="word">
            <div className="flex flex-wrap items-center gap-2">
              <Input
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                className="max-w-48 font-mono"
                aria-label="word"
              />
              {EXAMPLES.map((w) => (
                <Button key={w} size="sm" variant="outline" onClick={() => setTyped(w)}>
                  {w}
                </Button>
              ))}
              <Button
                size="sm"
                variant="outline"
                onClick={() => setTyped(data.test.words[(word.length * 37 + 11) % data.test.words.length].word)}
              >
                a held-out word
              </Button>
            </div>
          </ControlRow>
          {liangStates && (
            <ControlRow label="Liang, start position">
              <Player
                label="start"
                value={at}
                onChange={(k) => setLiangStep({ word, step: k })}
                count={liangStates.length}
                format={(k) => (k === 0 ? 'nothing matched' : `patterns from “${liangStates[k].dotted[k - 1]}” (${k})`)}
                startReason="the word opens fully matched, so the comparison above it is complete; step back to replay"
              />
            </ControlRow>
          )}
          <ControlRow label="threshold">
            <Slider label="hyphen if P ≥" value={threshold} onChange={setThreshold} min={0.05} max={0.95} step={0.01} />
          </ControlRow>
        </>
      }
      readouts={
        valid ? (
          <>
            <Readout label="word" value={`${word} (${split})`} />
            <Readout label="dictionary" value={truth ? markHyphens(word, truth) : '—'} />
            {liangNow && <Readout label="Liang at this start" value={markHyphens(word, liangNow.hyphens)} />}
          </>
        ) : undefined
      }
      caption={
        <>
          Hyphens are marked against the Moby dictionary when the word is listed: green where the dictionary agrees, red
          where it does not, and a dot where the dictionary has a point the method missed. Dictionary points are not the
          only acceptable ones: Moby splits by syllable (&quot;man-y&quot;, &quot;a-bout&quot;), where a typesetter
          would not break, and some &quot;wrong&quot; hyphens are defensible breaks. Liang&apos;s layout lists every
          pattern that matched <code>.word.</code> under the letters it covers, its digits in the gaps (odd in the
          primary colour); the bottom row is the largest digit per gap, with odd ones hyphenated. The player steps
          through the start positions of <code>liangSteps</code>; the patterns starting at the current position are
          shaded. The bars are each network&apos;s P(hyphen) per gap, the line the threshold (drag it). The saliency
          chart shows, for the gap picked below the bars, how much the logit drops when each letter is replaced by
          padding (occlusion): the window MLP sees only three letters each side.
        </>
      }
    >
      {!valid ? (
        <div className="py-6 text-center text-sm text-muted-foreground">Type a word of at least two letters a–z.</div>
      ) : (
        <div className="flex flex-col gap-3">
          <table className="w-full text-sm">
            <tbody>
              <tr className="border-b border-border">
                <td className="py-1 pr-3 text-muted-foreground">dictionary (Moby)</td>
                <td className="py-1 pr-3">
                  {truth ? (
                    <Marked word={word} hyphens={truth} truth={null} />
                  ) : (
                    <span className="text-muted-foreground">{word} is not listed</span>
                  )}
                </td>
                <td className="py-1 text-xs text-muted-foreground">{split}</td>
              </tr>
              {methods.map((m) => (
                <tr key={m.name} className="border-b border-border">
                  <td className="py-1 pr-3">
                    <span
                      className="mr-1.5 inline-block size-2 rounded-full"
                      style={{ background: seriesColor(mode, m.slot) }}
                    />
                    {m.name}
                  </td>
                  <td className="py-1 pr-3">
                    {m.hyphens ? (
                      <Marked word={word} hyphens={m.hyphens} truth={truth} />
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="py-1 text-xs text-muted-foreground">{m.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="grid gap-3 lg:grid-cols-2">
            <div>
              <div className="mb-1 text-xs text-muted-foreground">
                Liang&apos;s patterns that matched, and the gap values
              </div>
              {liangState && liangNow ? (
                <LiangLayout state={liangState} hyphens={liangNow.hyphens} />
              ) : (
                <div className="py-6 text-center text-sm text-muted-foreground">
                  Train the patterns in the first figure.
                </div>
              )}
            </div>
            <div>
              <Plots cols={1} scale={0.4}>
                <Plot x={gapAxis} y={probAxis} title="P(hyphen) per gap">
                  {probs && (
                    <Bars
                      name="window MLP"
                      x={gapLabelsOf.map((_, i) => i - 0.18)}
                      y={probs.window}
                      width={0.34}
                      slot={WINDOW_SLOT}
                    />
                  )}
                  {probs && (
                    <Bars
                      name="BiLSTM"
                      x={gapLabelsOf.map((_, i) => i + 0.18)}
                      y={probs.rnn}
                      width={0.34}
                      slot={RNN_SLOT}
                    />
                  )}
                  {truth && (
                    <Points
                      name="dictionary point"
                      x={truth.map((i) => i)}
                      y={truth.map(() => 0.98)}
                      emphasis
                      size={7}
                    />
                  )}
                  <Handle
                    kind="y"
                    at={threshold}
                    onDrag={(t) => setThreshold(Math.round(100 * Math.max(0.05, Math.min(0.95, t))) / 100)}
                    label={`threshold ${threshold.toFixed(2)}`}
                  />
                </Plot>
              </Plots>
              {probs && (
                <>
                  <div className="mt-1 flex flex-wrap items-center gap-1 text-xs">
                    <span className="text-muted-foreground">saliency for gap</span>
                    {gapLabelsOf.map((g, i) => (
                      <button
                        key={i}
                        type="button"
                        aria-label={`gap ${g}`}
                        onClick={() => setGap({ word, gap: i })}
                        className={cn(
                          'rounded border px-1 font-mono',
                          i === focus ? 'border-foreground bg-muted' : 'border-border',
                        )}
                      >
                        {g}
                      </button>
                    ))}
                  </div>
                  <Plots cols={1} scale={0.32}>
                    <Plot x={letterAxis} y={rowsAxis} title={`which letters decide the gap ${gapLabelsOf[focus]}`}>
                      <Raster
                        x={[...word].map((_, j) => j)}
                        y={[0, 1]}
                        z={[rnnRow ?? [], windowRow ?? []]}
                        scale="diverging"
                        valueLabel="Δ logit"
                      />
                    </Plot>
                  </Plots>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </Figure>
  )
}
