/**
 * Views of a linear-chain CRF over CRF++ feature templates, shared by the CRF pages: a template editor with presets and
 * parse errors at their line; the CRF++ tabular picture of the cells a template reads around a position; the feature
 * strings firing at a position with their weights; the top features × labels and the transition weights as heatmaps;
 * and the training curves.
 */
import { useMemo, type ReactNode } from 'react'
import type { FiringFeature, TemplateCrf } from 'aifn-applied/inference/sequence-models'
import { topFeatures, transitionWeights } from 'aifn-applied/inference/sequence-models'
import type { FeatureTemplates, TemplateSyntaxError, TokenRows } from 'aifn/text/features'
import { StatusText } from '@lab/controls'
import { seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { cn } from '@lab/lib/utils'
import { Button } from '@lab/ui/button'
import { Curve, Plot, Plots, Raster, useAxis } from '@lab/viz'

/** The result of parsing a template source: the templates, or the error with its line and column. */
export type ParsedTemplates =
  { templates: FeatureTemplates; error: null } | { templates: null; error: TemplateSyntaxError | Error }

const fmt = (v: number) => (v === 0 ? '0' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2))

// ── Template editor ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A CRF++ template editor: a monospace text area with line numbers, preset buttons, a summary of what was parsed, and
 * a parse error marked at its line.
 */
export function TemplateEditor({
  value,
  onChange,
  presets,
  parsed,
}: {
  value: string
  onChange: (source: string) => void
  presets: Readonly<Record<string, string>>
  parsed: ParsedTemplates
}) {
  const lines = value.split('\n')
  const errorLine = parsed.error && 'line' in parsed.error ? (parsed.error as TemplateSyntaxError).line : null
  const t = parsed.templates
  return (
    <div className="col-span-full flex w-full max-w-3xl flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="text-muted-foreground">presets</span>
        {Object.entries(presets).map(([name, src]) => (
          <Button
            key={name}
            size="sm"
            variant={src === value ? 'default' : 'outline'}
            onClick={() => onChange(src)}
            aria-label={`preset ${name}`}
          >
            {name}
          </Button>
        ))}
      </div>
      <div className="flex max-h-72 overflow-auto rounded-lg border border-input font-mono text-xs leading-5">
        <div className="border-r border-border bg-muted/40 px-1.5 py-1.5 text-right text-muted-foreground select-none">
          {lines.map((_, i) => (
            <div key={i} className={cn(errorLine === i + 1 && 'font-bold text-destructive')}>
              {i + 1}
            </div>
          ))}
        </div>
        <textarea
          aria-label="CRF++ templates"
          spellCheck={false}
          wrap="off"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={Math.max(4, lines.length)}
          className="min-w-0 flex-1 resize-none bg-transparent px-2 py-1.5 leading-5 whitespace-pre outline-none"
        />
      </div>
      {parsed.error ? (
        <StatusText tone="error">{parsed.error.message.replace(/^parseTemplates: /, '')}</StatusText>
      ) : (
        t && (
          <StatusText>
            {t.templates.filter((x) => x.kind === 'unigram').length} unigram (U) and{' '}
            {t.templates.filter((x) => x.kind === 'bigram').length} bigram (B) templates, reading{' '}
            {t.columns === 0 ? 'no columns' : `columns 0–${t.columns - 1}`} and rows up to ±{t.reach}
          </StatusText>
        )
      )}
    </div>
  )
}

// ── The CRF++ tabular picture ────────────────────────────────────────────────────────────────────────────────────────

/**
 * The input around position n as CRF++'s documentation draws it: one row per token (with `_B-k` / `_B+k` rows past
 * the ends), one column per attribute. The current row is marked; the cells the chosen template reads are outlined
 * and labelled with their macro %x[r,c]. Clicking a token row moves the position there.
 */
export function TemplateCells({
  rows,
  columns,
  n,
  templates,
  template,
  onPosition,
}: {
  rows: TokenRows
  columns: readonly string[]
  n: number
  templates: FeatureTemplates | null
  /** Index of the template whose cells are outlined. */
  template: number
  onPosition: (n: number) => void
}) {
  const t = templates?.templates[template]
  const reach = Math.max(2, templates?.reach ?? 2)
  const read = new Map<string, string>()
  t?.macros.forEach((m) => read.set(`${n + m.row}:${m.column}`, `%x[${m.row},${m.column}]`))
  const at = Array.from({ length: 2 * reach + 1 }, (_, i) => n - reach + i)
  return (
    <table className="w-full border-collapse font-mono text-xs">
      <thead>
        <tr className="text-left text-muted-foreground">
          <th className="py-0.5 pr-2 font-medium">r</th>
          {columns.map((c, j) => (
            <th key={c} className="py-0.5 pr-2 font-medium">
              {c} <span className="text-muted-foreground/70">(col {j})</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {at.map((i) => {
          const inside = i >= 0 && i < rows.length
          return (
            <tr
              key={i}
              onClick={inside ? () => onPosition(i) : undefined}
              className={cn(
                'border-t border-border',
                i === n && 'bg-muted font-semibold',
                inside ? 'cursor-pointer' : 'text-muted-foreground',
              )}
            >
              <td className="py-0.5 pr-2 text-muted-foreground">{i - n > 0 ? `+${i - n}` : i - n}</td>
              {columns.map((_, j) => {
                const label = read.get(`${i}:${j}`)
                const cell = i < 0 ? `_B${i}` : i >= rows.length ? `_B+${i - rows.length + 1}` : rows[i][j]
                return (
                  <td key={j} className="py-0.5 pr-2">
                    <span
                      className={cn(
                        'inline-block rounded px-1',
                        label && 'outline-2 outline-primary',
                        !inside && !label && 'opacity-50',
                      )}
                    >
                      {cell}
                    </span>
                    {label && <span className="ml-1 text-[10px] text-primary">{label}</span>}
                  </td>
                )
              })}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

// ── Firing features ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The strings firing at position n, each conjoined with the label k (and, for bigram strings, the previous label i)
 * as `U01:o/HYPH` or `B/O→HYPH`, with its weight λ. Their sums are log ψ_n(k) and log Ψ_{n−1}(i, k). Clicking a row
 * picks its template.
 */
export function FiringFeatureList({
  features,
  labels,
  label,
  previous,
  trained,
  template,
  templates,
  onTemplate,
}: {
  features: readonly FiringFeature[]
  labels: readonly string[]
  /** k, the label of the cell. */
  label: number
  /** i, the previous label for bigram strings (null at n = 0). */
  previous: number | null
  trained: boolean
  template: number
  templates: FeatureTemplates | null
  onTemplate: (t: number) => void
}) {
  const K = labels.length
  const mode = useTheme().resolved
  const rows = features.map((f) => {
    const t = templates?.templates.findIndex((x) => x.text === f.template) ?? -1
    const w = f.kind === 'unigram' ? f.weights[label] : previous === null ? 0 : f.weights[previous * K + label]
    const name =
      f.kind === 'unigram' ? `${f.string}/${labels[label]}` : `${f.string}/${labels[previous ?? 0]}→${labels[label]}`
    return { f, t, w, name }
  })
  const uni = rows.filter((r) => r.f.kind === 'unigram').reduce((a, r) => a + r.w, 0)
  const bi = rows.filter((r) => r.f.kind === 'bigram').reduce((a, r) => a + r.w, 0)
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.w)))
  return (
    <div className="text-xs">
      <table className="w-full border-collapse">
        <thead>
          <tr className="text-left text-muted-foreground">
            <th className="py-0.5 pr-2 font-medium">feature f(y, x, n) that fires</th>
            <th className="py-0.5 pr-2 font-medium">template</th>
            <th className="py-0.5 text-right font-medium">weight λ</th>
            <th className="w-24 py-0.5" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              onClick={() => r.t >= 0 && onTemplate(r.t)}
              className={cn('cursor-pointer border-t border-border', r.t === template && 'bg-muted')}
            >
              <td className={cn('py-0.5 pr-2 font-mono', r.f.id < 0 && 'text-muted-foreground line-through')}>
                {r.name}
              </td>
              <td className="py-0.5 pr-2 font-mono text-muted-foreground">{r.f.template}</td>
              <td className="py-0.5 text-right font-mono tabular-nums">
                {r.f.id < 0 ? 'unseen' : trained ? fmt(r.w) : '—'}
              </td>
              <td className="py-0.5 pl-2">
                {trained && r.f.id >= 0 && (
                  <div className="relative h-2 w-24 rounded bg-muted">
                    <div
                      className="absolute top-0 h-2 rounded"
                      style={{
                        left: r.w >= 0 ? '50%' : `${50 - (50 * Math.abs(r.w)) / max}%`,
                        width: `${(50 * Math.abs(r.w)) / max}%`,
                        background: r.w >= 0 ? seriesColor(mode, 0) : seriesColor(mode, 1),
                      }}
                    />
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-1 text-muted-foreground">
        Σ unigram weights = log ψ_n({labels[label]}) = {trained ? fmt(uni) : '—'}
        {previous !== null && (
          <>
            {' '}
            · Σ bigram weights = log Ψ_n−1({labels[previous]}, {labels[label]}) = {trained ? fmt(bi) : '—'}
          </>
        )}
        . Struck-through strings were not seen in training (or fell below -f) and fire nothing.
      </div>
    </div>
  )
}

// ── Weights ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The top unigram strings by max_k |λ_{u,k}| as a strings × labels heatmap, and the transition weights of `B`. */
export function CrfWeightPlots({ crf, top = 16, scale = 1 }: { crf: TemplateCrf; top?: number; scale?: number }) {
  const best = useMemo(() => topFeatures(crf, top), [crf, top])
  const T = useMemo(() => transitionWeights(crf), [crf])
  const labels = crf.labels
  const strings = best.map((b) => b.string).reverse()
  const labelAxis = useAxis({ label: 'label y_n', categories: labels })
  const stringAxis = useAxis({ label: 'feature string', categories: strings })
  const fromAxis = useAxis({ label: 'previous y_n−1', categories: labels })
  const toAxis = useAxis({ label: 'label y_n', categories: labels })
  const z = useMemo(() => best.map((b) => b.weights).reverse(), [best])
  return (
    <Plots cols={2} scale={scale}>
      <Plot x={labelAxis} y={stringAxis} title={`largest weights λ (top ${best.length} strings)`}>
        {z.length > 0 && (
          <Raster x={labels.map((_, k) => k)} y={strings.map((_, i) => i)} z={z} scale="diverging" valueLabel="λ" />
        )}
      </Plot>
      <Plot x={toAxis} y={fromAxis} title={T ? 'transition weights λ_B(i, k)' : 'no plain B template: no transitions'}>
        {T && <Raster x={labels.map((_, k) => k)} y={labels.map((_, k) => k)} z={T} scale="diverging" valueLabel="λ" />}
      </Plot>
    </Plots>
  )
}

// ── Training curves ──────────────────────────────────────────────────────────────────────────────────────────────────

/** NLL (and the full objective), the (pseudo-)gradient norm on a log axis, and the non-zero weights, per step. */
export function CrfTrainingPlots({
  history,
  maxSteps,
  stepLabel = 'iteration',
  runKey,
  extra,
}: {
  history: {
    readonly objective: readonly number[]
    readonly nll: readonly number[]
    readonly gradNorm: readonly number[]
    readonly active: readonly number[]
  } | null
  maxSteps: number
  stepLabel?: string
  runKey: unknown
  extra?: ReactNode
}) {
  const x = useMemo(() => (history ? history.nll.map((_, i) => i) : []), [history])
  const stepAxis = useAxis({ label: stepLabel, range: [0, maxSteps], key: maxSteps, integer: true })
  const nllAxis = useAxis({ label: 'nats', hold: 'union', key: runKey, range: [0, undefined] })
  const gradAxis = useAxis({ label: '‖∇‖ (log)', log: true, hold: 'union', key: runKey })
  const activeAxis = useAxis({ label: 'non-zero weights', hold: 'union', key: runKey, range: [0, undefined] })
  return (
    <Plots cols={3} scale={0.45}>
      <Plot x={stepAxis} y={nllAxis} title="training objective">
        {history && <Curve name="objective L(λ)" x={x} y={history.objective} slot={0} />}
        {history && <Curve name="−log-likelihood" x={x} y={history.nll} slot={0} dashed />}
      </Plot>
      <Plot x={stepAxis} y={gradAxis} title="gradient norm">
        {history && <Curve name="‖∇L‖ (pseudo-gradient for OWL-QN)" x={x} y={history.gradNorm} slot={1} />}
      </Plot>
      <Plot x={stepAxis} y={activeAxis} title="active weights">
        {history && <Curve name="non-zero λ" x={x} y={history.active} slot={2} />}
        {extra}
      </Plot>
    </Plots>
  )
}
