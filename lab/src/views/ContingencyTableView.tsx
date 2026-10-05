import { useMemo, useState, type ReactNode } from 'react'
import { confusionMargins, type ConfusionMatrix } from 'aifn-compute/learning/metrics'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { chrome, interpolateColors, sequential } from 'aifn-render/design'
import { useTheme } from 'aifn-render/design'
import { cn } from 'aifn-render/lib/utils'
import { formatNumber } from 'aifn-render/viz'

export type ContingencyTableViewProps = {
  /** A confusion matrix of counts from `aifn/learning/metrics` (`confusionMatrix`): rows actual, columns predicted. */
  table: ConfusionMatrix
  /** A name per class, in the table's order (default the labels). */
  classNames?: readonly string[]
  /**
   * The index (in the table's order) of the positive class of a two-class table: names the cells TP, FN, FP, TN and
   * the margin rates TPR, FNR, TNR, FPR, PPV, FDR, NPV, FOR. Omitted, or with K > 2 classes, the rates are each
   * class's recall and precision.
   */
  positive?: number
  /** A share under each count: of every case (`all`, default), of its row, of its column, or none. */
  share?: 'all' | 'row' | 'column' | 'none'
  /** A colour per class (a palette slot's colour), shown as a dot by each class name, e.g. to match a scatter. */
  classColors?: readonly string[]
  /** The names of the two axes (default "actual" and "predicted"). */
  rowLabel?: string
  columnLabel?: string
  className?: string
}

type Cell = readonly [number, number]
/** What a margin rate divides: the cells summed on top and the cells summed underneath. */
type Focus = { numerator: Cell[]; denominator: Cell[] }
type Rate = { label: string; name: string; value: number; formula: string; focus: Focus }

const pct = (v: number) => (Number.isFinite(v) ? `${formatNumber(100 * v)}%` : '–')
const rate = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '–')

/**
 * A contingency table (confusion matrix) with full margins: actual classes as rows and predicted classes as columns,
 * each cell a count with an optional share, shaded lightly by count. Row totals carry the row rates (recall and miss
 * rate; TPR, FNR, TNR, FPR for two classes), column totals the column rates (precision and false discovery rate; PPV,
 * FDR, NPV, FOR), and the corner the grand total with the prevalence and the accuracy. Hovering a rate outlines the
 * cells it divides: the numerator solid, the rest of the denominator dashed. The margins come from
 * `aifn/learning/metrics` `confusionMargins`.
 */
export function ContingencyTableView({
  table,
  classNames,
  positive,
  classColors,
  share = 'all',
  rowLabel = 'actual',
  columnLabel = 'predicted',
  className,
}: ContingencyTableViewProps) {
  const { resolved: mode } = useTheme()
  const [focus, setFocus] = useState<Focus | null>(null)
  const K = table.classes.length
  const counts = useMemo(() => toFlat(table.matrix), [table])
  const m = useMemo(() => confusionMargins(table), [table])
  const binary = K === 2 && positive !== undefined && (positive === 0 || positive === 1)
  const names = classNames ?? table.classes.map(String)
  const at = (j: number, k: number) => counts[j * K + k]
  const row = (j: number) => Array.from({ length: K }, (_, k): Cell => [j, k])
  const col = (k: number) => Array.from({ length: K }, (_, j): Cell => [j, k])
  const all = Array.from({ length: K * K }, (_, i): Cell => [Math.floor(i / K), i % K])
  const [actual, predicted] = [toFlat(m.actual), toFlat(m.predicted)]
  const [rowRate, rowMiss, colRate, colMiss] = [m.rowRate, m.rowMiss, m.columnRate, m.columnMiss].map(toFlat)

  const cellName = (j: number, k: number) =>
    binary ? `${j === k ? 'T' : 'F'}${k === positive ? 'P' : 'N'}` : j === k ? 'correct' : ''
  const rowRates = (j: number): Rate[] => {
    const pos = binary && j === positive
    const hit: Rate = {
      label: binary ? (pos ? 'TPR' : 'TNR') : 'recall',
      name: binary ? (pos ? 'recall, sensitivity' : 'specificity') : `recall of ${names[j]}`,
      value: rowRate[j],
      formula: binary ? (pos ? 'TP / (TP + FN)' : 'TN / (TN + FP)') : 'diagonal / row total',
      focus: { numerator: [[j, j]], denominator: row(j) },
    }
    const miss: Rate = {
      label: binary ? (pos ? 'FNR' : 'FPR') : 'miss',
      name: binary ? (pos ? 'false-negative rate' : 'false-positive rate') : `miss rate of ${names[j]}`,
      value: rowMiss[j],
      formula: binary ? (pos ? 'FN / (TP + FN)' : 'FP / (TN + FP)') : '(row total − diagonal) / row total',
      focus: { numerator: row(j).filter(([, k]) => k !== j), denominator: row(j) },
    }
    return [hit, miss]
  }
  const colRates = (k: number): Rate[] => {
    const pos = binary && k === positive
    const hit: Rate = {
      label: binary ? (pos ? 'PPV' : 'NPV') : 'precision',
      name: binary ? (pos ? 'precision' : 'negative predictive value') : `precision of ${names[k]}`,
      value: colRate[k],
      formula: binary ? (pos ? 'TP / (TP + FP)' : 'TN / (TN + FN)') : 'diagonal / column total',
      focus: { numerator: [[k, k]], denominator: col(k) },
    }
    const miss: Rate = {
      label: binary ? (pos ? 'FDR' : 'FOR') : 'FDR',
      name: binary ? (pos ? 'false discovery rate' : 'false omission rate') : `false discovery rate of ${names[k]}`,
      value: colMiss[k],
      formula: binary ? (pos ? 'FP / (TP + FP)' : 'FN / (TN + FN)') : '(column total − diagonal) / column total',
      focus: { numerator: col(k).filter(([j]) => j !== k), denominator: col(k) },
    }
    return [hit, miss]
  }
  const corner: Rate[] = [
    ...(binary
      ? [
          {
            label: 'prevalence',
            name: 'prevalence',
            value: toFlat(m.prevalence)[positive ?? 0],
            formula: '(TP + FN) / n',
            focus: { numerator: row(positive ?? 0), denominator: all },
          },
        ]
      : []),
    {
      label: 'accuracy',
      name: 'accuracy',
      value: m.accuracy,
      formula: binary ? '(TP + TN) / n' : 'diagonal / n',
      focus: { numerator: Array.from({ length: K }, (_, j): Cell => [j, j]), denominator: all },
    },
  ]

  // Light shading by count: the first part of the sequential ramp, so text stays legible in both themes.
  const stops = sequential(mode)
  const max = Math.max(...counts, 1)
  const shade = (v: number) => interpolateColors(stops, 0.05 + 0.4 * (v / max))
  const inFocus = (cells: Cell[] | undefined, j: number, k: number) => !!cells?.some(([a, b]) => a === j && b === k)
  const shareOf = (j: number, k: number) =>
    share === 'all' ? at(j, k) / m.n : share === 'row' ? at(j, k) / actual[j] : at(j, k) / predicted[k]

  const rates = (list: Rate[], vertical = true) => (
    <div className={cn('flex gap-x-3 gap-y-0.5', vertical ? 'flex-col' : 'flex-wrap')}>
      {list.map((r) => (
        <button
          key={r.label}
          type="button"
          title={`${r.name} = ${r.formula}`}
          onPointerEnter={() => setFocus(r.focus)}
          onPointerLeave={() => setFocus(null)}
          onFocus={() => setFocus(r.focus)}
          onBlur={() => setFocus(null)}
          className="flex flex-wrap items-baseline justify-between gap-x-1.5 rounded px-1 text-left hover:bg-muted focus-visible:bg-muted focus-visible:outline-none"
        >
          <span className="text-muted-foreground">{r.label}</span>
          <span className="font-mono text-foreground tabular-nums">{rate(r.value)}</span>
        </button>
      ))}
    </div>
  )
  const label = (j: number) => (
    <span className="line-clamp-2">
      {classColors?.[j] && (
        <span
          aria-hidden
          className="mr-1 inline-block size-2 rounded-full align-middle"
          style={{ background: classColors[j] }}
        />
      )}
      {names[j]}
    </span>
  )
  const head = (key: string, children: ReactNode) => (
    <th key={key} scope="col" className="px-2 py-1 text-center align-bottom font-normal text-muted-foreground">
      {children}
    </th>
  )
  const { ink } = chrome(mode)

  return (
    <div className={cn('flex h-full w-full flex-col overflow-auto text-xs', className)}>
      <table className="h-full w-full table-fixed border-separate border-spacing-1">
        <colgroup>
          <col className="w-[18%]" />
          {names.map((n) => (
            <col key={n} />
          ))}
          <col className="w-[13%]" />
          <col className="w-[24%]" />
        </colgroup>
        <thead>
          <tr>
            <th />
            <th
              scope="colgroup"
              colSpan={K}
              className="border-b pb-0.5 text-center text-[11px] font-medium tracking-wide text-muted-foreground uppercase"
            >
              {columnLabel}
            </th>
            <th />
            <th />
          </tr>
          <tr>
            <th className="text-left align-bottom text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
              {rowLabel}
            </th>
            {names.map((n, k) => head(n, <span className="text-foreground">{label(k)}</span>))}
            {head('total', 'total')}
            {head('rates', 'row rates')}
          </tr>
        </thead>
        <tbody>
          {names.map((name, j) => (
            <tr key={name}>
              <th scope="row" className="px-1 text-left font-normal text-foreground">
                {label(j)}
              </th>
              {names.map((_, k) => {
                const num = inFocus(focus?.numerator, j, k)
                const den = !num && inFocus(focus?.denominator, j, k)
                return (
                  <td
                    key={k}
                    className={cn(
                      'rounded-md px-2 py-1 text-center align-middle transition-shadow',
                      focus && !num && !den && 'opacity-45',
                    )}
                    style={{
                      background: shade(at(j, k)),
                      boxShadow: num ? `inset 0 0 0 2px ${ink}` : undefined,
                      outline: den ? `1.5px dashed ${ink}` : undefined,
                      outlineOffset: den ? -3 : undefined,
                    }}
                  >
                    {cellName(j, k) && <div className="text-[10px] text-muted-foreground">{cellName(j, k)}</div>}
                    <div className="font-mono text-sm text-foreground tabular-nums">{formatNumber(at(j, k))}</div>
                    {share !== 'none' && (
                      <div className="font-mono text-[10px] text-muted-foreground tabular-nums">
                        {pct(shareOf(j, k))}
                      </div>
                    )}
                  </td>
                )
              })}
              <td className="rounded-md border px-2 py-1 text-center align-middle">
                <div className="text-[10px] text-muted-foreground">{binary ? (j === positive ? 'P' : 'N') : ''}</div>
                <div className="font-mono text-foreground tabular-nums">{formatNumber(actual[j])}</div>
              </td>
              <td className="px-0.5 align-middle">{rates(rowRates(j))}</td>
            </tr>
          ))}
          <tr>
            <th scope="row" className="px-1 text-left font-normal text-muted-foreground">
              total
            </th>
            {names.map((_, k) => (
              <td key={k} className="rounded-md border px-2 py-1 text-center align-middle">
                <div className="text-[10px] text-muted-foreground">{binary ? (k === positive ? 'PP' : 'PN') : ''}</div>
                <div className="font-mono text-foreground tabular-nums">{formatNumber(predicted[k])}</div>
              </td>
            ))}
            <td className="rounded-md border bg-muted px-2 py-1 text-center align-middle">
              <div className="text-[10px] text-muted-foreground">n</div>
              <div className="font-mono font-medium text-foreground tabular-nums">{formatNumber(m.n)}</div>
            </td>
            <td />
          </tr>
          <tr>
            <th scope="row" className="px-1 text-left align-top font-normal text-muted-foreground">
              column rates
            </th>
            {names.map((_, k) => (
              <td key={k} className="px-0.5 align-top">
                {rates(colRates(k))}
              </td>
            ))}
            <td colSpan={2} className="px-0.5 align-top">
              {rates(corner)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
