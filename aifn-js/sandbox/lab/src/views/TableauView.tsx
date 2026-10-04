import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { cn } from '@lab/lib/utils'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@lab/ui/table'
import { formatValue } from './format'

export type TableauViewProps = {
  /** The tableau, (m + 1) × (k + 1): constraint rows, then the reduced-cost row; the last column is the right-hand side. */
  tableau: Tensor
  /** The basic column of each constraint row (int32, length m). */
  basis: Tensor
  /** Labels of the k columns. */
  labels: readonly string[]
  /** The column entering at the next pivot (highlighted), or −1. */
  entering?: number
  /** The row leaving at the next pivot (highlighted), or −1. */
  leaving?: number
  /** The ratio test for the entering column, shown beside each row. */
  ratios?: Tensor
}

/** Short numbers for a dense table: four significant figures, and exact zeros as a dot. */
const cell = (v: number) => (Math.abs(v) < 1e-12 ? '·' : formatValue(Number(v.toPrecision(4))))

/**
 * A simplex tableau: one row per constraint, labelled by its basic variable, the reduced-cost row last, and the
 * right-hand side. The entering column, the leaving row and their pivot are highlighted, with the ratio test beside.
 */
export function TableauView({ tableau, basis, labels, entering = -1, leaving = -1, ratios }: TableauViewProps) {
  const [rows, width] = tableau.shape
  const t = toFlat(tableau)
  const b = toFlat(basis)
  const q = ratios ? toFlat(ratios) : null
  return (
    <div className="overflow-x-auto">
      <Table className="text-xs">
        <TableHeader>
          <TableRow>
            <TableHead className="h-7 px-2 text-xs">basis</TableHead>
            {labels.map((l, j) => (
              <TableHead
                key={j}
                className={cn(
                  'h-7 px-2 text-right text-xs',
                  j === entering && 'bg-muted font-semibold text-foreground',
                )}
              >
                {l}
              </TableHead>
            ))}
            <TableHead className="h-7 px-2 text-right text-xs">rhs</TableHead>
            {q && <TableHead className="h-7 px-2 text-right text-xs">ratio</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {Array.from({ length: rows }, (_, i) => {
            const objective = i === rows - 1
            return (
              <TableRow key={i} className={cn(objective && 'border-t-2', i === leaving && 'bg-muted/60')}>
                <TableCell className="px-2 py-1 font-medium">{objective ? 'd' : labels[b[i]]}</TableCell>
                {Array.from({ length: width }, (_, j) => (
                  <TableCell
                    key={j}
                    className={cn(
                      'px-2 py-1 text-right font-mono tabular-nums',
                      j === entering && 'bg-muted',
                      i === leaving && j === entering && 'font-bold text-foreground ring-1 ring-foreground ring-inset',
                    )}
                  >
                    {cell(t[i * width + j])}
                  </TableCell>
                ))}
                {q && (
                  <TableCell className="px-2 py-1 text-right font-mono text-muted-foreground tabular-nums">
                    {objective || !Number.isFinite(q[i]) ? '' : cell(q[i])}
                  </TableCell>
                )}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}
