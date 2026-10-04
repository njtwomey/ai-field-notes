import { useMemo, useState, type ReactNode } from 'react'
import {
  formatNumber,
  Handle,
  Plot,
  seriesLayers,
  type SeriesSpec,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
} from 'aifn-render'
import type { CoefficientPath } from '@/generated/contracts'
import { cn } from '@/lib/utils'
import { featureStyle, nearest } from './features'

/**
 * Coefficient paths against log₁₀ of the penalty, with a draggable penalty line that snaps to the precomputed grid,
 * and a table of the coefficients at that penalty. Precomputed paths come from python/mlc/figures/linear_models.py.
 */
export function CoefficientPathExplorer({
  path,
  symbol,
  ols,
  start,
  children,
}: {
  path: CoefficientPath
  /** The penalty's symbol in labels, e.g. λ or α. */
  symbol: string
  /** OLS coefficients, shown in the table for comparison. */
  ols?: number[]
  /** Starting penalty value; the nearest grid point is used. */
  start: number
  /** Extra readouts or controls, given the selected grid index. */
  children?: (index: number) => ReactNode
}) {
  const last = path.penalty.length - 1
  // The grid index of the penalty: the slider steps along the grid and the handle snaps to it. The parent's Figure
  // owns the state rows, so this body keeps its own index.
  const [i, setI] = useState(() => nearest(path.penalty.map(Math.log10), Math.log10(start)))
  const logs = useMemo(() => path.penalty.map((p) => Math.log10(p)), [path])

  const series = useMemo(
    (): SeriesSpec[] =>
      path.features.map((name, j) => ({ type: 'line', x: logs, y: path.coef[j], ...featureStyle(name) })),
    [path, logs],
  )

  const xAxis = useAxis({ label: `log₁₀ ${symbol}`, hold: 'union' })
  const yAxis = useAxis({ label: 'coefficient', hold: 'union' })
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-6">
        <div className="w-64">
          <Slider
            label={`penalty ${symbol}`}
            value={i}
            onChange={setI}
            min={0}
            max={last}
            step={1}
            format={(k) => formatNumber(path.penalty[k])}
          />
        </div>
        {children?.(i)}
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
        <Plot x={xAxis} y={yAxis} height={340}>
          {seriesLayers(series)}
          <Handle kind="x" at={logs[i]} label={symbol} onDrag={(x) => setI(nearest(logs, x))} />
        </Plot>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>feature</TableHead>
              <TableHead className="text-right">at {symbol}</TableHead>
              {ols && <TableHead className="text-right">OLS</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {path.features.map((name, j) => {
              const c = path.coef[j][i]
              return (
                <TableRow key={name}>
                  <TableCell className="py-1 font-mono text-xs">{name}</TableCell>
                  <TableCell
                    className={cn('py-1 text-right font-mono text-xs tabular-nums', c === 0 && 'text-muted-foreground')}
                  >
                    {c === 0 ? '0' : formatNumber(c)}
                  </TableCell>
                  {ols && (
                    <TableCell className="py-1 text-right font-mono text-xs text-muted-foreground tabular-nums">
                      {formatNumber(ols[j])}
                    </TableCell>
                  )}
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
