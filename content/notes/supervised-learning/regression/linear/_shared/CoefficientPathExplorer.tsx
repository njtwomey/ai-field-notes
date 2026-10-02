import { useMemo, type ReactNode } from 'react'
import { ParamSlider, XYChart, formatNumber, useParam, type Handle, type XYSeries } from 'aifn-render'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from 'aifn-render'
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
  const index = useParam(nearest(path.penalty.map(Math.log10), Math.log10(start)), { min: 0, max: last, step: 1 })
  const i = index.value
  const logs = useMemo(() => path.penalty.map((p) => Math.log10(p)), [path])

  const series = useMemo(
    (): XYSeries[] =>
      path.features.map((name, j) => ({ type: 'line', x: logs, y: path.coef[j], ...featureStyle(name) })),
    [path, logs],
  )
  const handles: Handle[] = [
    {
      kind: 'x',
      at: logs[i],
      label: symbol,
      // Snap to the nearest precomputed penalty.
      onDrag: (x) => index.set(nearest(logs, x)),
    },
  ]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-6">
        <div className="w-64">
          <ParamSlider label={`penalty ${symbol}`} param={index} format={(k) => formatNumber(path.penalty[k])} />
        </div>
        {children?.(i)}
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
        <XYChart height={340} series={series} xLabel={`log₁₀ ${symbol}`} yLabel="coefficient" handles={handles} />
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
