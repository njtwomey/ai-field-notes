import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { KnotPath, LarsPaths as Paths } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { cn } from '@/lib/utils'
import { featureStyle } from '../_shared/features'

type Method = 'lar' | 'lasso'

/** Coefficients at ‖w‖₁ = t: the paths are straight lines between consecutive knots. */
function at(path: KnotPath, t: number): number[] {
  const norms = path.l1_norm
  let k = norms.findIndex((n) => n >= t)
  if (k <= 0) k = k === 0 ? 1 : norms.length - 1
  const span = norms[k] - norms[k - 1]
  const f = span > 0 ? Math.min(Math.max((t - norms[k - 1]) / span, 0), 1) : 1
  return path.coef.map((row) => row[k - 1] + f * (row[k] - row[k - 1]))
}

/** LARS and LARS–lasso paths on the diabetes data, precomputed by python/mlc/figures/linear_models.py. */
export function LarsPaths() {
  const { data } = useFigure<Paths>('least-angle-regression/paths')
  const [method, setMethod] = useState<Method>('lasso')
  const path = data?.[method]
  const max = path ? path.l1_norm[path.l1_norm.length - 1] : 1
  // Position along the path as a fraction of the OLS solution's ‖w‖₁, so one control serves both methods.
  const t = useParam(0.4, { min: 0, max: 1, step: 0.005 })
  const shown = t.value * max

  const series = useMemo(
    (): XYSeries[] =>
      path && data
        ? data.features.map((name, j) => ({
            type: 'line',
            x: path.l1_norm,
            y: path.coef[j],
            ...featureStyle(name),
          }))
        : [],
    [path, data],
  )
  if (!data || !path) return null
  const coef = at(path, shown)
  const handles: Handle[] = [{ kind: 'x', at: shown, label: '‖w‖₁', onDrag: (x) => t.set(x / max) }]

  return (
    <Interactive
      title="Least-angle regression on the diabetes data"
      caption="Coefficients against ‖w‖₁, the total absolute size of the fit, from the empty model (left) to OLS (right). Each kink is a knot where a feature joins the active set. In plain LARS the ten features enter in ten steps and never leave. With the lasso modification, the coefficient of s3 reaches zero at α = 0.104 and s3 leaves the active set; it rejoins at α = 0.062 and ends with the opposite sign. The LARS–lasso path is exactly the lasso path. The table lists each change to the active set with the α at which it happens, in the lasso's scaling. Drag the ‖w‖₁ line to read the coefficients anywhere along the path."
      controls={
        <>
          <ParamChoice
            label="method"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'lar', label: 'LARS' },
              { value: 'lasso', label: 'LARS–lasso' },
            ]}
          />
          <ParamSlider label="‖w‖₁" param={t} format={(f) => formatNumber(f * max)} />
        </>
      }
      readout={
        <>
          <Readout label="knots" value={path.l1_norm.length - 1} />
          <Readout label="active features" value={coef.filter((c) => Math.abs(c) > 1e-9).length} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
        <XYChart height={360} series={series} xLabel="‖w‖₁" yLabel="coefficient" handles={handles} />
        <div className="space-y-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>at α</TableHead>
                <TableHead>event</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {path.events.map((e) => (
                <TableRow key={`${e.knot}-${e.feature}-${e.kind}`}>
                  <TableCell className="py-1 font-mono text-xs tabular-nums">{formatNumber(e.alpha)}</TableCell>
                  <TableCell className={cn('py-1 text-xs', e.kind === 'drop' && 'font-semibold')}>
                    {data.features[e.feature]} {e.kind === 'enter' ? 'joins' : 'leaves'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </Interactive>
  )
}
