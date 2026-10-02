/** The training figure of the stochastic vector field mixture pages. */
import { useMemo, type ReactNode } from 'react'
import type { SvfmRun } from 'aifn-applied/neural/ode-mixtures'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { Curve, Plot, useAxis } from '@lab/viz'
import { meanOf, thin } from './shared'

/** The training figure: the minibatch objective, the work of the realised paths and the path losses at checkpoints. */
export function TrainingFigure({
  id,
  run,
  runKey,
  marker,
  lossLabel,
}: {
  id: string
  run: SvfmRun | null
  runKey: unknown
  marker: ReactNode
  lossLabel: string
}) {
  const steps = run?.steps ?? 1
  const it = useAxis({ label: 'iteration', range: [0, steps], key: [runKey, steps], integer: true })
  const lossAxis = useAxis({ label: lossLabel, hold: 'union', key: runKey })
  const nfeAxis = useAxis({ label: 'function evaluations', range: [0, undefined], hold: 'union', key: runKey })
  const pathAxis = useAxis({ label: 'loss (log)', log: true, hold: 'union', key: runKey })
  const l = useMemo(() => (run ? thin(run.loss) : null), [run])
  const series = useMemo(() => {
    if (!run) return null
    const c = run.checkpoints
    return {
      x: c.map((k) => k.step),
      mean: c.map((k) => meanOf(k.nfe)),
      batch: c.map((k) => k.batchNfe),
      t: c.map((k) => Math.max(1e-6, k.transport)),
      v: c.map((k) => Math.max(1e-6, k.variance)),
    }
  }, [run])
  return (
    <Figure
      title="Training"
      id={id}
      purpose="The objective over the run, the work of solving the realised paths, and the path losses; the marker picks the model shown above."
      defaultSize="XL"
      caption={
        <>
          Left: the minibatch objective. Middle: function evaluations (NFE) of the shown points&apos; realised paths at
          each checkpoint, solved by Dormand–Prince (rtol 10⁻⁴): the mean over points solved one at a time, and all of
          them solved as one system (the NFE usually reported, which the hardest points set). Right: TLoss (eq. 8) and
          VLoss (eq. 9) on the whole set, whether or not they were trained on. Drag the iteration marker, or click a
          chart, to show that checkpoint&apos;s model above; it starts at the last checkpoint.
        </>
      }
    >
      <Dashboard>
        <DashboardRow>
          <DashboardCell>
            <Plot x={it} y={lossAxis} title="objective (minibatch)">
              {l && <Curve name={lossLabel} x={l.x} y={l.y} slot={2} />}
              {marker}
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={it} y={nfeAxis} title="NFE of the realised paths">
              {series && series.x.length > 0 && (
                <Curve name="mean per point" x={series.x} y={series.mean} slot={3} showPoints />
              )}
              {series && series.x.length > 0 && (
                <Curve name="all points as one system" x={series.x} y={series.batch} slot={4} showPoints dashed />
              )}
              {marker}
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={it} y={pathAxis} title="TLoss and VLoss">
              {series && series.x.length > 0 && <Curve name="TLoss" x={series.x} y={series.t} slot={5} showPoints />}
              {series && series.x.length > 0 && (
                <Curve name="VLoss" x={series.x} y={series.v} slot={6} showPoints dashed />
              )}
              {marker}
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
