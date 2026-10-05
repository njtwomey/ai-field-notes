import { useMemo, useState } from 'react'
import { isContiguous, isTensor, size, slice, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { Select, Slider } from 'aifn-render/controls'
import { PanelSlot } from 'aifn-render/layout'
import { Table, TableBody, TableCell, TableRow } from 'aifn-render/ui/table'
import { Bars, Curve, formatNumber, Plot, Raster, useAxis } from 'aifn-render/viz'
import { formatValue } from './format'
import { registerKind, registerView } from './registry'

export type TensorMode = 'chart' | 'table'

/** The view choice, for a `Select` in a figure's controls. */
export const TENSOR_MODES = (rank: number) =>
  [
    { value: 'chart', label: rank === 1 ? 'plot' : 'heatmap' },
    { value: 'table', label: 'table' },
  ] as const

/** Most cells a table shows before truncating. */
const TABLE_LIMIT = 400

/** A heatmap of a matrix with row 0 at the top: y is minus the row index so that ECharts' upward axis reads down. */
function MatrixHeatmap({ rows }: { rows: number[][] }) {
  const { x, y, z, range, signed } = useMemo(() => {
    const m = rows.length
    const n = rows[0]?.length ?? 0
    let lo = Infinity
    let hi = -Infinity
    for (const r of rows) for (const v of r) if (Number.isFinite(v)) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)]
    const signed = lo < 0 && hi > 0
    const bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1
    return {
      x: Array.from({ length: n }, (_, j) => j),
      y: Array.from({ length: m }, (_, i) => i - (m - 1)),
      z: Array.from({ length: m }, (_, i) => rows[m - 1 - i]),
      range: (signed ? [-bound, bound] : [lo === Infinity ? 0 : lo, hi === -Infinity ? 1 : hi]) as [number, number],
      signed,
    }
  }, [rows])
  const xAxis = useAxis({ label: 'column' })
  // Rows run down the page: the axis shows the row index, the negated y.
  const yAxis = useAxis({ label: 'row', format: flipped })
  return (
    <Plot x={xAxis} y={yAxis}>
      <Raster x={x} y={y} z={z} range={range} scale={signed ? 'diverging' : 'sequential'} valueLabel="value" />
    </Plot>
  )
}

const flipped = (v: number) => formatNumber(v === 0 ? 0 : -v)

/** A vector against its index: bars for up to 32 values, else a line. */
function VectorChart({ values, name }: { values: number[]; name: string }) {
  const xs = useMemo(() => values.map((_, i) => i), [values])
  const xAxis = useAxis({ label: 'index' })
  const yAxis = useAxis({ label: 'value' })
  return (
    <Plot x={xAxis} y={yAxis} legend={false}>
      {values.length <= 32 ? <Bars name={name} x={xs} y={values} /> : <Curve name={name} x={xs} y={values} />}
    </Plot>
  )
}

function ValueTable({ rows }: { rows: number[][] }) {
  const cells = rows.length * (rows[0]?.length ?? 0)
  const shown =
    cells > TABLE_LIMIT ? rows.slice(0, Math.max(1, Math.floor(TABLE_LIMIT / (rows[0]?.length || 1)))) : rows
  return (
    <div className="space-y-1 overflow-auto">
      <Table className="font-mono text-xs">
        <TableBody>
          {shown.map((r, i) => (
            <TableRow key={i}>
              {r.map((v, j) => (
                <TableCell key={j} className="px-2 py-1 text-right tabular-nums">
                  {formatValue(v)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {shown.length < rows.length && (
        <p className="text-xs text-muted-foreground">
          Showing {shown.length} of {rows.length} rows.
        </p>
      )}
    </div>
  )
}

/** Shape, dtype, size and layout of a tensor, in one muted line. */
export function TensorMeta({ tensor, name }: { tensor: Tensor; name?: string }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
      {name && <span className="font-medium text-foreground">{name}</span>}
      <span>shape [{tensor.shape.join(', ')}]</span>
      <span>{tensor.dtype}</span>
      <span>{size(tensor)} elements</span>
      <span>{isContiguous(tensor) ? 'contiguous' : `strided (${tensor.strides.join(', ')})`}</span>
    </div>
  )
}

export type TensorPanelProps = {
  tensor: Tensor
  /** A name shown above the panel, e.g. "L". */
  name?: string
  mode?: TensorMode
}

/**
 * The body of a tensor view, for use inside a `Figure` (several side by side, e.g. a matrix and its factors): a number
 * for rank 0, a line (or bars, for up to 32 values) for rank 1, a heatmap for rank 2 (diverging when values change
 * sign), and a heatmap of the last two axes with a slider over the leading ones for higher ranks; or a table. The chart
 * takes the frame's height.
 */
export function TensorPanel({ tensor, name, mode = 'chart' }: TensorPanelProps) {
  const [lead, setLead] = useState(0)
  const rank = tensor.shape.length
  const leading = rank > 2 ? tensor.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1
  const index = Math.min(lead, leading - 1)
  const rows = useMemo(() => {
    if (rank <= 1) return [toFlat(tensor)]
    if (rank === 2) return toRows(tensor)
    // Walk the leading axes in row-major order to the `index`-th matrix.
    const specs: number[] = []
    let rest = index
    for (let k = rank - 3; k >= 0; k--) {
      specs.unshift(rest % tensor.shape[k])
      rest = Math.floor(rest / tensor.shape[k])
    }
    return toRows(slice(tensor, ...specs))
  }, [tensor, rank, index])
  const values = rows[0]

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <TensorMeta tensor={tensor} name={name} />
      {rank > 2 && (
        <Slider
          className="max-w-sm"
          label={`matrix of leading axes [${tensor.shape.slice(0, -2).join(', ')}]`}
          value={index}
          onChange={setLead}
          min={0}
          max={leading - 1}
          step={1}
        />
      )}
      {rank === 0 ? (
        <div className="font-mono text-2xl">{formatValue(values[0])}</div>
      ) : mode === 'table' ? (
        <ValueTable rows={rows} />
      ) : rank === 1 ? (
        <VectorChart values={values} name={name ?? 'value'} />
      ) : (
        <MatrixHeatmap rows={rows} />
      )}
    </div>
  )
}

export type TensorModePanelProps = {
  tensor: Tensor
  /** Start with the table instead of the chart. */
  initialMode?: TensorMode
}

/** A `TensorPanel` with its chart / table choice among the enclosing figure's controls. */
export function TensorModePanel({ tensor, initialMode = 'chart' }: TensorModePanelProps) {
  const [mode, setMode] = useState<TensorMode>(initialMode)
  const rank = tensor.shape.length
  return (
    <>
      {rank > 0 && (
        <PanelSlot slot="controls">
          <Select label="view" value={mode} onChange={setMode} options={TENSOR_MODES(rank)} />
        </PanelSlot>
      )}
      <TensorPanel tensor={tensor} mode={mode} />
    </>
  )
}

registerKind('tensor', isTensor)
registerView<Tensor>({
  key: 'tensor/values',
  kind: 'tensor',
  description: 'A tensor as a plot (rank 1), a heatmap (rank 2, leading axes on a slider) or a table of values.',
  title: (t) => `tensor [${t.shape.join(', ')}]`,
  render: (t) => <TensorModePanel tensor={t} />,
})
