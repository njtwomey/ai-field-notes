import { useMemo, useState } from 'react'
import { countParams, treeLeaves } from 'aifn/foundation/pytree'
import { norm, type Tensor } from 'aifn/foundation/tensor'
import { Select } from '@lab/controls'
import { PanelSlot } from '@lab/layout'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@lab/ui/table'
import { Readout } from '@lab/viz'
import { formatValue } from './format'
import { registerView } from './registry'
import { TensorPanel } from './TensorView'

export type ParamsPanelProps = {
  /** A parameter pytree (nested objects and arrays of tensors), e.g. a layer's `init(stream)`. */
  params: unknown
  /** Gradients with the same structure, to list their norms beside the parameters. */
  grads?: unknown
}

const leafNorm = (v: Tensor | number) => (typeof v === 'number' ? Math.abs(v) : norm(v))

/**
 * A generic view of a parameter pytree (`aifn/nn`): one row per leaf with its path, shape, size, norm and (with
 * `grads`) gradient norm, and the chosen leaf drawn by `TensorPanel`.
 */
export function ParamsPanel({ params, grads }: ParamsPanelProps) {
  const leaves = useMemo(() => treeLeaves(params), [params])
  const gradNorms = useMemo(() => {
    if (grads === undefined) return null
    return new Map(treeLeaves(grads).map(({ path, value }) => [path, leafNorm(value)]))
  }, [grads])
  const [chosen, setChosen] = useState(leaves[0]?.path ?? '')
  const leaf = leaves.find((l) => l.path === chosen) ?? leaves[0]
  return (
    <>
      <PanelSlot slot="controls">
        <>
          {leaves.length > 0 && (
            <Select
              label="tensor"
              value={leaf.path}
              onChange={setChosen}
              options={leaves.map((l) => ({ value: l.path, label: l.path }))}
            />
          )}
        </>
      </PanelSlot>
      <PanelSlot slot="readouts">
        <Readout label="parameters" value={String(countParams(params))} />
        <Readout label="tensors" value={String(leaves.length)} />
      </PanelSlot>
      <div className="grid h-full gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="overflow-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>path</TableHead>
                <TableHead>shape</TableHead>
                <TableHead className="text-right">‖θ‖</TableHead>
                {gradNorms && <TableHead className="text-right">‖∇‖</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody>
              {leaves.map(({ path, value }) => (
                <TableRow
                  key={path}
                  className={path === leaf?.path ? 'bg-muted' : undefined}
                  onClick={() => setChosen(path)}
                >
                  <TableCell className="font-mono">{path}</TableCell>
                  <TableCell>{typeof value === 'number' ? 'scalar' : `[${value.shape.join(', ')}]`}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatValue(leafNorm(value))}</TableCell>
                  {gradNorms && (
                    <TableCell className="text-right tabular-nums">{formatValue(gradNorms.get(path) ?? NaN)}</TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="min-h-0">
          {leaf && typeof leaf.value !== 'number' && <TensorPanel tensor={leaf.value} name={leaf.path} />}
        </div>
      </div>
    </>
  )
}

// A parameter pytree has no kind of its own; it is drawn when asked for by key ('params/table').
registerView<unknown>({
  key: 'params/table',
  kind: 'params',
  description: 'A parameter pytree: one row per leaf with its path, shape and norm, and the chosen leaf drawn.',
  title: () => 'parameters',
  render: (p) => <ParamsPanel params={p} />,
})
