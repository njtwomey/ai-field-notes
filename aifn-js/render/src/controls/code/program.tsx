/** The components of a program on a page: its entry function's controls and its status line (hooks in `useProgram`). */
import { useEffect, useMemo } from 'react'
import type { Space } from 'aifn/foundation/space'
import { fromSpace } from '@render/state/schema'
import { StatusText } from '../base/StatusText'
import { ParamControls } from '../schema/ParamControls'
import { useParams } from '../schema/variants'
import type { ArgValues, ProgramRun } from './useProgram'

/** Controls for an entry function's parameters, from its signature's `Space`; reports their values. */
export function EntryControls({
  space,
  initial,
  onChange,
}: {
  space: Space
  initial: ArgValues
  onChange: (values: ArgValues) => void
}) {
  const defs = useMemo(() => fromSpace(space), [space])
  const p = useParams(defs, initial)
  useEffect(() => onChange(p.values as ArgValues), [p.values, onChange])
  return <ParamControls {...p} />
}

/**
 * One status line for a run: its error in the destructive tone with the line and column, else its time; `problem`
 * reports the page's own complaint about the value (a wrong shape). Printed output follows.
 */
export function ProgramStatus({ run, problem }: { run: ProgramRun; problem?: string | null }) {
  const { result } = run
  const error = result && !result.ok ? result.error : null
  const where = error?.line !== undefined ? ` (line ${error.line}, column ${error.column})` : ''
  const message = run.failure ?? (error ? `${error.name}: ${error.message}${where}` : problem)
  return (
    <div className="flex min-h-6 flex-col gap-1 font-mono text-xs" aria-live="polite">
      {message ? (
        <StatusText tone="error">{message}</StatusText>
      ) : (
        <StatusText>
          {result?.ok ? `ran in ${result.ms.toFixed(1)} ms` : 'running…'}
          {run.running && result ? ' · running…' : ''}
        </StatusText>
      )}
      {result && result.output.length > 0 && (
        <pre className="max-h-24 overflow-auto whitespace-pre-wrap text-muted-foreground">
          {result.output.join('\n')}
        </pre>
      )}
    </div>
  )
}
