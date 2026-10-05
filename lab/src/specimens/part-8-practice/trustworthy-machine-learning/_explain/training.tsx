/**
 * Shared by the explanation pages: a small MLP trained in the worker (`aifn-methods/neural/full-batch`'s
 * `fullBatchComparison`, Adam on minibatches) on a dataset task, with a Train / Stop row and a player over its
 * checkpoints, and the trained network as an `aifn/learning/explain` `DenseNetwork`.
 */
import type { ReactNode } from 'react'
import { Player, StatusText } from 'aifn-render/controls'
import { ControlRow } from 'aifn-render/layout'
import { Button } from 'aifn-render/ui/button'
import type { useTrainedMlp } from './mlp'

/** The Train / Stop row with progress, status and (optionally) the checkpoint player. */
export function TrainRow({
  label = 'train',
  trained,
  stale,
  mlp,
  onTrain,
  note,
  player = true,
  extra,
}: {
  label?: string
  trained: boolean
  stale: boolean
  mlp: ReturnType<typeof useTrainedMlp>
  onTrain: () => void
  note?: string
  player?: boolean
  extra?: ReactNode
}) {
  const { run, checkpoints } = mlp
  const step = checkpoints.at(-1)?.iteration ?? 0
  return (
    <ControlRow label={label}>
      <div className="flex flex-wrap items-center gap-3">
        {run.running ? (
          <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
            Stop
          </Button>
        ) : (
          <Button size="sm" variant={!trained || stale ? 'default' : 'outline'} aria-label="Train" onClick={onTrain}>
            {trained ? 'Retrain' : 'Train'}
          </Button>
        )}
        <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
          <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, mlp.done)}%` }} />
        </div>
        <StatusText tone={trained && run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
          {!trained
            ? 'Not trained yet: press Train.'
            : run.error
              ? `failed: ${run.error}`
              : stale
                ? 'Settings changed since this model was trained: press Retrain.'
                : run.stopped
                  ? `stopped at step ${step}`
                  : `step ${step}${run.running ? '…' : ''}${Number.isFinite(mlp.accuracy) && !run.running ? ` · training accuracy ${(100 * mlp.accuracy).toFixed(1)}%` : ''}${note ? ` · ${note}` : ''}`}
        </StatusText>
        {extra}
      </div>
      {player && checkpoints.length > 0 && (
        <Player
          label="checkpoint"
          value={mlp.cpIndex}
          onChange={mlp.setCp}
          count={Math.max(1, checkpoints.length)}
          format={(p) => `step ${checkpoints[p]?.iteration ?? 0}`}
          startReason="the explanations are of the trained network; step back to explain earlier checkpoints"
        />
      )}
    </ControlRow>
  )
}
