/**
 * Train / Stop for any streamed training run in the compute worker, as `GymTrainer` does for the gym: nothing trains
 * until Train is pressed; the settings current at that press are the run's; a changed setting marks the shown run as
 * stale until Retrain. `useTrainedRun` (its own file) holds that state and streams the task; `TrainControls` draws
 * the button, a progress bar and the status line, with a page's extra buttons (presets) beside.
 */
import type { ReactNode } from 'react'
import { StatusText } from '@lab/controls'
import { ControlRow } from '@lab/layout'
import { Button } from '@lab/ui/button'
import type { TrainedRun } from './useTrainedRun'

export type TrainControlsProps = {
  run: TrainedRun<unknown, unknown>
  /** Fraction of the run done, 0 … 1. */
  progress: number
  /** Where the run is, e.g. "1200 / 2000 steps". */
  progressText: string
  /** Buttons beside Train (presets). */
  actions?: ReactNode
  label?: ReactNode
}

/** The Train / Retrain / Stop row with a progress bar and the run's status. */
export function TrainControls({ run: r, progress, progressText, actions, label = 'train' }: TrainControlsProps) {
  const { run, trained, stale } = r
  return (
    <ControlRow label={label}>
      <div className="flex flex-wrap items-center gap-3">
        {run.running ? (
          <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
            Stop
          </Button>
        ) : (
          <Button size="sm" variant={!trained || stale ? 'default' : 'outline'} aria-label="Train" onClick={r.train}>
            {trained ? 'Retrain' : 'Train'}
          </Button>
        )}
        {actions}
        <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
          <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, Math.max(0, progress))}%` }} />
        </div>
        <StatusText tone={run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
          {!trained
            ? 'Not trained yet: choose the settings, then press Train.'
            : run.error
              ? `failed: ${run.error}`
              : stale
                ? `Settings changed since this run (${progressText} shown): press Retrain to train with them.`
                : run.stopped
                  ? `stopped at ${progressText}`
                  : `${progressText}${run.running ? '…' : ''}`}
        </StatusText>
      </div>
    </ControlRow>
  )
}
