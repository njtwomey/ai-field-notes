import { Play, RotateCcw, StepForward } from 'lucide-react'
import { ParamButton } from './controls'

/** Step / Run / Reset for iterative algorithms (k-means, EM, ...). Step and Run disable once `done`. */
export function StepControls({
  onStep,
  onRun,
  onReset,
  done,
}: {
  onStep: () => void
  onRun: () => void
  onReset: () => void
  done: boolean
}) {
  return (
    <div className="flex gap-2">
      <ParamButton onClick={onStep} disabled={done}>
        <StepForward /> Step
      </ParamButton>
      <ParamButton onClick={onRun} disabled={done}>
        <Play /> Run
      </ParamButton>
      <ParamButton onClick={onReset}>
        <RotateCcw /> Reset
      </ParamButton>
    </div>
  )
}
