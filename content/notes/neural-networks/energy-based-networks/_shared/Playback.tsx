import { Pause, Play } from 'lucide-react'
import { ParamButton } from '@/components/viz'

export function PlayButton({
  playing,
  onToggle,
  disabled,
}: {
  playing: boolean
  onToggle: () => void
  disabled?: boolean
}) {
  return (
    <ParamButton onClick={onToggle} disabled={disabled}>
      {playing ? <Pause /> : <Play />} {playing ? 'Pause' : 'Play'}
    </ParamButton>
  )
}
