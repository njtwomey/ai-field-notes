import { Pause, Play } from 'lucide-react'
import { Button } from 'aifn-render'

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
    <Button variant="outline" size="sm" onClick={onToggle} disabled={disabled}>
      {playing ? <Pause /> : <Play />} {playing ? 'Pause' : 'Play'}
    </Button>
  )
}
