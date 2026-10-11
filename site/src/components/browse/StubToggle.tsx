import { Switch } from '@/components/ui/switch'
import { setHideStubs, stubCount, useHideStubs } from '@/lib/stubs'
import { cn } from '@/lib/utils'

/** A switch that hides placeholder notes everywhere notes are listed or counted. Renders nothing when there are none. */
export function StubToggle({ className }: { className?: string }) {
  const hide = useHideStubs()
  if (stubCount === 0) return null
  return (
    <label className={cn('inline-flex cursor-pointer items-center gap-2 text-xs text-muted-foreground', className)}>
      <Switch size="sm" checked={hide} onCheckedChange={setHideStubs} />
      <span>
        Hide stubs <span className="tabular-nums">({stubCount})</span>
      </span>
    </label>
  )
}
