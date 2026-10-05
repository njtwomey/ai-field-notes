import { seriesColor } from 'aifn-render/design'
import { useTheme } from 'aifn-render/design'
import { cn } from 'aifn-render/lib/utils'

/**
 * Class chips for the multi-panel data views, whose panels draw no legend of their own: each chip shows the class's
 * colour and count, and a click hides or shows the class in every panel at once. Colours are the categorical slots in
 * class order, as grouped `Points` layers use them.
 */
export function ClassLegend({
  names,
  counts,
  hidden,
  onToggle,
}: {
  names: readonly string[]
  counts?: readonly number[]
  hidden: ReadonlySet<number>
  onToggle: (c: number) => void
}) {
  const { resolved: mode } = useTheme()
  return (
    <div className="col-span-full flex flex-wrap items-center gap-1.5" role="group" aria-label="classes">
      {names.map((name, c) => {
        const off = hidden.has(c)
        return (
          <button
            key={c}
            type="button"
            aria-pressed={!off}
            onClick={() => onToggle(c)}
            title={off ? `Show ${name}` : `Hide ${name}`}
            className={cn(
              'inline-flex h-6 items-center gap-1.5 rounded-md border px-2 text-xs transition-opacity hover:bg-muted',
              off && 'opacity-45',
            )}
          >
            <span
              className="size-2.5 rounded-full"
              style={{
                background: off ? 'transparent' : seriesColor(mode, c),
                border: `1.5px solid ${seriesColor(mode, c)}`,
              }}
            />
            <span className={cn(off && 'line-through')}>{name}</span>
            {counts && <span className="text-muted-foreground tabular-nums">{counts[c]}</span>}
          </button>
        )
      })}
    </div>
  )
}
