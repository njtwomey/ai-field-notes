import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Maximize,
  Minus,
  MoveHorizontal,
  MoveVertical,
  Plus,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Button } from '@lab/ui/button'
import { ButtonGroup } from '@lab/ui/button-group'
import { Popover, PopoverContent, PopoverTrigger } from '@lab/ui/popover'
import { cn } from '@lab/lib/utils'
import { formatNumber } from './format'
import { validRange, type Axis, type Range, type Viewport } from './viewport'

/** Zoom factor of one button press; pan moves a quarter of the span. */
const STEP = 2
const PAN = 0.25

/**
 * The zoom and pan toolbar, compact by default: one small button per axis (↔ for x, ↕ for y) that opens that axis's
 * full controls in a popover (its range as two typed fields, pan back, zoom out, zoom in, pan forward), then an
 * auto-scale button that fits both axes to the data. Pinch or Ctrl/⌘-scroll over the chart zooms as well (handled by the
 * chart).
 */
export function ViewportControls({
  viewport,
  log = {},
  axes = ['x', 'y'],
  bothAxes = true,
  note,
  className,
}: {
  viewport: Viewport
  log?: { x?: boolean; y?: boolean }
  axes?: Axis[]
  /** The auto-scale button; off in Subplots panels, whose grid has one "fit all". */
  bothAxes?: boolean
  /** A short note at the start of the toolbar, e.g. "units not equal". */
  note?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex min-h-6 items-center justify-end gap-2', className)}>
      {note && <span className="mr-auto text-[11px] text-muted-foreground">{note}</span>}
      <ButtonGroup aria-label="Axes">
        {axes.map((axis) => (
          <AxisPopover key={axis} axis={axis} viewport={viewport} log={!!log[axis]} />
        ))}
        {bothAxes && (
          <IconButton label="Auto-scale: fit both axes to the data" onClick={viewport.reset}>
            <Maximize />
          </IconButton>
        )}
      </ButtonGroup>
    </div>
  )
}

function IconButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  children: ReactNode
}) {
  return (
    <Button variant="outline" size="icon-xs" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      {children}
    </Button>
  )
}

/** One axis's button; its full controls open in a popover beside it. */
function AxisPopover({ axis, viewport, log }: { axis: Axis; viewport: Viewport; log: boolean }) {
  const Icon = axis === 'x' ? MoveHorizontal : MoveVertical
  const label = `${axis} axis: range, pan and zoom`
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="outline" size="icon-xs" aria-label={label} title={label} />}>
        <Icon />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-auto p-2">
        <AxisControls axis={axis} viewport={viewport} log={log} />
      </PopoverContent>
    </Popover>
  )
}

function AxisControls({ axis, viewport, log }: { axis: Axis; viewport: Viewport; log: boolean }) {
  const range = viewport.shown[axis]
  const [Back, Forward] = axis === 'x' ? [ArrowLeft, ArrowRight] : [ArrowDown, ArrowUp]
  const [back, forward] = axis === 'x' ? ['left', 'right'] : ['down', 'up']
  return (
    <span className="flex items-center gap-1.5">
      <RangeFields axis={axis} range={range} log={log} onChange={(r) => viewport.set(axis, r)} />
      <ButtonGroup aria-label={`${axis} axis`}>
        <IconButton label={`Pan ${back}`} onClick={() => viewport.pan(axis, -PAN)} disabled={!range}>
          <Back />
        </IconButton>
        <IconButton label={`Zoom ${axis} out`} onClick={() => viewport.zoom(STEP, [axis])} disabled={!range}>
          <Minus />
        </IconButton>
        <IconButton label={`Zoom ${axis} in`} onClick={() => viewport.zoom(1 / STEP, [axis])} disabled={!range}>
          <Plus />
        </IconButton>
        <IconButton label={`Pan ${forward}`} onClick={() => viewport.pan(axis, PAN)} disabled={!range}>
          <Forward />
        </IconButton>
      </ButtonGroup>
    </span>
  )
}

/** Two typed fields for an axis range. Enter or blur commits both if they form a valid range; Escape reverts. */
function RangeFields({
  axis,
  range,
  log,
  onChange,
}: {
  axis: Axis
  range: Range | undefined
  log: boolean
  onChange: (r: Range) => void
}) {
  const [draft, setDraft] = useState<[string, string] | null>(null)
  const shown: [string, string] = draft ?? (range ? [formatNumber(range[0]), formatNumber(range[1])] : ['', ''])
  const commit = () => {
    if (!draft) return
    const next: Range = [Number(draft[0].replace('−', '-')), Number(draft[1].replace('−', '-'))]
    setDraft(null)
    if (validRange(next, log)) onChange(next)
  }
  const field = (i: 0 | 1) => (
    <input
      aria-label={`${axis} ${i === 0 ? 'minimum' : 'maximum'}`}
      className="h-6 w-16 rounded-md border border-input bg-transparent px-1.5 text-right font-mono text-[11px] tabular-nums outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
      value={shown[i]}
      disabled={!range}
      onFocus={(e) => {
        setDraft(shown)
        e.currentTarget.select()
      }}
      onChange={(e) => setDraft(i === 0 ? [e.target.value, shown[1]] : [shown[0], e.target.value])}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur()
        else if (e.key === 'Escape') {
          setDraft(null)
          // Blur after the reset so the blur's commit sees no draft.
          requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
        }
      }}
    />
  )
  return (
    <span className="flex items-center gap-1 font-mono text-[11px] text-muted-foreground">
      {axis}
      {field(0)}
      <span aria-hidden>–</span>
      {field(1)}
    </span>
  )
}
