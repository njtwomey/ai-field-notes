import { ChevronRight } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { cn } from 'aifn-render/lib/utils'
import { formatValue } from './format'

/**
 * A collapsible, typed tree of any plain value: numbers, strings, arrays, typed arrays, tensors (anything with
 * `shape`, `data` and `dtype`), streams (`key` and `uniform`), functions and nested objects. Each node shows its type
 * and size. Leaves that differ from `previous` at the same path are highlighted, so scrubbing a trace shows what a
 * step changed. Expansion is keyed by path and may be controlled, so it survives a change of `value`.
 */
export function StateTree({
  value,
  previous,
  name = 'state',
  expanded,
  onToggle,
  defaultDepth = 1,
}: {
  value: unknown
  /** The value to compare against; leaves that differ are highlighted. */
  previous?: unknown
  name?: string
  /** Controlled expansion: the set of expanded paths. */
  expanded?: ReadonlySet<string>
  onToggle?: (path: string) => void
  /** Uncontrolled: nodes shallower than this start expanded. */
  defaultDepth?: number
}) {
  const [own, setOwn] = useState<Set<string>>(() => new Set())
  const [collapsedOwn, setCollapsedOwn] = useState<Set<string>>(() => new Set())
  const controlled = expanded !== undefined
  const isOpen = (path: string, depth: number) =>
    controlled ? expanded.has(path) : own.has(path) || (depth < defaultDepth && !collapsedOwn.has(path))
  const toggle = (path: string, depth: number) => {
    if (controlled) return onToggle?.(path)
    const open = isOpen(path, depth)
    const flip = (s: Set<string>, on: boolean) => {
      const next = new Set(s)
      if (on) next.add(path)
      else next.delete(path)
      return next
    }
    if (depth < defaultDepth) setCollapsedOwn((s) => flip(s, open))
    else setOwn((s) => flip(s, !open))
  }
  return (
    <div className="font-mono text-xs leading-5">
      <Node
        label={name}
        value={value}
        previous={previous}
        hasPrevious={previous !== undefined}
        path={name}
        depth={0}
        isOpen={isOpen}
        toggle={toggle}
        ancestors={[]}
      />
    </div>
  )
}

type NodeProps = {
  label: string
  value: unknown
  previous: unknown
  hasPrevious: boolean
  path: string
  depth: number
  isOpen: (path: string, depth: number) => boolean
  toggle: (path: string, depth: number) => void
  ancestors: unknown[]
}

const PAGE = 100
const MAX_DEPTH = 16

type TensorLike = {
  shape: readonly number[]
  data: ArrayLike<number>
  dtype: string
  strides?: readonly number[]
  offset?: number
}
const isTypedArray = (v: unknown): v is ArrayLike<number> & { constructor: { name: string } } =>
  ArrayBuffer.isView(v) && !(v instanceof DataView)
const isTensor = (v: unknown): v is TensorLike =>
  typeof v === 'object' && v !== null && 'shape' in v && 'data' in v && 'dtype' in v
const isStream = (v: unknown): v is { key: string } =>
  typeof v === 'object' && v !== null && typeof (v as { uniform?: unknown }).uniform === 'function' && 'key' in v

/** Shallow difference for highlighting: numbers by value (NaN equals NaN), everything else by identity or JSON. */
function differs(a: unknown, b: unknown): boolean {
  if (typeof a === 'number' && typeof b === 'number') return !(a === b || (Number.isNaN(a) && Number.isNaN(b)))
  return a !== b
}

/** Whether anything under a node changed, looking at most `budget` leaves deep (for a marker on collapsed nodes). */
function subtreeChanged(a: unknown, b: unknown, budget = { left: 5000 }): boolean {
  if (budget.left-- <= 0) return false
  if (a === b) return false
  if (typeof a !== 'object' || a === null || typeof b !== 'object' || b === null) return differs(a, b)
  if (isTensor(a) && isTensor(b)) return subtreeChanged(a.data, b.data, budget)
  const ka = Object.keys(a)
  if (ka.length !== Object.keys(b).length) return true
  return ka.some((k) => subtreeChanged((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], budget))
}

function Leaf({
  label,
  children,
  type,
  changed,
}: {
  label: string
  children: ReactNode
  type: string
  changed?: boolean
}) {
  return (
    <div className="flex items-baseline gap-2 pl-4">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn('rounded px-0.5 tabular-nums', changed && 'bg-primary/15 text-foreground')}>{children}</span>
      <span className="text-[10px] text-muted-foreground/70">{type}</span>
    </div>
  )
}

function Node(props: NodeProps) {
  const { label, value, previous, hasPrevious, path, depth, isOpen, toggle, ancestors } = props
  const [limit, setLimit] = useState(PAGE)
  const changed = hasPrevious && differs(value, previous)

  if (value === null || value === undefined)
    return (
      <Leaf label={label} type={String(value)} changed={changed}>
        <span className="text-muted-foreground">{String(value)}</span>
      </Leaf>
    )
  if (typeof value === 'number')
    return (
      <Leaf label={label} type="number" changed={changed}>
        <span className={cn(!Number.isFinite(value) && 'text-destructive')}>{formatValue(value)}</span>
      </Leaf>
    )
  if (typeof value === 'boolean')
    return (
      <Leaf label={label} type="boolean" changed={changed}>
        {String(value)}
      </Leaf>
    )
  if (typeof value === 'string')
    return (
      <Leaf label={label} type="string" changed={changed}>
        &quot;{value}&quot;
      </Leaf>
    )
  if (typeof value === 'bigint')
    return (
      <Leaf label={label} type="bigint" changed={changed}>
        {String(value)}n
      </Leaf>
    )
  if (typeof value === 'function')
    return (
      <Leaf label={label} type="function">
        <span className="text-muted-foreground">ƒ {value.name || 'anonymous'}()</span>
      </Leaf>
    )
  if (isStream(value))
    return (
      <Leaf label={label} type="Stream" changed={hasPrevious && (previous as { key?: string })?.key !== value.key}>
        &quot;{value.key}&quot;
      </Leaf>
    )
  if (typeof value !== 'object')
    return (
      <Leaf label={label} type={typeof value}>
        {String(value)}
      </Leaf>
    )
  if (ancestors.includes(value) || depth > MAX_DEPTH)
    return (
      <Leaf label={label} type="circular">
        <span className="text-muted-foreground">…</span>
      </Leaf>
    )

  // Containers: work out the type label, an inline preview and the children.
  let type: string
  let preview: string | null = null
  let entries: [string, unknown, unknown][]
  const prev = previous as Record<string, unknown> | undefined
  if (isTensor(value)) {
    type = `Tensor ${value.dtype} [${value.shape.join(', ')}]`
    const flat = Array.from(value.data as ArrayLike<number>)
    preview = numericPreview(flat)
    const previousRows = isTensor(previous) ? tensorRows(previous) : undefined
    entries = tensorRows(value).map((row, i) => [value.shape.length > 1 ? `[${i}]` : String(i), row, previousRows?.[i]])
  } else if (isTypedArray(value)) {
    type = `${value.constructor.name}(${value.length})`
    preview = numericPreview(value)
    entries = Array.from(value, (v, i) => [String(i), v, (previous as ArrayLike<number> | undefined)?.[i]])
  } else if (Array.isArray(value)) {
    type = `Array(${value.length})`
    if (value.every((v) => typeof v === 'number')) preview = numericPreview(value as number[])
    entries = value.map((v, i) => [String(i), v, (previous as unknown[] | undefined)?.[i]])
  } else if (value instanceof Map) {
    type = `Map(${value.size})`
    entries = [...value.entries()].map(([k, v]) => [
      String(k),
      v,
      (previous as Map<unknown, unknown> | undefined)?.get?.(k),
    ])
  } else if (value instanceof Set) {
    type = `Set(${value.size})`
    entries = [...value].map((v, i) => [String(i), v, undefined])
  } else {
    const keys = Object.keys(value)
    const ctor = (value as object).constructor?.name
    type = ctor && ctor !== 'Object' ? `${ctor} {${keys.length}}` : `{${keys.length}}`
    entries = keys.map((k) => [k, (value as Record<string, unknown>)[k], prev?.[k]])
  }

  const open = isOpen(path, depth)
  const subtree = hasPrevious && !open && subtreeChanged(value, previous)
  const shown = entries.slice(0, limit)
  return (
    <div>
      <button
        type="button"
        onClick={() => toggle(path, depth)}
        className="flex w-full items-baseline gap-2 rounded text-left hover:bg-muted"
        aria-expanded={open}
      >
        <ChevronRight className={cn('size-3 shrink-0 self-center transition-transform', open && 'rotate-90')} />
        <span className="text-muted-foreground">{label}</span>
        <span className="text-[10px] text-muted-foreground/70">{type}</span>
        {!open && preview && <span className="truncate tabular-nums">{preview}</span>}
        {subtree && <span className="size-1.5 shrink-0 self-center rounded-full bg-primary" title="changed" />}
      </button>
      {open && (
        <div className="ml-1.5 border-l pl-1">
          {shown.map(([k, v, p]) => (
            <Node
              key={k}
              label={k}
              value={v}
              previous={p}
              hasPrevious={hasPrevious}
              path={`${path}.${k}`}
              depth={depth + 1}
              isOpen={isOpen}
              toggle={toggle}
              ancestors={[...ancestors, value]}
            />
          ))}
          {entries.length > limit && (
            <button
              type="button"
              onClick={() => setLimit((l) => l + PAGE * 5)}
              className="pl-4 text-muted-foreground underline-offset-2 hover:underline"
            >
              … {entries.length - limit} more
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function numericPreview(values: ArrayLike<number>): string {
  const n = values.length
  const head = Array.from({ length: Math.min(n, 6) }, (_, i) => formatValue(values[i]))
  return `[${head.join(', ')}${n > 6 ? ', …' : ''}]`
}

/** The rows of a tensor along its first axis, as plain nested arrays (read through strides). */
function tensorRows(t: TensorLike): unknown[] {
  const strides = t.strides ?? t.shape.map((_, d) => t.shape.slice(d + 1).reduce((a, b) => a * b, 1))
  const build = (d: number, offset: number): unknown =>
    d === t.shape.length
      ? t.data[offset]
      : Array.from({ length: t.shape[d] }, (_, i) => build(d + 1, offset + i * strides[d]))
  if (t.shape.length === 0) return [t.data[t.offset ?? 0]]
  return build(0, t.offset ?? 0) as unknown[]
}
