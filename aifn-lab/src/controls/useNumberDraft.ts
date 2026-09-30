import { useState, type FocusEvent, type KeyboardEvent } from 'react'
import { formatNumber } from '@lab/viz'

/**
 * The editing behaviour of a typed number: while focused the field holds a draft; Enter or blur commits it (the owner
 * clamps and snaps); Escape reverts; ↑ and ↓ step by `step` (× 10 with Shift). Accepts "−" and exponent notation.
 */
export function useNumberDraft({
  value,
  onCommit,
  step,
  format = formatNumber,
}: {
  value: number
  onCommit: (v: number) => void
  step: number
  format?: (v: number) => string
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const commit = (text: string | null) => {
    setDraft(null)
    if (text === null || text.trim() === '') return
    const n = Number(text.trim().replace(/^−/, '-'))
    if (Number.isFinite(n)) onCommit(n)
  }
  return {
    value: draft ?? format(value),
    inputMode: 'decimal' as const,
    onFocus: (e: FocusEvent<HTMLInputElement>) => {
      setDraft(String(value))
      e.currentTarget.select()
    },
    onChange: (e: { target: { value: string } }) => setDraft(e.target.value),
    onBlur: () => commit(draft),
    onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') e.currentTarget.blur()
      else if (e.key === 'Escape') {
        setDraft(null)
        const el = e.currentTarget
        // Blur once the reset has rendered, so the blur's commit sees no draft.
        requestAnimationFrame(() => el.blur())
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault()
        const base = draft !== null && Number.isFinite(Number(draft)) ? Number(draft) : value
        const next = base + (e.key === 'ArrowUp' ? 1 : -1) * step * (e.shiftKey ? 10 : 1)
        onCommit(next)
        setDraft(null)
      }
    },
  }
}
