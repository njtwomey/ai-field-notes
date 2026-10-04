/**
 * Drawing Huffman's forest: where each node of a `HuffmanState` sits, and a tween that moves nodes smoothly from one
 * step's places to the next. Presentation only: the forest itself (roots, children, queue order) is read from the state.
 *
 * Layout: the roots, in queue order (pop order, left to right), sit on one row, the queue row; each root's subtree
 * hangs below it with its leaves one slot apart in digit order and every parent centred over its children. At step 0
 * the row is the K leaves sorted by weight; a merge pulls the popped subtrees down one level under their new parent,
 * which takes the place in the row where the queue inserted it; at the end the row holds the root of the whole tree.
 * Every leaf takes the same width at every step, so subtrees slide rather than resize.
 */
import type { HuffmanState } from 'aifn-methods/information/coding'
import { useEffect, useRef, useState } from 'react'

export type Point = { x: number; y: number }

/** Grid units between neighbouring leaves, between subtrees in the queue row, and between levels. */
export const SLOT = 1.2
export const GAP = 0.8
export const LEVEL = 1.45

/** Each node's centre at this state (nodes not yet created are absent), centred on x = 0. */
export function forestLayout(state: HuffmanState): Map<number, Point> {
  const pos = new Map<number, Point>()
  let cursor = 0
  const place = (v: number, depth: number) => {
    const node = state.nodes[v]
    if (node.children.length === 0) {
      pos.set(v, { x: cursor, y: depth * LEVEL })
      cursor += SLOT
      return
    }
    for (const c of node.children) place(c, depth + 1)
    const first = pos.get(node.children[0])!
    const last = pos.get(node.children[node.children.length - 1])!
    pos.set(v, { x: (first.x + last.x) / 2, y: depth * LEVEL })
  }
  state.queue.forEach((root, i) => {
    if (i > 0) cursor += GAP
    place(root, 0)
  })
  const shift = (cursor - SLOT) / 2
  for (const p of pos.values()) p.x -= shift
  return pos
}

const ease = (a: number) => (a < 0.5 ? 2 * a * a : 1 - (-2 * a + 2) ** 2 / 2)

/**
 * Positions eased from what is on screen to `target` over `ms` milliseconds whenever `target` changes; a change of
 * `key` (a new run) jumps instead. Nodes new to `target` start at their place. Honours reduced motion.
 */
export function useTweened(target: Map<number, Point>, key: unknown, ms = 420): Map<number, Point> {
  const [shown, setShown] = useState(target)
  const current = useRef(target)
  const lastKey = useRef(key)
  useEffect(() => {
    const reduced =
      typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
    if (lastKey.current !== key || reduced) {
      lastKey.current = key
      current.current = target
      setShown(target)
      return
    }
    const from = current.current
    const start = performance.now()
    let frame = 0
    const tick = (now: number) => {
      const a = Math.min(1, (now - start) / ms)
      const e = ease(a)
      const next = new Map<number, Point>()
      for (const [id, p] of target) {
        const f = from.get(id) ?? p
        next.set(id, { x: f.x + (p.x - f.x) * e, y: f.y + (p.y - f.y) * e })
      }
      current.current = next
      setShown(next)
      if (a < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, key, ms])
  return shown
}
