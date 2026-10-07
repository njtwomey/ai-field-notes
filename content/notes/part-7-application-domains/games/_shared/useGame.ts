import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { isFree, move, type Pose, type World } from './world'

/** Walking speed in cells per second, doubled while Shift is held. */
const SPEED = 2.6
/** Turning speed in radians per second. */
const TURN = 2.2

const KEYS = new Set(['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 'a', 's', 'd', 'q', 'e', 'shift'])

/**
 * A game loop: the camera's pose, moved by the keyboard (arrows or WASD, Q and E to turn, Shift to run) while the
 * game area has focus, and a frame callback run on every animation frame while the game area is on screen. The pose
 * lives in a ref so moving never re-renders React; `frame` draws.
 */
export function useGame(world: World, frame: (pose: Pose, dt: number) => void) {
  const pose = useRef<Pose>(world.start)
  const keys = useRef(new Set<string>())
  const area = useRef<HTMLDivElement>(null)
  const frameRef = useRef(frame)
  useLayoutEffect(() => {
    frameRef.current = frame
  })
  const worldRef = useRef(world)
  const [focused, setFocused] = useState(false)

  useEffect(() => {
    worldRef.current = world
    pose.current = world.start
  }, [world])

  useEffect(() => {
    let raf = 0
    let last = performance.now()
    let visible = true
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting))
    if (area.current) io.observe(area.current)
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      if (visible) {
        const k = keys.current
        const on = (...names: string[]) => (names.some((n) => k.has(n)) ? 1 : 0)
        const forward = on('arrowup', 'w') - on('arrowdown', 's')
        const strafe = on('d') - on('a')
        const turn = on('arrowright', 'e') - on('arrowleft', 'q')
        if (forward || strafe || turn) {
          const v = SPEED * (k.has('shift') ? 2 : 1) * dt
          pose.current = move(worldRef.current, pose.current, forward * v, strafe * v, turn * TURN * dt)
        }
        frameRef.current(pose.current, dt)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      io.disconnect()
    }
  }, [])

  const onKeyDown = useCallback((e: KeyboardEvent) => {
    const k = e.key.toLowerCase()
    if (!KEYS.has(k)) return
    e.preventDefault()
    keys.current.add(k)
  }, [])
  const onKeyUp = useCallback((e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase()), [])

  /** Props for the focusable element that holds the game. */
  const areaProps = {
    ref: area,
    tabIndex: 0,
    onKeyDown,
    onKeyUp,
    onFocus: () => setFocused(true),
    onBlur: () => {
      keys.current.clear()
      setFocused(false)
    },
  }

  /**
   * Pointer handlers for the map canvas: pressing on an empty cell moves the camera there, and dragging turns it to
   * face the pointer. `scale` is the map's pixels per cell.
   */
  const pressing = useRef(false)
  const mapPointer = (scale: number) => {
    const at = (e: PointerEvent<HTMLCanvasElement>) => {
      const r = e.currentTarget.getBoundingClientRect()
      return { x: (e.clientX - r.left) / scale, y: (e.clientY - r.top) / scale }
    }
    return {
      onPointerDown: (e: PointerEvent<HTMLCanvasElement>) => {
        const p = at(e)
        if (isFree(worldRef.current, p.x, p.y)) pose.current = { ...pose.current, x: p.x, y: p.y }
        pressing.current = true
        e.currentTarget.setPointerCapture(e.pointerId)
        area.current?.focus({ preventScroll: true })
      },
      onPointerMove: (e: PointerEvent<HTMLCanvasElement>) => {
        if (!pressing.current) return
        const p = at(e)
        const dx = p.x - pose.current.x
        const dy = p.y - pose.current.y
        if (dx * dx + dy * dy > 0.09) pose.current = { ...pose.current, theta: Math.atan2(dy, dx) }
      },
      onPointerUp: () => (pressing.current = false),
    }
  }

  return { pose, areaProps, focused, mapPointer }
}
