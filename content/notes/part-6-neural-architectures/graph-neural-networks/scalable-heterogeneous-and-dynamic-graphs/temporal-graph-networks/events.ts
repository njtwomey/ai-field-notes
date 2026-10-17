// The note's event stream: ten timed interactions between five users and three items, each with a rating. The first
// four events are the worked example.
import type { TimedEvent } from 'aifn-methods/neural/graph'

export const USERS = 5
export const ITEMS = 3
export const NODES = USERS + ITEMS
export const user = (u: number) => u - 1
export const item = (i: number) => USERS + i - 1
export const NAMES = [
  ...Array.from({ length: USERS }, (_, k) => `u${k + 1}`),
  ...Array.from({ length: ITEMS }, (_, k) => `i${k + 1}`),
]

const RAW: [number, number, number, number][] = [
  // user, item, time, rating
  [1, 1, 1, 4],
  [2, 2, 2, 2],
  [2, 1, 3, 0],
  [3, 2, 5, 1],
  [4, 3, 6, 3],
  [1, 3, 8, 5],
  [5, 1, 9, 2],
  [4, 2, 11, 4],
  [3, 3, 12, 1],
  [5, 2, 14, 3],
]

export const EVENTS: TimedEvent[] = RAW.map(([u, i, time, rating]) => ({
  source: user(u),
  destination: item(i),
  time,
  feature: [rating],
}))

/** Users in a column on the left, items on the right. */
export const POSITIONS: [number, number][] = [
  ...Array.from({ length: USERS }, (_, k): [number, number] => [0, 4 - k]),
  ...Array.from({ length: ITEMS }, (_, k): [number, number] => [3, 3.5 - 1.5 * k]),
]

/** The worked example's update: the average of the node's memory and the other endpoint's memory plus the rating. */
export const averageUpdate = (self: readonly number[], other: readonly number[], feature: readonly number[]) => [
  0.5 * (self[0] + other[0] + feature[0]),
]
