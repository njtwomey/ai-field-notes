import { rng } from '@/lib/math'

/**
 * Cliff walking (Sutton and Barto, Example 6.6): a 12 × 4 grid, start at the bottom-left, goal at the bottom-right,
 * and the cells between them are a cliff. Every step pays −1; stepping into the cliff pays −100 and sends the agent
 * back to the start. Moves are deterministic and a move off the grid leaves the agent in place.
 */
export const WIDTH = 12
export const HEIGHT = 4
export const START = 0
export const GOAL = WIDTH - 1
export const ACTIONS: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]
const N_STATES = WIDTH * HEIGHT
/** Guards against an early episode wandering for ever; far above any learned episode's length. */
const MAX_STEPS = 2000

export const xy = (s: number): [number, number] => [s % WIDTH, Math.floor(s / WIDTH)]
export const isCliff = (s: number) => s > START && s < GOAL

function step(s: number, a: number): { next: number; reward: number } {
  const [x, y] = xy(s)
  const nx = Math.min(WIDTH - 1, Math.max(0, x + ACTIONS[a][0]))
  const ny = Math.min(HEIGHT - 1, Math.max(0, y + ACTIONS[a][1]))
  const t = ny * WIDTH + nx
  return isCliff(t) ? { next: START, reward: -100 } : { next: t, reward: -1 }
}

export type Method = 'q-learning' | 'sarsa'
export type Training = { returns: number[]; Q: number[][] }

/** One training run: `episodes` episodes of ε-greedy control with step size α and γ = 1. */
export function train(method: Method, episodes: number, eps: number, alpha: number, seed: number): Training {
  const r = rng(seed)
  const Q = Array.from({ length: N_STATES }, () => [0, 0, 0, 0])
  // ε-greedy with random tie-breaking, so an untrained agent does not always walk the same way.
  const act = (s: number) => {
    if (r.uniform() < eps) return Math.floor(r.uniform() * 4)
    const best = Math.max(...Q[s])
    const ties = [0, 1, 2, 3].filter((a) => Q[s][a] === best)
    return ties[Math.floor(r.uniform() * ties.length)]
  }
  const returns: number[] = []
  for (let e = 0; e < episodes; e++) {
    let s = START
    let a = act(s)
    let total = 0
    for (let t = 0; t < MAX_STEPS && s !== GOAL; t++) {
      const { next, reward } = step(s, a)
      total += reward
      const nextA = act(next)
      // SARSA bootstraps from the action it will actually take; Q-learning from the greedy action.
      const target = next === GOAL ? reward : reward + (method === 'sarsa' ? Q[next][nextA] : Math.max(...Q[next]))
      Q[s][a] += alpha * (target - Q[s][a])
      s = next
      a = nextA
    }
    returns.push(total)
  }
  return { returns, Q }
}

/** The path a greedy agent follows from the start under Q, stopping at the goal, a repeat or after 60 cells. */
export function greedyPath(Q: number[][]): number[] {
  const path = [START]
  const seen = new Set(path)
  let s = START
  while (s !== GOAL && path.length < 60) {
    const best = Math.max(...Q[s])
    const { next } = step(s, Q[s].indexOf(best))
    if (seen.has(next)) break
    path.push(next)
    seen.add(next)
    s = next
  }
  return path
}
