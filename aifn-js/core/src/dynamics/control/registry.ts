/**
 * The functions of `aifn/dynamics/control`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as ackermann from './ackermann'
import * as lqr from './lqr'

const fn = definer<FunctionInfo>('function', 'dynamics/control')
const LQR = ['linear-quadratic-regulator', 'linear-quadratic-gaussian-control']

fn(
  {
    key: 'lqr',
    name: 'Continuous-time LQR',
    summary: 'The optimal state-feedback gain from the continuous algebraic Riccati equation.',
    role: 'solver',
    notes: LQR,
    cite: ['kalman1960'],
  },
  lqr.lqr,
)
fn(
  {
    key: 'dlqr',
    name: 'Discrete-time LQR',
    summary: 'The optimal state-feedback gain from the discrete algebraic Riccati equation.',
    role: 'solver',
    notes: LQR,
  },
  lqr.dlqr,
)
fn(
  {
    key: 'closedLoopPoles',
    name: 'Closed-loop poles',
    tex: '\\operatorname{eig}(A - BK)',
    role: 'property',
    notes: ['pole-placement', ...LQR],
  },
  lqr.closedLoopPoles,
)
fn(
  {
    key: 'ackermann',
    name: "Ackermann's formula",
    summary: 'The unique single-input gain placing the poles at chosen locations.',
    role: 'solver',
    notes: ['pole-placement'],
    cite: ['ackermann1972'],
  },
  ackermann.ackermann,
)

/** The functions of the module, keyed by name. */
export const controlFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', lqr, ackermann) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
