/**
 * `gymSetup`: a `GymTrainer` setup from registry keys. The environment and agent are built on the page from
 * `environmentRegistry` and `agentRegistry` (for replay and evaluation), and the same two become worker tasks at their
 * registry addresses (`<module>/<key>`), so a page names each once instead of pairing a factory with an address.
 */
import { agentRegistry, environmentRegistry } from 'aifn-methods/gym'
import type { Agent, Environment } from 'aifn/foundation/contracts'
import { call } from '@lab/state'
import type { GymSetup } from './GymTrainer'

type AnyEnv = Environment<unknown, unknown, unknown>
type AnyAgent = Agent<unknown, unknown, unknown>

/** The setup for environment `envKey` and agent `agentKey` of the gym registries, built from their parameters. */
export function gymSetup(
  envKey: string,
  envParams: object,
  agentKey: string,
  agentParams: object,
  extra: Pick<GymSetup, 'evaluationEnv'> = {},
): GymSetup {
  const e = environmentRegistry[envKey]
  const a = agentRegistry[agentKey]
  if (!e) throw new Error(`gymSetup: no environment '${envKey}' in the gym registry`)
  if (!a) throw new Error(`gymSetup: no agent '${agentKey}' in the gym registry`)
  return {
    env: (e as unknown as (p: object) => AnyEnv)(envParams),
    agent: (a as unknown as (p: object) => AnyAgent)(agentParams),
    envTask: call(`${e.info.module}/${envKey}`, envParams),
    agentTask: call(`${a.info.module}/${agentKey}`, agentParams),
    ...extra,
  }
}

/** An environment of the gym registry built on the page (an evaluation variant, say). */
export function gymEnvironment(envKey: string, envParams: object): AnyEnv {
  const e = environmentRegistry[envKey]
  if (!e) throw new Error(`gymEnvironment: no environment '${envKey}' in the gym registry`)
  return (e as unknown as (p: object) => AnyEnv)(envParams)
}
