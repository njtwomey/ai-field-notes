# aifn environments: one protocol for bandits and reinforcement learning

Status: **lean first build done (2026-10-01)**: contracts, `rollout`/`episodes`, the maze, a random and a Q-learning
agent, one lab page. §3a lists where the build departs from this proposal; §9a lists what is left.

## 1. Why

Today aifn has two unrelated shapes and no shared loop.

| Today                                   | Shape                                                                     | Consumers                                  |
| --------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------ |
| `BanditEnvironment` (decisions/bandits) | `context(s)`, `rewards(s, ctx)` = every arm's reward, `means(ctx)` oracle | 12 bandit policies through `banditRun`     |
| `TabularMdp` (decisions/rl)             | explicit `outcomes[s·A + a]` table, `start`, `terminal`, `gamma`, grid    | value/policy iteration, 7 tabular learners |
| `EnvironmentInfo` (registry)            | `family: 'bandit' \| 'mdp'`, `params: Space`                              | the catalog                                |

Each learner samples the MDP's table itself (`sampleOutcome`), and each bandit run sees every arm's reward. A new
environment needs new simulation code per agent family, and a new agent works on one family only.

The goal is the Gym idea in aifn's style: **one environment protocol, one agent protocol, one rollout**, so that N
environments and M agents cost N + M, every rollout is a playable `Trace`, and the lab gets one generic episode view.

## 2. Principles

- **Pure, like every aifn Algorithm.** Environment state is plain data; randomness is an explicit `Stream`. No hidden
  RNG, no mutation. A rollout is reproducible, cloneable, seekable (`trace`/`seek`/`extend`), and two agents can face
  identical randomness (common random numbers, as `banditRun` does today).
- **Gymnasium's semantics, not its API.** `reset`/`step`; `terminated` (a true end: do not bootstrap) kept apart from
  `truncated` (a time limit: do bootstrap). Observation and action domains are declared.
- **Capabilities, not subclasses.** What an environment can do beyond `step` is optional and declared: an explicit
  model for planners, an oracle for regret, a renderer for the lab. As with models' `decide`/`predictive`.
- **One definition.** Domains reuse the distributions' `Support`; agents reuse `Algorithm`; spaces of parameters reuse
  `Space`; registration reuses `define`.

## 3. The contracts (core, `foundation/contracts/environment.ts`)

The types live in core contracts (criterion C6: every environment and agent must agree on them). Implementations live
in applications.

```ts
/** A set of values: a support (from distributions) and a shape. `Discrete(n)` and `Box(lo, hi, shape)` in Gym terms. */
export type Domain =
  | { kind: 'values'; support: Support; shape: Shape } //  {type:'integers', lower:0, upper:n-1}, [] — or an interval box
  | { kind: 'record'; fields: Readonly<Record<string, Domain>> } // structured observations (a grid cell and a context)

export interface Step<S, O> {
  state: S
  observation: O
  reward: Scalar
  /** A terminal state: the return ends here, values do not bootstrap past it. */
  terminated: boolean
  /** Cut short (time limit, budget): the episode stops but values still bootstrap. */
  truncated: boolean
}

/** An environment. `S` is its hidden state (plain data), `O` what the agent sees, `A` an action. */
export interface Environment<S, O, A> {
  readonly name: string
  readonly observation: Domain
  readonly action: Domain
  /** Discount the problem is posed with (agents may use their own). */
  readonly gamma: Scalar
  /** Episode length cap that sets `truncated` (Infinity for none). */
  readonly horizon: number
  reset(stream: Stream): { state: S; observation: O }
  step(state: S, action: A, stream: Stream): Step<S, O>
  /** Actions allowed in a state (masking), when not every action of `action` is. */
  legal?(state: S): readonly A[]

  // ── optional capabilities ──
  /** The explicit dynamics, for planners: a tabular MDP view or a differentiable transition. */
  readonly model?: EnvironmentModel<S, A>
  /** Ground truth for evaluation: expected rewards, optimal values, regret. Never shown to agents. */
  readonly oracle?: EnvironmentOracle<S, A>
  /** How to draw a state (grid layout, pendulum angle, cart position), for the lab. */
  readonly render?: RenderSpec<S>
}

export type EnvironmentModel<S, A> =
  | {
      kind: 'tabular'
      states: number
      actions: number
      outcomes: readonly Outcome[][]
      terminal: Uint8Array
      encode(s: S): number
      decode(i: number): S
    } // value iteration, policy iteration, exact evaluation
  | { kind: 'dynamics'; transition(s: S, a: A): S; reward(s: S, a: A): Scalar } // deterministic, traceable: MPC, iLQR

export interface EnvironmentOracle<S, A> {
  /** E[reward | state, action] (a bandit's arm means; per round for a contextual bandit). */
  expectedReward?(state: S, action: A): Scalar
  /** The best achievable expected reward in a state (for pseudo-regret). */
  bestExpectedReward?(state: S): Scalar
  /** Optimal state values, when known in closed form or by exact planning. */
  optimalValues?(): Tensor
}
```

The agent is an `Algorithm` whose state carries what it has learnt, plus two hooks the rollout calls:

```ts
/** A learning agent. Its state is plain data; `act` and `learn` are pure. */
export interface Agent<S_agent, O, A> {
  readonly name: string
  init(env: EnvironmentShape<O, A>, stream: Stream): S_agent
  /** Choose an action; returns the action and what the lab may show (scores, probabilities). */
  act(
    agent: S_agent,
    observation: O,
    stream: Stream,
  ): { action: A; scores?: Float64Array; probabilities?: Float64Array }
  /** Update from one transition. Episode-level learners (Monte Carlo, REINFORCE) buffer here and learn at the end. */
  learn(agent: S_agent, t: Transition<O, A>): S_agent
}

export interface Transition<O, A> {
  observation: O
  action: A
  reward: Scalar
  next: O
  terminated: boolean
  truncated: boolean
}
```

### 3a. As built (departures from the sketch above)

- **`Domain`** is `{ kind: 'discrete'; n; names? }` only (owner's direction): agents read its size and names, the
  random agent draws from it, and the registry pairs agents and environments by `DomainKind`. A continuous
  `{ kind: 'box', … }` joins the union for the pendulum; `Support` and `record` domains are dropped for now. Helpers
  in `aifn/foundation/space`: `discreteDomain`, `domainContains`, `domainSize`, `sampleDomain(stream, domain)`.
- **`EnvironmentShape`** is a named interface (name, domains, `gamma`, `horizon`) that `Environment` extends and
  `Agent.init` receives. `act` returns a `Decision<A>` (`Choice` was taken by the bandits).
- **`horizon` is enforced by the rollout** (Gymnasium's `TimeLimit`): `env.step` sets `truncated` only for its own
  reasons, and the rollout marks a non-terminal step `truncated` once the episode reaches `horizon`.
- **`EnvironmentModel`** is `TabularModel<S>` only (`states`, `actions`, `outcomes`, `terminal`, `terminalValue`,
  `gamma`, `encode`, `decode`); `Outcome` moved to core. Planners take `MdpTables` (the tables a `TabularMdp` and a
  `TabularModel` share), so `valueIteration(env.model)` works unchanged. `dynamics` waits for classic control.
- **`RenderSpec`** is `GridRender<S>`: `width`, `height`, `cells` (kind per cell), `cell(state)`, `actionVectors`.
- **`EnvironmentInfo`** gains optional `observation`/`action` (`DomainKind`) and `capabilities`; `AgentInfo` (kind
  `agent`) has `params` and `requires: { observation?, action?, model? }`. Infos carry domain kinds, not domains,
  because a maze's size depends on its arguments.
- **Rollout semantics**: the step that ends an episode shows the arrival (`ended: true`); the next step resets from
  `child(ctx.stream, 'reset')` and acts. `episodeReturn` is the undiscounted sum of rewards. `episodes` draws step k
  of an episode from `child(ctx.stream, 'step', k)`. `compare` is not built.
- **MDPs become environments through an adapter**, `mdpEnvironment(mdp, { horizon })` (default 4 × states), so every
  gridworld builder keeps returning `TabularMdp`. Arriving at a terminal state pays the outcome's reward plus γ times
  its terminal value, so returns agree with the model's values. `mazeEnvironment({ layout, slip, gamma, rewards,
horizon })` wraps `maze` with built-in `MAZES` (`small`, `classic`, `traps`).

## 4. The loop (applications, `decisions/rollout.ts`)

```ts
/** One environment step per algorithm step; episodes restart on terminated/truncated. Traceable and playable. */
rollout(env, agent, { episodes?, steps?, record? }): Algorithm<void, RolloutState>

/** One episode per algorithm step (the shape today's tabular learners have). */
episodes(env, agent, options): Algorithm<void, EpisodeState>

/** Replicates on child streams with common random numbers across agents; regret curves, returns, quantiles. */
compare(env, agents, { replicates, steps }): ComparisonResult
```

`RolloutState` holds `t`, `episode`, `envState`, `observation`, `agent`, the last transition, `episodeReturn`, and,
when the environment has an oracle, `regret` and `cumulativeRegret`. Streams follow `banditRun`'s convention:
`child(ctx.stream, 'env')` and `child(ctx.stream, 'agent')`, so two agents traced on one root stream face identical
environment randomness.

## 5. Bandits are the one-step case

A (contextual) bandit is an environment with `horizon: 1` whose observation is the round's context (`null`-shaped for a
context-free bandit), whose action is an arm (`integers [0, arms − 1]`), and whose `step` returns `terminated: true`.

- The environment no longer hands every arm's reward to the loop: the pulled arm's reward is the only reward, as in
  reality. Regret comes from the **oracle** (`expectedReward`, `bestExpectedReward`), which the bandit environments
  have exactly.
- `BanditPolicy` becomes an `Agent`: `choose` → `act`, `update` → `learn`. Policies that need bounded rewards (EXP3,
  KL-UCB) read the reward domain (`interval [0, 1]`) instead of a `bounded` flag.
- `banditRun` becomes `rollout(bandit, policy)`; `regretCurves` becomes `compare`. Common random numbers keep working:
  the environment draws every arm's reward from its stream whether pulled or not, so the pulled arm's reward does not
  depend on the agent.
- Non-stationary and adversarial bandits fit as environment state (drifting means) without new machinery.

## 6. MDPs

- `tabularMdp(...)` returns an `Environment` with `model: { kind: 'tabular', ... }` and an `oracle` whose
  `optimalValues` comes from value iteration. Gridworlds, cliff walking, mazes and FrozenLake keep their builders and
  gain `render: { kind: 'grid', ... }`.
- Planners (value iteration, policy iteration, policy evaluation) take `env.model` (requiring the tabular capability)
  and stay `Algorithm`s, unchanged in substance.
- Learners (TD(0), SARSA, Q-learning, expected SARSA, n-step SARSA, Monte Carlo control, REINFORCE) become `Agent`s;
  they no longer call `sampleOutcome` on the table, so they work on any environment with a discrete observation and
  action domain, not only tabular MDPs.

## 7. What it unlocks (later phases, not this plan's build)

- **Classic control** on core's ODE solvers: cart-pole, mountain car, pendulum, acrobot (continuous observations; the
  `dynamics` model is differentiable, so iLQR/MPC can use `grad`).
- **Function approximation**: tile coding, linear TD, semi-gradient SARSA; DQN and actor–critic on the nn module.
- **Off-policy evaluation**: importance sampling and doubly robust estimators over logged `Transition`s (the notes in
  `reinforcement-learning/model-based-and-offline`).
- **Lab**: one environment × agent picker; an episode Player; return/regret curves from `compare`; the grid renderer
  with values and the greedy policy.

## 8. Registry

`EnvironmentInfo` widens to the protocol:

```ts
export interface EnvironmentInfo extends Info {
  readonly kind: 'environment'
  readonly family: 'bandit' | 'contextual-bandit' | 'mdp' | 'control'
  readonly params: Space
  readonly observation: Domain
  readonly action: Domain
  readonly capabilities: readonly ('model' | 'oracle' | 'render')[]
}
```

Agents register as `kind: 'agent'` with `info.requires` (e.g. `{ observation: 'discrete', action: 'discrete' }`, or
`model: 'tabular'` for planners), so the lab offers only valid environment × agent pairs, as the GAM page offers only
valid family × link pairs.

## 9. Build plan

One agent, about half a day; each step ends green (`make check`).

1. **Contracts**: `Domain`, `Environment`, `Step`, `Transition`, `Agent`, `EnvironmentModel`, `EnvironmentOracle`,
   `RenderSpec` in core contracts; `Domain` helpers (`contains`, `uniform(domain, stream)` for a random agent,
   `size` of a discrete domain). Contract tests.
2. **Rollout**: `rollout`, `episodes`, `compare` in `aifn-applied/decisions`. A protocol test (as for Algorithms):
   purity, clone/revive of every state, common random numbers across agents, `terminated` vs `truncated` honoured.
3. **Bandits**: port the three environments and twelve policies; delete `BanditEnvironment`, `banditRun`,
   `regretCurves` (break freely). Regret against the existing fixtures must be identical for the same seeds.
4. **MDPs**: `tabularMdp` returns an `Environment`; port planners to `env.model`, learners to `Agent`; delete
   `sampleOutcome` from the learners. Existing value-iteration and Q-learning results identical for the same seeds.
5. **Registry and catalog**: widen `EnvironmentInfo`, register agents with `requires`; a generated test runs every
   valid environment × agent pair for a few episodes.
6. **Lab**: migrate the bandit and RL pages to `rollout`/`compare`; add one "environment × agent" page with the
   episode Player, as the demonstration of the protocol.
7. **Docs**: an architecture section, and the note links (`multi-armed-bandits`, `contextual-bandits`,
   `markov-decision-processes`, `reinforcement-learning`).

Then, as separate work: classic control environments (step 7 of §7) and function-approximation agents.

### 9a. Left after the lean build

- Bandits: port the three environments to `Environment` (horizon 1, oracle for regret) and the twelve policies to
  `Agent`; replace `banditRun`/`regretCurves` with `rollout`/`compare`; delete `BanditEnvironment`.
- Tabular learners still on their episodic `Algorithm`s with `sampleOutcome`: SARSA, expected SARSA, n-step SARSA,
  Monte Carlo control, TD(0) prediction, REINFORCE, and the episodic `qLearning`/`tdControl` (the cliff page uses
  them). Port each to an `Agent`, then delete the episodic forms and `sampleOutcome` from the learners.
- `oracle` on the MDP environments (`optimalValues` from value iteration), `legal` masking, a `compare` with common
  random numbers, a generated test over every valid environment × agent pair from the registry.
- Planners registered as agents/algorithms with `requires: { model: 'tabular' }`; the lab's planning and cliff pages
  moved onto `mdpEnvironment`; the `box` domain and the pendulum on core's ODE solvers.

## 10. Questions for the owner

1. **Placement**: contracts in core and implementations in applications (recommended), or a new core `decisions`
   family? Core would only be justified once a core algorithm depends on environments (none does).
2. **Bandit rewards**: the pulled arm's reward only, with regret from the oracle (recommended), or keep the
   all-rewards round for counterfactual plots (it can also come from the oracle)?
3. **Agent shape**: a separate `Agent` with `act`/`learn` (recommended: one transition at a time suits TD and bandits,
   and episode learners buffer), or agents as plain `Algorithm`s over episodes as today's learners are?
4. **Scope of the first build**: steps 1–7 only, or include classic control (cart-pole, mountain car, pendulum) in the
   same run?

## 11. Decided (2026-10-01)

- Contracts in core, implementations in applications; bandits reveal the pulled arm's reward only (regret from the
  oracle); a separate `Agent` with `act`/`learn`.
- **Lean first build**: the contracts, `rollout`/`episodes`, and a basic **maze** as the prototype use case, with one
  or two agents (random, Q-learning) and one lab page. Break freely; port the rest of bandits/MDPs only as far as the
  maze path needs; tests at the end. Next: an inverted pendulum (classic control) on core's ODE solvers.
- **Domains (owner, 2026-10-01)**: discrete action spaces only to start, the action domain first-class
  (`{ kind: 'discrete', n, names? }`), declared by the environment and read by agents (size, names), used by the random
  agent and by the registry to pair agents with environments; room left in `Domain` for a continuous `box`;
  observations discrete for now too.
