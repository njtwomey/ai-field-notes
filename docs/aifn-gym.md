# aifn gym: one protocol for bandits, reinforcement learning and control

Status: **built (2026-10-01)**: the contracts, `rollout`/`episodes`/`compare`, every bandit, MDP and control
environment and every agent on the protocol, registered and paired by a generated test, and the lab's `applied/gym`
pages. §3a lists where the build departs from this proposal, §4a gives the layout, §9a what was done in the port.

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

## 3. The contracts (core, `foundation/contracts/gym.ts`)

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

- **`Domain`** is `DiscreteDomain | BoxDomain`: `{ kind: 'discrete'; n; names? }` (owner's direction), and, for
  classic control, `{ kind: 'box'; low; high; shape; names? }` with flat row-major bounds (Gymnasium's `Box`).
  Agents read a domain's size and names, the random agent draws from it, and the registry pairs agents and
  environments by `DomainKind`. `Support` and `record` domains are dropped for now. Helpers in
  `aifn/foundation/space`: `discreteDomain`, `boxDomain`, `domainContains`, `clipToDomain` (round and clip, or clip
  each element), `domainSize` (`Infinity` for a box: continuous), `domainDimension` (numbers per value) and
  `sampleDomain(stream, domain)` (an integer, or a `Float64Array` uniform on a bounded box).
- **`EnvironmentShape`** is a named interface (name, domains, `gamma`, `horizon`, and the optional `model`) that
  `Environment` extends and `Agent.init` receives, so a planning agent can read the model. `act(agent, observation,
stream, legal?)` returns a `Decision<A>` (`Choice` was taken by the bandits); the rollout passes `env.legal(state)`
  as `legal` when the environment masks actions, and a `Transition` carries `nextLegal` for maxima at `next`.
- **`horizon` is enforced by the rollout** (Gymnasium's `TimeLimit`): `env.step` sets `truncated` only for its own
  reasons, and the rollout marks a non-terminal step `truncated` once the episode reaches `horizon`.
- **`EnvironmentModel`** is `TabularModel<S> | DynamicsModel<S>` (`states`, `actions`, `outcomes`, `terminal`, `terminalValue`,
  `gamma`, `encode`, `decode`); `Outcome` moved to core. Planners take `MdpTables` (the tables a `TabularMdp` and a
  `TabularModel` share), so `valueIteration(env.model)` works unchanged. The union
  also holds `DynamicsModel<S>` (§3b).
- **`RenderSpec`** is `GridRender<S> | PendulumRender<S>`. Grid: `width`, `height`, `cells` (kind per cell),
  `cell(state)`, `actionVectors`. Pendulum: `length` and `angle(state)` (radians from upright, anticlockwise).
- **`EnvironmentInfo`** gains optional `observation`/`action` (`DomainKind`) and `capabilities`; `AgentInfo` (kind
  `agent`) has `params` and `requires: { observation?, action?, model?, families? }` (`families`: the environment
  families an agent is meant for, so a bandit policy is not offered a maze). Infos carry domain kinds, not domains,
  because a maze's size depends on its arguments. Every registered factory builds from its `params` defaults.
- **Rollout semantics**: the step that ends an episode shows the arrival (`ended: true`); the next step resets from
  `child(ctx.stream, 'reset')` and acts. `episodeReturn` is the undiscounted sum of rewards, `totalReward` the sum
  over the run, and `cumulativeRegret` adds `bestExpectedReward(s) − expectedReward(s, a)` per step when the oracle
  has both (the bandits). `episodes` draws step k of an episode from `child(ctx.stream, 'step', k)`. `compare(env,
agents, { replicates, steps, stream?, points? })` runs replicate k of every agent on `child(stream, k)` exactly as a
  trace of `rollout` on that root (common random numbers), and returns cumulative reward and regret spreads (mean, sd,
  10 % and 90 % quantiles), mean episodes, mean action counts and every replicate's final value.
- **MDPs become environments through an adapter**, `mdpEnvironment(mdp, { horizon })` (default 4 × states), so every
  gridworld builder keeps returning `TabularMdp`. Arriving at a terminal state pays the outcome's reward plus γ times
  its terminal value, so returns agree with the model's values. Its `oracle` has `optimalValues` (V* by value
  iteration, computed once) and `expectedReward`; `legal` lists the actions with outcomes when some state has an
  illegal one (an all-zero row of `tabularMdp`'s P), and planners give illegal actions Q = −∞. The registered forms
  are `gridworldEnvironment`, `cliffWalkingEnvironment`, `mazeEnvironment` (built-in `MAZES`: `small`, `classic`,
  `traps`) and `frozenLakeEnvironment`.
- **Bandits** are `Environment`s with horizon 1 (`bernoulliBandit({ means })`, `gaussianBandit({ means, sd })`,
  `linearBandit({ theta, arms, mode, noise })`). The linear bandit's observation is the round's arms, a flat
  arms × d array in the box [−1, 1] of shape [arms, d], drawn by `reset`. `step` draws every arm's reward in arm order
  and reveals the pulled one's.

### 3b. Classic control: the inverted pendulum (2026-10-01)

- **`DynamicsModel<S>`** (`kind: 'dynamics'`): `stateSize`, `actionSize`, `transition(x, u)`, `reward(x, u)`,
  `encode(state)`, `decode(x)`. x and u are vectors; both functions are written with tensor primitives, so they
  accept traced values: `jacobian`, `grad`, and later iLQR and MPC differentiate through them.
  `AgentRequires.model` takes `'dynamics'`.
- **`pendulumEnvironment(options)`** (`aifn-applied/gym/environments/control`) is Gymnasium's `Pendulum-v1`: state
  (θ, θ̇) with θ = 0 upright; observation box (cos θ, sin θ, θ̇) ∈ [−1, 1]² × [−8, 8]; θ̈ = (3g/2l) sin θ + (3/ml²) u
  − c θ̇ with g = 10, m = l = 1 and damping c = 0 by default; reward −(θ² + 0.1 θ̇² + 0.001 u²) with θ wrapped to
  [−π, π) and u clipped; horizon 200 steps of dt = 0.05 s; reset θ ~ U[−π, π), θ̇ ~ U[−1, 1), or a fixed `start`.
  Actions: a box [−`maxTorque`, `maxTorque`] (a length-1 `Float64Array`, default limit 2), or, with `torques: n`,
  n evenly spaced torque levels as a discrete domain. `parameters` exposes the constants; `pendulumEnergy` the specific
  energy ½θ̇² + (3g/2l) cos θ.
- **Integrator: RK4, not Gymnasium's semi-implicit Euler.** One step is one classical RK4 step of size dt from
  `rungeKutta(…, 'rk4')` (`aifn/dynamics/ode`) with u held constant, then θ̇ is clipped to ±8. Gymnasium updates
  θ̇ ← θ̇ + dt θ̈ and then θ ← θ + dt θ̇, so trajectories agree to O(dt) per step but not exactly. RK4 was chosen because
  it is core's traceable solver and keeps the unforced, undamped energy to O(dt⁴) per unit time.
  `step` calls `model.transition`, so the simulated and the modelled dynamics are one definition.
- **Agents** (`aifn-applied/gym/agents/control`): `lineariseDynamics(model, x̄, ū)` gives A = ∂f/∂x and B = ∂f/∂u of the
  discrete transition by `jacobian`; `pendulumLqr` solves `dlqr` (`aifn/dynamics/control`) with Q = diag(1, 0.1) and
  R = 0.001 (the reward's weights). `swingUpAgent({ swingUp, energyGain, switchAngle, r })` reads the plant and solves the gain in `init(env)`, then pumps energy with
  u = k (e* − e) sign(θ̇) (Åström and Furuta, 2000) and hands over to u = −K (θ, θ̇) once |θ| < 0.6. With
  `swingUp: false` it is LQR alone. It plays the nearest level on a discrete pendulum and reports
  `scores[0]` = 1 when LQR acted. From hanging at rest it is upright by step 75 (return ≈ −377) with torque limit 2,
  and by step 108 with limit 1. With limit 0.5 it does not reach upright within 200 steps. `randomAgent` draws
  uniformly from a box.
- **Not built:** a learning agent on the discrete-torque pendulum. Tabular Q-learning needs a discrete observation,
  so it waits for a tile-coding or discretising observation wrapper.
- **Lab:** `applied/gym` → "Environment × agent: an inverted pendulum": the rod at the played step, with a draggable
  start, θ and θ̇ and the torque over the episode, return per episode, and the episode Player. T

### 3c. Classic control: the cart-pole (2026-10-01)

- **`cartPoleEnvironment(options)`** (`aifn-applied/gym/environments/control`) is Gymnasium's `CartPole-v1`. The state
  and observation are (x, ẋ, θ, θ̇) as a box, with θ from upright and positive towards +x. There are two discrete
  actions, push left and push right, each a force of ±10 N. The reward is +1 per step. An episode is `terminated`
  when |θ| > 12° or |x| > 2.4 m, and the rollout truncates it at 500 steps. Every state element starts uniform on
  [−`jitter`, `jitter`] (default 0.05). The render spec is `CartPoleRender` (`poleLength`, `trackLimit`, `cart`,
  `angle`).
- **Integrator: explicit Euler with τ = 0.02 s, as Gymnasium** (Florian's equations without friction). The step is
  written once with tensor primitives. `step` runs it on numbers, which is fast enough for a few hundred thousand
  steps in the browser. `model.transition(x, u)` runs the same step on values, with u a force in newtons, so
  `lineariseDynamics` and `jacobian` work on it.
- **Agents** (`aifn-applied/gym/agents/control`):
  - `lqrBangBangAgent({ r })`. Its `init` linearises the model about the origin by autodiff and solves `dlqr` with
    Q = I and R = r. It pushes towards the sign of −K o. It holds 500 steps from every tested start, at jitter 0.05
    and 0.1.
  - `crossEntropyAgent({ population, eliteFraction, initialStd, noise })`. This is the cross-entropy method over
    linear threshold policies 1[w · (o, 1) > 0], with a decaying extra noise noise / (1 + g) on σ (Szita and
    Lőrincz, 2006). It is an episode-level `Agent` whose state is plain data, and it draws each generation's
    population from a seed fixed in `init`. `greedy` plays the mean weight vector. `scalars` reports the generation
    and elite mean returns. It was chosen over REINFORCE and discretised Q-learning because its defaults
    (population 20, elite 20 %, σ₀ = 0.3, noise 0.5) reach a greedy mean return of 500 within 40 generations
    (800 episodes, about 1 s) on 10 of 10 probe seeds.
  - `linearPolicyAgent(weights)` is unregistered. It plays a fixed linear threshold policy.
- **Lab:** `applied/gym` → "Environment × agent: a cart-pole" runs on `GymTrainer`. Training runs in the worker,
  then the page shows the learning curve, the generation mean return and the episode lengths. Any episode can be
  picked to replay or evaluate. `CartPoleView` draws the cart and the pole. `GymTrainer` has no slot for the x and θ
  time series or the action strip, so the page does not show them yet.
  he rod is drawn by
  `PendulumView` (`aifn-lab/src/views/gym/`), which `GymTrainer` also uses for the `pendulum` render kind.
  The swing-up, LQR and cart-pole agents implement `greedy`.

### 3d. Function approximation: the deep Q-network (2026-10-01)

- **`dqnAgent(options)`** (`aifn-applied/gym/agents/dqn.ts`) is DQN after Mnih et al. (2015) on the schedule of
  Stable-Baselines3's `DQN`, for a box observation and discrete actions (`requires: { observation: 'box', action:
'discrete', families: ['control'] }`). The Q-network is `qNetwork(sizes, activation, layerNorm)`: dense layers from
  aifn nn (`Linear`, optional `LayerNorm`, `ActivationLayer`) with PyTorch's default initialisation (weights and biases
  uniform on ±1/√fan-in), one linear output per action. Each gradient step draws a fresh minibatch and takes one Adam
  step (`adamRule`, ε 10⁻⁸, chained after `clipByGlobalNorm(clipNorm)`) on the Huber loss between Q(o, a) and
  y = r + γ (1 − terminated) Q̄(o′, a*). With double DQN (van Hasselt et al., 2016) a* is the online network's argmax
  at o′, otherwise the target's (SB3). A truncated transition still bootstraps (SB3's `handle_timeout_termination`).
  Defaults: 4 → 64 → 64 → 2 ReLU, lr 10⁻³, batch 64, buffer 10 000, warm-up 1000, one gradient step every 4 steps,
  sync 500, ε 1 → 0.05 over 10 000 steps, double DQN, γ the environment's. `SB3_CARTPOLE` holds the Zoo recipe.
- **The mapping from SB3** (`DQN`, `OffPolicyAlgorithm.collect_rollouts`, `DQN.train` and `DQN._on_step`), with t the
  environment steps so far:

  | SB3                                           | `dqnAgent`                                | Semantics                                                                       |
  | --------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
  | `learning_starts`                             | `warmup`                                  | uniformly random actions while t < warmup; training only once t > warmup        |
  | `train_freq` (steps), `gradient_steps`        | `updateEvery`, `gradientSteps`            | after step t with t mod updateEvery = 0: `gradientSteps` updates, fresh batches |
  | `target_update_interval`, `tau`               | `targetSync`, `tau` (default 1)           | Q̄ ← τ Q + (1 − τ) Q̄ after every step with t mod targetSync = 0, before training |
  | `exploration_fraction`, `_final_eps`          | `epsilonSteps`, `epsilonEnd`              | ε = 1 + (ε_end − 1) min(1, t / epsilonSteps); epsilonSteps = fraction × budget  |
  | `max_grad_norm = 10`, smooth L1, Adam         | `clipNorm`, Huber (δ 1), `adamRule`       | the same rules                                                                  |
  | DQN (not double), `optimize_memory_usage` off | `double: false`; o′ stored per transition | the target network's max at o′                                                  |
  | `total_timesteps`                             | `train(…, { steps })`                     | SB3 stops at exactly the budget; `train` finishes the episode that crosses it   |

  Known differences: aifn computes in float64 (SB3 float32); the random streams differ, so runs match in law, not
  draw for draw; SB3 samples random actions during warm-up from the action space and ε-greedy from NumPy, aifn from
  its own streams; SB3 cuts the last episode at the budget.

- **Plain-data state:** the two networks and Adam's moments as parameter pytrees of tensors, counters, the minibatch
  seed and the buffer. `learn` takes no stream, so minibatch u is drawn from `child(stream(seed), 'batch', u)` with the
  seed fixed in `init`. `act` draws ε-greedy exploration from its own stream. The state passes the Algorithm protocol
  (clone and revive, seek, extend), and `replay` reproduces training episodes exactly.
- **The replay buffer is persistent.** A copied ring buffer costs O(capacity) per step, and a mutated one corrupts the
  checkpoints and trace states that share it. The buffer stores transitions (o, a, r, terminated, o′; Float32) in
  sealed chunks of 256 that are never written again and are shared by every later state, plus a tail of fewer than 256
  that each push copies. A state holds only the chunks that overlap its window of the last `bufferSize` transitions,
  so the window is bounded like a ring buffer while checkpoints share all but their tails.
- **Checkpoint memory** (151 checkpoints from `train`'s default spacing, unique typed-array bytes): about 16–17.5 MB for
  300 episodes with the 64 × 2 network, of which 1–2 MB is replay data and the rest networks and Adam moments (about
  110 KB per checkpoint); about 7 MB with 32 × 2. So `train` keeps its usual spacing: thinning the checkpoints would
  save memory but make `replay` and `evaluateEpisode` re-run more learning steps on the page.
- **Cost** (Node, batch 64, aifn tensors and reverse-mode autodiff): 4 → 64 → 64 → 2 forward 0.45 ms, value and
  gradient 1.5 ms, Adam with clipping 0.3 ms; one update with the two target forwards about 2.7 ms. 4 → 32 → 32 → 2:
  about 1 ms. An environment step with the greedy forward and the buffer push is about 20 µs. 300 episodes at the
  defaults take 3–40 s depending on how long the episodes become.
- **How well it learns (first build, before the SB3 schedule and init):** DQN learns on the cart-pole and then partly forgets. With the defaults the moving
  return peaks at 150–350 around episodes 200–275 and falls back to about 100–150, and the final greedy policy
  returns 20–180; some seeds first lock onto one action (returns ≈ 10) for a hundred or more episodes. No probe recipe
  (lr 3 × 10⁻⁴–2.3 × 10⁻³, update every 1–4 steps, sync 100–1000, buffer 10–100 k, Polyak averaging, γ 0.995, nets
  32 × 2 to 128) reached a final greedy mean of 450 on most seeds within 500–800 episodes. A PyTorch DQN on Gymnasium's
  CartPole equations with the same recipe gave the same curve shape (peaks 180–260, final greedy means 111, 111 and 272
  on three seeds), so the plateau is the recipe and budget, not the implementation. In PyTorch, a fast ε decay (over
  2000 steps) with Polyak averaging reached returns of 500 on some seeds but did not stay there. The owner dropped the
  ≥ 450 bar; the test asserts that DQN learns (the mean return of the last 50 of 200 episodes is at least three times
  the random agent's on three of three seeds, with 32 × 2, an update every 2 steps and ε over 5000 steps).
- **The Zoo recipe on the small network** (50 000 steps; lr 2.3 × 10⁻³, batch 64, buffer 100 000, learning starts
  1000, γ 0.99, target every 10 steps, 128 gradient steps every 256, ε to 0.04 over 0.16 of the budget, plain DQN;
  network 4 → 32 → 32 → 2 as the owner chose, not the Zoo's 256 × 2). aifn takes 26–27 s per run in Node (24 700
  gradient steps at about 1 ms each). The mean return of the episodes ending in the last tenth of the budget, seeds
  0, 1, 2: aifn 81, 105, 111; a torch reproduction of SB3's loop on Gym's CartPole equations with the same network
  273, 23, 268. Both curves rise to about 90–220 by 10 000 steps and then wander (aifn seed 0 averages 493 between 30 000
  and 35 000 steps; torch seed 2 reaches 500 at 45 000), so the shape agrees and the seed-to-seed spread is wider than
  the gap. With the Zoo's 256 × 2 network the torch loop rises more slowly and steadily: 388, 167 and 474. In aifn that
  network costs about 27 ms per gradient step (forward 5.4 ms, value and gradient 17 ms, clipped Adam 4.9 ms; one
  64 × 256 by 256 × 256 matmul is 4.9 ms), about 11 minutes for the budget, so the page keeps 32 × 2.
- **Lab:** the cart-pole page trains DQN on a step budget (default 50 000) with the Zoo recipe as its default and a
  "SB3 RL Zoo preset" button that restores it (training fields only; the network row is separate). Rows: DQN
  (learning rate, budget, train every, gradient steps, batch, target sync, buffer, double DQN, an "advanced" switch),
  advanced (γ, learning starts, final ε, ε decay as a fraction of the budget, gradient clip) and network (hidden
  layers, width, activation, layer norm). `GymTrainer` takes a `scalars` prop that draws up to three agent scalars in
  their own row: the page shows ε, the loss and the mean max Q at the sampled next states (the last two averaged over
  each episode's updates).

## 4. The loop (applications, `gym/rollout.ts`)

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

### 4a. Layout (owner's direction, 2026-10-01: everything under `gym`)

| Where                                                                         | What                                                                                 |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| core `aifn/foundation/contracts` (`gym.ts`)                                   | `Domain`, `Environment`, `Step`, `Transition`, `Agent`, models, oracle, render specs |
| core `aifn/foundation/space` (`domain.ts`)                                    | `discreteDomain`, `boxDomain`, `domainContains`, `sampleDomain`, …                   |
| `aifn-applied/gym` (`index.ts`, `rollout.ts`, `mdp.ts`)                       | `rollout`, `episodes`, `compare`; `TabularMdp` and its helpers; the two registries   |
| `aifn-applied/gym/environments` (`bandits`, `gridworlds`)                     | bandits; grid MDPs and `mdpEnvironment`; child `control` (the pendulum)              |
| `aifn-applied/gym/agents` (`random`, `bandits`, `tabular`, `planning`, `dqn`) | random; bandit policies; tabular learners; planners; DQN; child `control`            |
| lab `applied/gym`                                                             | bandit regret and round-by-round, maze, cliff, planning and pendulum pages           |

Registry addresses read `gym/environments/<key>` and `gym/agents/<key>`; `gym/index.ts` exports
`environmentRegistry`, `agentRegistry`, `compatible(envInfo, agentInfo)` and `validPairs()`.

### 4b. Headless training and the trainer (2026-10-01)

- **One training-run row (2026-10-01).** Every `GymTrainer` page spreads `run: trainingRun(defaults)` (`views/gym`)
  into its `useFigureState`: the budget in episodes or environment steps, its size and the seed, in the figure's URL
  state; `GymTrainer` reads `state.run`, so pages declare no budget or seed fields. `gymSetup(envKey, envParams,
agentKey, agentParams)` builds the page's environment and agent from the gym registries and the matching worker
  tasks at their registry addresses, replacing each page's factory-and-address tables. SB3's ε fraction becomes steps
  through `epsilonStepsFor(fraction, budgetSteps)` beside `SB3_CARTPOLE`.
- **Step budgets and Stop (2026-10-01).** `train` and `training` take `{ steps }` instead of `{ episodes }`: episodes
  run until at least that many environment steps are done, and the episode that crosses the budget runs to its end, so
  every episode replays as it happened. `Training` reports `budget` and `steps`; under a step budget the checkpoint
  spacing starts at `checkpointEvery` and doubles, dropping every other checkpoint, whenever there would be more than
  `maxCheckpoints`. `GymTrainer` takes `steps` in its setup and shows progress in steps. While training, its Train
  button becomes Stop: `useStreamed(task).stop()` terminates the worker (`ComputeWorker.cancel`) and keeps the last
  streamed partial run, which carries every checkpoint so far, so curves, replay and evaluate work on it; the status
  reads "stopped at N episodes / M steps".
- `train(env, agent, { episodes, seed, checkpointEvery?, maxCheckpoints = 200 })` (`gym/train.ts`) runs episodes with
  no display and returns plain data: per-episode columns (return, length, terminated, pseudo-regret when the oracle
  knows it, the first action, and the agent's optional `scalars`) and agent-state checkpoints every
  `max(checkpointEvery, ⌈episodes / maxCheckpoints⌉)` episodes from 0, plus the final state. Episode e runs on
  `child(stream(seed), 'step', e − 1)` exactly as a trace of `episodes` would. `training` is the same run as a
  generator yielding partial runs; the lab's compute worker streams any generator result as partial answers.
- `replay(env, agent, t, e)` re-runs training episode e from the nearest checkpoint, exactly as it happened;
  `evaluateEpisode(env, agent, t, e, seed)` plays a fresh episode of the policy after e episodes with no learning,
  using the agent's optional `greedy` action (its `act` otherwise). Both return a `Trajectory` (environment states,
  observations, actions, rewards). `runEpisode` is the one episode loop under `episodes`, `train` and both.
- Lab: `GymTrainer` (`@lab/views`) takes the page's controls and a `setup` (environment and agent on the page and as
  worker tasks), trains on mount and on Train, draws return (with a moving average), the first agent scalar and length
  or cumulative regret, with a draggable episode marker, a replay/evaluate switch, a Player over the chosen episode,
  and the environment drawn by `GYM_RENDERERS[render.kind]` (`grid`, `bandit`, `pendulum`, `cartpole`).
- **Outcomes**: an environment's optional `ending(state, 'terminated' | 'truncated', steps)` returns an `EpisodeEnd`
  (`success`, `reason`): the cart-pole fails when the pole falls or the cart leaves the track and succeeds when it
  survives the time limit; a grid MDP succeeds at a goal and fails in a hole, a trap or a time-out. Trajectories carry
  `ending` and training an `outcome` column. At an episode's last step `GymTrainer` draws the scene in the theme's
  destructive or success tone with the reason; the learning curve marks each episode by outcome and plots the
  failure share over a trailing window. Per-step panels (`GYM_SERIES` per render kind, plus the actions as a strip or
  lines) sit under the Player; the pendulum page evaluates from a draggable start angle (`evaluationEnv`).

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
- **Function approximation**: tile coding, linear TD, semi-gradient SARSA; actor–critic on the nn module (DQN is built,
  §3d).
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
2. **Rollout**: `rollout`, `episodes`, `compare` in `aifn-applied/gym` (`gym/rollout.ts`). A protocol test (as for Algorithms):
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

### 9a. The port (done 2026-10-01)

- **Bandits**: the three environments are `Environment`s with horizon 1 and an oracle; the twelve policies (uniform,
  explore-then-commit, ε-greedy and decaying ε-greedy, UCB1, KL-UCB, Bernoulli and Gaussian Thompson sampling, EXP3,
  LinUCB, greedy ridge, linear Thompson sampling) are `Agent`s whose round number is one more than their pulls; EXP3
  recomputes the pulled arm's probability in `learn`. `banditRun`, `regretCurves`, `BanditEnvironment`, `BanditPolicy`
  and the `bounded` flag are deleted. **Parity:** `compare` reproduces the old `regretCurves` exactly (final regret and
  pulls to 1e-9, every replicate) for all nine context-free policies on Bernoulli and Gaussian arms. Intended
  difference: the linear bandit draws its context from the reset stream and its noise from the step's `env` stream,
  so its runs differ from the old ones.
- **Tabular learners** are `Agent`s (`gym/agents/tabular.ts`): `tdControlAgent` (`qLearningAgent`, `sarsaAgent`,
  `expectedSarsaAgent`), `nStepSarsaAgent`, `monteCarloControlAgent`, `tdPredictionAgent({ policy })` and
  `reinforceAgent`. SARSA holds one transition until the next action arrives (so its next action is chosen before
  the update, as in the episodic form), n-step SARSA a window of n, Monte Carlo control and REINFORCE the episode. A
  truncated transition bootstraps on-policy methods from the ε-greedy expectation at `next`. The episodic
  `Algorithm` forms are deleted and no learner calls `sampleOutcome` (only `mdpEnvironment.step` does). Random
  streams are per step now, so learning runs differ from the old episodic ones on the same seed.
- **Planners**: `valueIteration`, `policyIteration` and `policyEvaluation` stay traceable `Algorithm`s on
  `MdpTables` (`env.model`); `valueIterationAgent` and `policyIterationAgent` plan on `env.model` in `init` and act
  greedily, registered with `requires: { model: 'tabular' }`. Policy iteration evaluates an improper policy under
  γ = 1 with γ = 1 − 10⁻⁹ (the exact system is singular).
- **Registry**: 7 MDP and bandit environments plus the pendulum, and every agent; `test/gym/registry.test.ts` runs
  every `validPairs()` pair for three episodes from the registry defaults.
- **Lab** (`applied/gym`): the regret page runs on `compare`, the round-by-round page on `rollout`, the planning page
  on the environments' `model`, the cliff page on `episodes` of the two agents; the maze page's overlay is a reveal
  toggle (off by default) and its two grids share equal-units axes in a larger top row.

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
- **One organisation (owner, 2026-10-01)**: all of it lives under `gym` (§4a): `aifn-applied/gym` replaces
  `decisions/bandits`, `decisions/reinforcement-learning` and `data/environments`; the core contract file is
  `contracts/gym.ts`; the lab has one module, `applied/gym`.
