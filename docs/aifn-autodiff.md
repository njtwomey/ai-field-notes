# How `grad` works in aifn

This explains reverse-mode automatic differentiation as aifn implements it today, step by step, with one worked example.
The code lives under `aifn-js/core/src/foundation/`: `tensor/tape.ts` (the hook), `tensor/primitive.ts` (primitives),
`autodiff/tape.ts` (the tape and the reverse sweep) and `autodiff/transforms.ts` (`grad` and friends). §8 says what the
planned redesign changes.

## 1. The idea in one paragraph

`grad(f)(x)` runs `f` once, forwards, and **records every primitive operation** it performs (add, multiply, sin,
matmul, …) on a **tape**, in the order they happen. Then it walks the tape **backwards**, starting from the output
with the value 1, and asks each recorded operation "given how much the output changes per unit change of your result,
how much does it change per unit change of each of your inputs?". That question is answered by the operation's
**vector–Jacobian product (vjp)**, written next to the operation when it is defined. Contributions that reach the same
value along different paths are added. When the walk reaches `x`, the accumulated number is ∂f/∂x.

This is the chain rule, applied mechanically in reverse order: for y = f(g(x)), dy/dx = f′(g(x)) · g′(x), and the
reverse walk multiplies the outer factor first.

## 2. Three kinds of value

Every numeric function in aifn accepts and returns a `Value`:

| Kind               | Example                     | What a primitive does with it                                                                  |
| ------------------ | --------------------------- | ---------------------------------------------------------------------------------------------- |
| a number           | `2`                         | computes and returns a number                                                                  |
| a tensor           | `tensor([1, 2])`            | computes elementwise or structurally and returns a tensor                                      |
| a **traced** value | `{ value: 2, id: 7, tape }` | computes on the wrapped value **and records itself on the tape**, returning a new traced value |

A traced value is a raw number or tensor plus a pointer to a node on a tape (`tensor/tape.ts`: `traced`, `isTraced`,
`unwrap`). Your function never sees the difference: `sin(x)` works the same whether `x` is a number or traced. That is
why `f` needs no special code to be differentiable.

## 3. Primitives: each operation defined once, with its derivative

A **primitive** is an operation with two rules, declared together (`definePrimitive`, or the shorter `defineUnary`,
`defineBinary`, `defineOp` in `tensor/primitive.ts`):

- **forward** (`impl`): compute the result from raw numbers or tensors;
- **vjp**: given `g`, the cotangent of the result (∂output/∂result), return one cotangent per input
  (∂output/∂input = g · ∂result/∂input, in the right shape).

For example, `exp` is declared as

```ts
unaryPrimitive('foundation/tensor/exp', Math.exp, (g, _x, y) => mul(g, y)) // d/dx eˣ = eˣ, which is the output y
```

and multiplication's rule is "the cotangent for a is g·b, the cotangent for b is g·a". When a primitive is called:

1. If **no input is traced**, it just runs `impl` and returns the raw result. This is the fast path: most of aifn runs
   this way, with no tape at all.
2. If **any input is traced**, it runs `impl` on the unwrapped inputs, then calls `tape.record(name, inputs, output,
vjp)`, which appends a **node** and returns the output wrapped as a traced value pointing at that node.

Two details matter for correctness:

- **Broadcasting.** If `a` has shape [3] and `b` is a number, `add(a, b)` broadcasts `b`. In the backward pass the
  cotangent for `b` must be summed back to a number. `sumLike(g, like)` does that reduction; the elementwise helpers
  apply it automatically.
- **Rules are written with primitives.** `mul(g, y)` inside `exp`'s rule is itself a primitive call. If the rule runs
  with traced inputs, it records itself too. This is what makes second derivatives possible (§6).

Every registered primitive lives in one registry (`tensor/registry.ts`), keyed `module/name`, and registering the same
id twice throws. The generated test suite (`core/test/primitives.test.ts`) checks every rule against finite
differences.

## 4. The tape

The tape (`autodiff/tape.ts`, class `GraphTape`) is an array of nodes in evaluation order:

```ts
type TapeNode = {
  op: string // 'exp', 'mul', … or 'input'
  inputs: readonly Value[] // traced inputs point at earlier nodes; raw inputs are constants
  output: number | Tensor // the forward value
  vjp: Vjp | null // the rule; null means "not differentiable"
  leaf: boolean // true for an input of the transform
}
```

Because nodes are appended as the computation runs, **the order of the array is a topological order** of the
computation graph: every node comes after the nodes it depends on. So walking the array backwards visits each node
after everything that uses it. No explicit graph search is needed.

## 5. The flow of `grad(f)(x)`, step by step

`grad` is `valueAndGrad` keeping only the gradient (`autodiff/transforms.ts`).

1. **Open a session.** Create a fresh `GraphTape` (or reuse the enclosing one if this `grad` is inside another
   transform; §6).
2. **Record the inputs.** `x` may be a number, a tensor or a pytree (nested arrays and objects of them). It is
   flattened into leaves, and each leaf becomes an `input` node on the tape and is replaced by its traced value.
   `argnums` picks which arguments are differentiated; the others stay raw constants.
3. **Run `f` forwards** with that tape active (`withTape`). Every primitive touching a traced value records a node.
   The result `y` is traced.
4. **Check the output** is a scalar (a number or a rank-0 tensor): a gradient is defined for scalar functions. The
   seed cotangent is 1.
5. **Reverse sweep** (`sweep` in `autodiff/tape.ts`):
   1. **Find the relevant nodes.** Only nodes between the first input and the output are considered. A quick forward
      pass over them marks which ones actually depend on an input; everything else is a constant for this gradient.
   2. **Seed** the output node's cotangent with 1.
   3. **Walk backwards** from the output. For each node that has a cotangent `g`:
      - if it is an input node, stop there: its cotangent is the answer for that input;
      - if its rule is `null` and it lies on a path to the output, throw `NotDifferentiableError` (never a silent 0);
      - otherwise call its `vjp(g, inputs, output)` to get one cotangent per input, and **add** each into that input
        node's running total (a value used twice receives two contributions).
6. **Collect.** Each input leaf's total cotangent is its gradient. Leaves the output never depended on get zeros of
   their own shape. The leaves are rebuilt into the argument's original structure, so a gradient has the same shape
   (or pytree shape) as its argument.

Cost: one forward pass plus one backward pass, whatever the number of inputs. That is why reverse mode suits
functions of many parameters with one scalar output (losses).

## 6. Nested `grad`: second derivatives

`grad(grad(f))(x)` works because the rules are written with primitives:

- The **outer** `grad` opens a tape and records `x` as an input.
- The **inner** `grad` sees that a tape is already active, so it **shares it** (a "nested session"). It records its
  own input as an identity node of the outer input, so it differentiates with respect to its node while the outer
  transform still sees the dependence.
- The inner reverse sweep runs **on the tape** (`createGraph: true`): its vjp rules receive traced values, so the
  backward computation is itself recorded as more nodes. The inner gradient is therefore a traced value, a function
  of `x` that the tape knows how to differentiate.
- The **outer** sweep then differentiates that recorded backward computation, giving the second derivative.

The outermost sweep of any `grad` runs its rules against a `NullTape`, which records nothing, so first derivatives
never pay for this machinery.

## 7. The other transforms, and inspection

| Transform                   | How it works today                                                                                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valueAndGrad(f)`           | as §5, returning `{ value, grad }` from the same passes                                                                                                                                              |
| `vjp(f, x)`                 | records one forward pass and returns a `pullback(u)` that runs a reverse sweep seeded with `u` instead of 1 (for non-scalar outputs)                                                                 |
| `jvp(f, x, v)`              | forward-mode directional derivative J·v, currently obtained by the "transpose trick": a vjp whose backward pass is recorded, then differentiated again. Three passes; the redesign makes it one (§8) |
| `jacobian(f)`, `hessian(f)` | one reverse sweep per output element (or `grad(grad(f))` per row), stacked. Fine for small problems                                                                                                  |
| `hvp(f, x, v)`              | Hessian–vector product without forming the Hessian                                                                                                                                                   |
| `stopGradient(x)`           | an identity primitive with a zero rule: the value passes, the derivative does not                                                                                                                    |
| `gradCheck(f, x)`           | compares `grad` with central finite differences and reports the error                                                                                                                                |
| `traceGraph(f, x)`          | runs the forward pass and a sweep with `keepAll`, returning every node with its value, its adjoint and the local partials: the data behind the lab's "Backpropagation, node by node" figure          |

## 8. What the planned redesign changes (design K §4)

The public API (`grad`, `valueAndGrad`, `vjp`, `jvp`, `hvp`, `jacobian`, `hessian`) keeps its names. Internally:

- The tape becomes one of **three interpreters** over the same primitive registry: _reverse_ (today's tape), _forward_
  (dual numbers carrying a tangent: true one-pass `jvp`) and _batch_ (`vmap`: run a function over a leading axis).
  They nest by level, as in JAX, which replaces today's "shared tape, identity input nodes" arrangement for nesting.
- Every general primitive declares a **jvp rule as well as its vjp**, and the generated tests check the two against
  each other (⟨u, J v⟩ = ⟨Jᵀ u, v⟩), so they cannot disagree.
- **Custom rules** (`customVjp`, `customJvp`) for composite functions, **checkpointing** (recompute instead of store),
  and **implicit differentiation** of fixed points and solves (differentiate the answer of an iterative algorithm
  without unrolling it).
- **Linear-algebra derivatives** for `eigh`, `svd`, `qr`, `expm`, and one factorisation per `cholesky` or `solve`.

## 9. A worked example, fully worked

f(x) = x · sin x at x = 1. The true derivative is f′(x) = sin x + x cos x, so f′(1) = sin 1 + cos 1 ≈ 0.8415 + 0.5403
= 1.3818, and f″(x) = 2 cos x − x sin x, so f″(1) ≈ 0.2391. We follow exactly what the code does.

```ts
import { grad } from 'aifn/autodiff'
import { mul, sin } from 'aifn/tensor'

const f = (x) => mul(x, sin(x))
grad(f)(1) // 1.3818…
grad(grad(f))(1) // 0.2391…
```

### 9.1 The two primitives involved

These are the definitions in `foundation/tensor/elementwise.ts`, lightly abbreviated. Each gives a forward rule and a
vjp rule `(g, …inputs) => cotangents`, where `g` is the cotangent arriving at the primitive's output:

```ts
sin = unaryPrimitive('foundation/tensor/sin', Math.sin, (g, x) => mul(g, cos(x))) // ∂ sin x / ∂x = cos x
mul = binaryPrimitive(
  'foundation/tensor/mul',
  (a, b) => a * b,
  (g, a, b) => [mul(g, b), mul(g, a)],
) // ∂ab/∂a = b, ∂ab/∂b = a
```

**Where the "vector–Jacobian product" comes from.** A primitive y = p(a, b, …) has a Jacobian: the matrix of partial
derivatives of its output with respect to each input. For `mul` with scalar inputs it is the 1×2 row
J = [∂y/∂a, ∂y/∂b] = [b, a]; for `sin` it is the 1×1 matrix [cos x]. The **vjp** of a cotangent g is the row vector
gᵀJ: one entry per input, each g times that input's partial. So `mul`'s rule returns [g·b, g·a] and `sin`'s returns
[g·cos x]. The rule never builds J as a matrix; it computes gᵀJ directly (for tensors that matters: see §9.5, where J
would be diagonal and huge).

### 9.2 Forward pass: what each call records

`grad(f)(1)` opens a fresh tape, records the input, and runs `f`. Each row below is one step of the code.

| Step | Code executed   | What `apply` does                                                                         | Tape after the step              | Returned         |
| ---- | --------------- | ----------------------------------------------------------------------------------------- | -------------------------------- | ---------------- |
| 1    | `tape.input(1)` | appends a leaf                                                                            | `n0 = input, value 1`            | x = ⟨n0, 1⟩      |
| 2    | `sin(x)`        | an input is traced → run `Math.sin(1)` = 0.8415 → `tape.record('sin', [x], 0.8415, rule)` | `n1 = sin(n0), value 0.8415`     | ⟨n1, 0.8415⟩     |
| 3    | `mul(x, ⟨n1⟩)`  | traced → run 1 · 0.8415 → record                                                          | `n2 = mul(n0, n1), value 0.8415` | y = ⟨n2, 0.8415⟩ |

(⟨nk, v⟩ is a traced value: node k, raw value v.) The tape is now three nodes in evaluation order. Each node stores
its inputs (pointers to earlier nodes), its forward value, and its vjp rule, with the rule's closure over nothing but
the primitive: the input and output values are passed in at sweep time.

### 9.3 Backward pass: the sweep, line by line

`y` is a scalar, so the seed is 1. `sweep(tape, y, 1, [x])`:

1. **Relevant range:** from the first input (n0) to the output (n2).
2. **Dependency pass** (forward over n0…n2): n0 is the input; n1 uses n0 → depends; n2 uses n0 and n1 → depends.
   Nothing here is a constant.
3. **Seed:** `cotangent = { n2: 1 }`.
4. **Walk backwards:**

| Visit | Node             | g arriving | vjp call                  | Values plugged in             | Returns     | Accumulate            | cotangent map after   |
| ----- | ---------------- | ---------- | ------------------------- | ----------------------------- | ----------- | --------------------- | --------------------- |
| a     | n2 = mul(n0, n1) | 1          | `(g, a, b) => [g·b, g·a]` | a = x = 1, b = sin 1 = 0.8415 | [0.8415, 1] | n0 += 0.8415; n1 += 1 | { n0: 0.8415, n1: 1 } |
| b     | n1 = sin(n0)     | 1          | `(g, x) => [g·cos x]`     | x = 1                         | [0.5403]    | n0 += 0.5403          | { n0: 1.3818, n1: 1 } |
| c     | n0 = input       | 1.3818     | –                         | –                             | –           | stop: n0 is the input | –                     |

5. **Collect:** the gradient is `cotangent[n0]` = **1.3818** = sin 1 + cos 1.

Each visit multiplies the cotangent arriving at a node by that node's local Jacobian, and **sends the products to the
node's inputs**. n0 received two contributions (0.8415 through `mul` directly, 0.5403 through `sin`) because x is used
twice; adding them is the multivariable chain rule.

### 9.4 The same thing as a product of Jacobians

Write the computation as z₀ = x, z₁ = sin z₀, z₂ = z₀ · z₁ = f. The chain rule over this graph is

∂f/∂x = ∂z₂/∂z₀ (direct) + ∂z₂/∂z₁ · ∂z₁/∂z₀ = z₁ + z₀ · cos z₀ = sin 1 + 1 · cos 1.

The sweep evaluates this **from the left**: it starts with ∂f/∂z₂ = 1, multiplies by the Jacobian of the last
operation (giving ∂f/∂z₀ partial = z₁ and ∂f/∂z₁ = z₀), then by the next one back (∂f/∂z₁ · cos z₀), adding at each
join. Left-to-right means every intermediate product is a row vector (the size of one value), never a matrix: that is
why the operation is a _vector_–Jacobian product, and why one backward pass gives the gradient with respect to every
input at once.

### 9.5 The same example with a tensor

f(x) = Σᵢ xᵢ sin xᵢ at x = [1, 2], written `sum(mul(x, sin(x)))`.

**Forward:**

| Node | op          | value            |
| ---- | ----------- | ---------------- |
| n0   | input       | [1, 2]           |
| n1   | sin(n0)     | [0.8415, 0.9093] |
| n2   | mul(n0, n1) | [0.8415, 1.8186] |
| n3   | sum(n2)     | 2.6601           |

**Backward** (seed 1 at n3):

| Visit            | vjp rule                                                                                      | Result                                                |
| ---------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| n3 = sum(n2)     | `g ↦ expand(g)`: g broadcast back to the shape of n2: every element contributed with weight 1 | n2 gets [1, 1]                                        |
| n2 = mul(n0, n1) | `[g·b, g·a]`, elementwise                                                                     | n0 += [1·0.8415, 1·0.9093]; n1 += [1·1, 1·2] = [1, 2] |
| n1 = sin(n0)     | `[g·cos x]`, elementwise                                                                      | n0 += [1·cos 1, 2·cos 2] = [0.5403, −0.8323]          |
| n0               | stop                                                                                          | **[1.3818, 0.0770]**                                  |

Check: f′ elementwise is sin xᵢ + xᵢ cos xᵢ = [0.8415 + 0.5403, 0.9093 − 0.8323]. ✓

The Jacobians here are matrices: J_sum is the 1×2 row [1, 1]; J_mul with respect to a is diag(b), with respect to b
is diag(a); J_sin is diag(cos x). A vjp against a diagonal Jacobian is an elementwise product, so the rules compute
`g * b` rather than building diag(b) and multiplying. For `sum`, gᵀJ_sum = g · [1, 1], which is that broadcast. With a
million elements the Jacobians would be 10⁶ × 10⁶; the rules never form them.

### 9.6 Second derivative: `grad(grad(f))(1)`, recorded and swept

The outer `grad` opens a tape and records x. The inner `grad` finds that tape active, shares it, and records its own
input as an identity of the outer one. Its forward pass is as in §9.2. Its sweep runs **on the tape** (it is nested),
so every multiplication inside a vjp rule is itself a primitive call on traced values, and gets recorded:

| Node | Recorded by                    | op                     | value  | meaning                      |
| ---- | ------------------------------ | ---------------------- | ------ | ---------------------------- |
| n0   | outer grad                     | input                  | 1      | x                            |
| n1   | inner grad                     | input (identity of n0) | 1      | the inner copy of x          |
| n2   | inner forward                  | sin(n1)                | 0.8415 | sin x                        |
| n3   | inner forward                  | mul(n1, n2)            | 0.8415 | f(x)                         |
| n4   | inner sweep, `mul`'s rule, g·b | mul(1, n2)             | 0.8415 | cotangent for n1 through mul |
| n5   | inner sweep, `mul`'s rule, g·a | mul(1, n1)             | 1      | cotangent for n2             |
| n6   | inner sweep, `sin`'s rule      | cos(n1)                | 0.5403 | cos x                        |
| n7   | inner sweep, `sin`'s rule      | mul(n5, n6)            | 0.5403 | cotangent for n1 through sin |
| n8   | inner sweep, accumulation      | add(n4, n7)            | 1.3818 | **f′(x)**, as a traced value |

The inner `grad` returns ⟨n8, 1.3818⟩: the first derivative, but traced, so the tape knows it as a function of x. The
outer `grad` now sweeps from n8 back to n0 (its rules run against the `NullTape`, recording nothing):

| Visit | Node                  | g                                  | Rule                                                | Contributions                 |
| ----- | --------------------- | ---------------------------------- | --------------------------------------------------- | ----------------------------- |
| a     | n8 = add(n4, n7)      | 1                                  | `[g, g]`                                            | n4 += 1; n7 += 1              |
| b     | n7 = mul(n5, n6)      | 1                                  | `[g·n6, g·n5]`                                      | n5 += cos 1 = 0.5403; n6 += 1 |
| c     | n6 = cos(n1)          | 1                                  | `[−g·sin x]`                                        | n1 += −0.8415                 |
| d     | n5 = mul(1, n1)       | 0.5403                             | `[g·n1, g·1]`, the constant 1 gets nothing          | n1 += 0.5403                  |
| e     | n4 = mul(1, n2)       | 1                                  | `[g·n2, g·1]`                                       | n2 += 1                       |
| f     | n3 = mul(n1, n2)      | none                               | not visited: n8 does not depend on f's value itself | –                             |
| g     | n2 = sin(n1)          | 1                                  | `[g·cos x]`                                         | n1 += 0.5403                  |
| h     | n1 = input (identity) | −0.8415 + 0.5403 + 0.5403 = 0.2391 | `[g]`                                               | n0 += 0.2391                  |

**f″(1) = 0.2391 = 2 cos 1 − sin 1.** ✓ The three contributions to n1 are the three places x appears in
f′(x) = sin x + x cos x once it is written out as the recorded nodes: through `sin` (n2 → cos 1), through the `x` in
`x cos x` (n5 → cos 1), and through the `cos` (n6 → −sin 1).

## 10. Composition: a gradient with no rule of its own

A function built from primitives needs no derivative rule of its own. Nobody writes one for it; the tape records its
primitives and the sweep chains their rules. This section shows that on a function with two parameters.

The **negative log-likelihood of one observation under a normal distribution**, dropping the constant ½ log 2π:

L(μ, σ) = ½ ((x − μ) / σ)² + log σ

```ts
import { grad } from 'aifn/autodiff'
import { add, div, log, mul, square, sub } from 'aifn/tensor'

// A plain function: not a primitive, no rule, nothing registered.
const nll = (mu, sigma, x) => add(mul(0.5, square(div(sub(x, mu), sigma))), log(sigma))

grad(nll, { argnums: [0, 1] })(1, 2, 3) // [∂L/∂μ, ∂L/∂σ] = [-0.5, 0]
```

The analytic answer, for checking: ∂L/∂μ = −(x − μ)/σ² and ∂L/∂σ = −(x − μ)²/σ³ + 1/σ. At x = 3, μ = 1, σ = 2 these
are −2/4 = −0.5 and −4/8 + 1/2 = 0. (σ = |x − μ| is where the likelihood is maximised in σ, hence the 0.)

**The six primitives it uses, and their rules** (from `foundation/tensor/elementwise.ts`):

| Primitive   | Forward | vjp rule            | In words                                  |
| ----------- | ------- | ------------------- | ----------------------------------------- |
| `sub(a, b)` | a − b   | `[g, −g]`           | a counts +1, b counts −1                  |
| `div(a, b)` | a / b   | `[g / b, −g·y / b]` | ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = −a/b² = −y/b |
| `square(x)` | x²      | `[g · 2x]`          |                                           |
| `mul(a, b)` | a·b     | `[g·b, g·a]`        |                                           |
| `log(x)`    | log x   | `[g / x]`           |                                           |
| `add(a, b)` | a + b   | `[g, g]`            |                                           |

**Forward pass** (x = 3 is a raw constant: it is not an argument being differentiated, so it is never traced):

| Node | op     | inputs  | value                                   |
| ---- | ------ | ------- | --------------------------------------- |
| n0   | input  | –       | μ = 1                                   |
| n1   | input  | –       | σ = 2                                   |
| n2   | sub    | 3, n0   | 3 − 1 = 2                               |
| n3   | div    | n2, n1  | 2 / 2 = 1 (the standardised residual z) |
| n4   | square | n3      | 1                                       |
| n5   | mul    | 0.5, n4 | 0.5                                     |
| n6   | log    | n1      | log 2 = 0.6931                          |
| n7   | add    | n5, n6  | L = 1.1931                              |

**Backward sweep**, seed 1 at n7:

| Visit | Node              | g                 | Rule applied                                | Contributions         |
| ----- | ----------------- | ----------------- | ------------------------------------------- | --------------------- |
| a     | n7 = add(n5, n6)  | 1                 | `[g, g]`                                    | n5 += 1; n6 += 1      |
| b     | n6 = log(n1)      | 1                 | `[g / x]`, x = 2                            | n1 += 0.5             |
| c     | n5 = mul(0.5, n4) | 1                 | `[g·b, g·a]`: the constant 0.5 gets nothing | n4 += 1 · 0.5 = 0.5   |
| d     | n4 = square(n3)   | 0.5               | `[g · 2x]`, x = 1                           | n3 += 0.5 · 2 = 1     |
| e     | n3 = div(n2, n1)  | 1                 | `[g / b, −g·y / b]`, b = 2, y = 1           | n2 += 0.5; n1 += −0.5 |
| f     | n2 = sub(3, n0)   | 0.5               | `[g, −g]`: the constant 3 gets nothing      | n0 += −0.5            |
| g     | n1 = input σ      | 0.5 − 0.5 = **0** | stop                                        |                       |
| h     | n0 = input μ      | **−0.5**          | stop                                        |                       |

The gradient [−0.5, 0] matches the analytic answer, and no one wrote ∂L/∂μ or ∂L/∂σ. σ received two contributions
that cancel: +0.5 through `log σ` (the normaliser pushes σ down) and −0.5 through the residual `(x − μ)/σ` (the fit
term pushes σ up). The sweep found both paths, as the chain rule requires.

**When to stop composing and write a rule instead.** The composite is exact and costs a few recorded nodes, which is
right here. A hand-written rule for the whole function pays off only for the reasons in the discussion of shortcuts:
far fewer nodes (a Cholesky factorisation rather than its loops), a stabler formula than the traced one (softplus's
`g · sigmoid(x)` rather than differentiating `log(1 + eˣ)`), or less memory.

aifn's `Normal.logProb` is built the same way: it standardises z = (x − μ)/σ with primitives, applies the
`normalLogPdf` primitive from `special`, and subtracts `log σ`, so it differentiates with respect to both parameters
and the value without a rule of its own.
