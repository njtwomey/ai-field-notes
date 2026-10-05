# How `grad` works in aifn

> **Historical (2026-10-05):** the engine no longer lives in this repository. It is the aifn-engine project
> (https://github.com/njtwomey/aifn-engine, docs at https://njtwomey.github.io/aifn-engine/), installed here as the
> released packages `aifn-compute`, `aifn-methods` and `aifn-render`. Paths below under `aifn-js/core`, `aifn-js/methods`
> and `aifn-js/render` refer to the old in-tree copy (now `packages/compute`, `packages/methods` and `packages/render`
> in the engine); the lab moved from `aifn-js/sandbox/lab` to `lab/` here.

This explains automatic differentiation as aifn implements it, step by step, with worked examples. The code lives
under `aifn-js/core/src/foundation/`: `tensor/trace.ts` (the hook every primitive goes through), `tensor/primitive.ts`
(primitives and their rules), and in `autodiff/`: `reverse.ts` (the reverse interpreter: records and the backward
sweep), `forward.ts` (the forward interpreter: dual numbers), `batch.ts` (the batch interpreter: `vmap`) and
`transforms.ts` (`grad` and friends), `custom.ts` (custom rules and checkpointing). Differentiation through solvers
sits higher in the tree: `aifn/numerics/implicit` (implicit differentiation), `aifn/foundation/trace`'s `unrolled`
(through an `Algorithm`'s steps) and `aifn/dynamics/ode`'s `odeAdjoint`. §8 records the state of the interpreters, §11
covers custom rules and solvers, and §12 covers complex numbers (differentiated as pairs of reals).

## 1. The idea in one paragraph

`grad(f)(x)` runs `f` once, forwards, on a stand-in for `x` called a **tracer**. Every primitive operation `f`
performs on a tracer (add, multiply, sin, matmul, …) is handed to the tracer's **interpreter**, which computes the
result and, for `grad`, **records** the operation, in the order it happens. Then it walks the records **backwards**,
starting from the output with the value 1, and asks each recorded operation "given how much the output changes per
unit change of your result, how much does it change per unit change of each of your inputs?". That question is
answered by the operation's **vector–Jacobian product (vjp)**, written (or derived) next to the operation when it is
defined. Contributions that reach the same value along different paths are added. When the walk reaches `x`, the
accumulated number is ∂f/∂x.

This is the chain rule, applied mechanically in reverse order: for y = f(g(x)), dy/dx = f′(g(x)) · g′(x), and the
reverse walk multiplies the outer factor first. `jvp` applies the same chain rule in forward order, and `vmap` uses
the same machinery to run a function on a whole batch; §7 covers both.

## 2. Values and tracers

Every numeric function in aifn accepts and returns a `Value`:

| Kind         | Example                                 | What a primitive does with it                                           |
| ------------ | --------------------------------------- | ----------------------------------------------------------------------- |
| a number     | `2`                                     | computes and returns a number                                           |
| a tensor     | `tensor([1, 2])`                        | computes elementwise or structurally and returns a tensor               |
| a **tracer** | `ReverseTracer { value: 2, record: 7 }` | hands the operation to the tracer's interpreter, which returns a tracer |

A tracer belongs to one **interpreter**, the one run by the transform in progress, and there are three kinds:

| Tracer          | Holds                                             | Made by                                             |
| --------------- | ------------------------------------------------- | --------------------------------------------------- |
| `ReverseTracer` | a value and the record that produced it           | `grad`, `valueAndGrad`, `vjp`, `jacobian` (reverse) |
| `ForwardTracer` | a primal value and its tangent (a dual number)    | `jvp`, `hvp`, `linearize`, `jacobian` (forward)     |
| `BatchTracer`   | a value with an extra batch axis, and where it is | `vmap`, `jacobian`                                  |

Every tracer has an **aval** (abstract value): the shape and dtype of the value the traced function sees, and whether
it is a number or a tensor. Your function never sees the difference: `sin(x)` works the same whether `x` is a number or
a tracer. That is why `f` needs no special code to be differentiable. `unwrap(x)` reads the concrete value behind a
tracer (following it down through every level); inside `vmap` there is none (a batch tracer holds a whole batch), so
code that must work under `vmap` computes with primitives rather than reading values.

## 3. Primitives: each operation defined once, with its rules

A **primitive** is an operation with its forward rule and the rules the transforms need, declared together
(`elementwise`, `definePrimitive` or the shorter `defineOp` in `tensor/primitive.ts`):

| Rule        | Says                                                              | Used by                 |
| ----------- | ----------------------------------------------------------------- | ----------------------- |
| `impl`      | the result, from raw numbers or tensors                           | everything              |
| `vjp`       | given the output's cotangent g, one cotangent per input (gᵀJ)     | the reverse interpreter |
| `jvp`       | given one tangent per input, the output's tangent (J·t)           | the forward interpreter |
| `transpose` | for a linear primitive, the adjoint map (it gives the vjp)        | the reverse interpreter |
| `batch`     | how to apply the primitive to batched inputs in one call          | the batch interpreter   |
| `shape`     | the output's shape, dtype and kind from the inputs', without data | the batch interpreter   |

How much the author writes depends on the kind of primitive:

- **Elementwise** (`elementwise({ id, f, derivative })`): a scalar rule `f` and **one derivative per argument**, written
  with primitives. Everything else is derived: vjp gᵢ = g·∂ᵢ (summed back over broadcast axes), jvp ṫ = Σᵢ tᵢ·∂ᵢ,
  batching by broadcasting, and the shape rule of broadcasting. A derivative may be `'zero'` (piecewise constant in
  that argument: comparisons, `sign`, `where`'s condition) or `null` (not differentiable: an error if reached).
- **Linear** (`linear: 'linear'`, e.g. `reshape`, `sum`): a `transpose`. The vjp is the transpose, and the jvp is the
  primitive applied to the tangents. **Multilinear** (`linear: 'multilinear'`, e.g. `matmul`): a transpose in each
  argument; the jvp is Σᵢ p(…, tᵢ, …), e.g. Ȧ B + A Ḃ.
- **Other** (e.g. `logsumexp`, `cholesky`): a `vjp` and a `jvp`, and optionally `batch` and `shape`.
- **Piecewise constant** (`zeroDerivative`: comparisons, `sign`, `stopGradient`): every derivative is zero, so the
  derivative interpreters compute the output and treat it as a constant.

For example, `exp` is declared as

```ts
exp = elementwise({ id: 'foundation/tensor/exp', f: Math.exp, derivative: [(_x, y) => y] }) // d/dx eˣ = eˣ = y
```

and multiplication as `derivative: [(a, b) => b, (a, b) => a]`: ∂(ab)/∂a = b and ∂(ab)/∂b = a.

**Missing rules are not errors.** A primitive with a vjp but no jvp still works in forward mode: the forward
interpreter uses the _transpose trick_ for that one application (§7.1). A primitive without a batching rule still
works under `vmap`: the batch interpreter loops over the examples and stacks the results (§7.2). `registry.list()`
reports, for each primitive, which rules are its own, derived or missing (`rules`).

**Every call goes through `apply`** (`tensor/trace.ts`):

1. If **no input is a tracer**, it just runs `impl` and returns the raw result. This is the fast path: most of aifn
   runs this way, with no transform at all (§8 lists the shortcuts in front of it).
2. Otherwise it finds the input tracer with the **highest level** and calls its interpreter's `process(p, inputs,
params)`. Each transform takes a new level when it starts, so an inner transform has a higher level than the
   transforms around it. The interpreter treats tracers of lower levels as constants: it replaces its own tracers by
   the values they stand for (**lowering** them), computes the result by calling `apply` again on the lowered inputs
   (so lower levels see the operation too), and wraps the result in a tracer of its own.

Two details matter for correctness:

- **Broadcasting.** If `a` has shape [3] and `b` is a number, `add(a, b)` broadcasts `b`. In the backward pass the
  cotangent for `b` must be summed back to a number. `sumLike(g, like)` does that reduction; the derived elementwise
  rules apply it automatically.
- **Rules are written with primitives.** `mul(g, y)` inside a rule is itself a primitive call. If the rule runs on
  tracers of an enclosing transform, that transform traces it too. This is what makes second derivatives possible (§6).

Every registered primitive lives in one registry (`tensor/registry.ts`), keyed `module/name`, and registering the same
id twice throws.

## 4. The reverse interpreter's records

The reverse interpreter (`autodiff/reverse.ts`, class `ReverseInterpreter`) keeps an array of records in evaluation
order:

```ts
type TapeRecord = {
  primitive: Primitive | null // null for an input leaf of the transform
  params: unknown
  inputs: readonly Value[] // lowered: this interpreter's tracers replaced by their values
  sources: readonly number[] // for each input, the record it came from, or −1 for a constant
  output: Value // lowered
  label?: string // for inputs: the leaf's path in the argument tree
}
```

`process(p, inputs, params)` lowers the inputs, computes `apply(p, lowered, params)`, appends a record and returns a
`ReverseTracer` pointing at it. Because records are appended as the computation runs, **the order of the array is a
topological order** of the computation graph: every record comes after the records it depends on. So walking the array
backwards visits each record after everything that uses it. No explicit graph search is needed. Every record depends
on an input of the transform (only this interpreter's tracers produce records), so nothing needs pruning.

## 5. The flow of `grad(f)(x)`, step by step

`grad` is `valueAndGrad` keeping only the gradient (`autodiff/transforms.ts`).

1. **Start an interpreter.** Create a `ReverseInterpreter`; it takes the next level.
2. **Record the inputs.** `x` may be a number, a tensor or a pytree (nested arrays and objects of them). It is
   flattened into leaves, and each leaf becomes an input record and is replaced by its tracer. `argnums` picks which
   arguments are differentiated; the others stay constants.
3. **Run `f` forwards.** Every primitive touching one of the tracers is recorded (§4). The result `y` is a tracer.
4. **Check the output** is a scalar (a number or a rank-0 tensor): a gradient is defined for scalar functions. The
   seed cotangent is 1.
5. **Backward sweep** (`backward` in `autodiff/reverse.ts`):
   1. **Seed** the output record's cotangent with 1.
   2. **Walk backwards** from the output record to the first input. For each record that has a cotangent `g`:
      - if it is an input record, stop there: its cotangent is the answer for that input;
      - if its primitive has no vjp, throw `NotDifferentiableError` (never a silent 0);
      - otherwise call `vjp(g, inputs, output, params, needed)` on the **lowered** values, where `needed` marks the
        inputs that came from records (the others are constants, and a rule may skip them), and **add** each
        cotangent into that input's record (a value used twice receives two contributions).
6. **Collect.** Each input's total cotangent is its gradient. Inputs the output never depended on get zeros of their
   own shape. The leaves are rebuilt into the argument's original structure, so a gradient has the same shape (or
   pytree shape) as its argument.

Cost: one forward pass plus one backward pass, whatever the number of inputs. That is why reverse mode suits
functions of many parameters with one scalar output (losses).

The rules run on lowered values. At the outermost level those are plain numbers and tensors, so a first-order
backward sweep computes and records nothing. Inside another transform they are that transform's tracers, so the
enclosing transform sees the backward computation (§6).

## 6. Nesting by level: second derivatives

`grad(grad(f))(x)` needs no special case:

- The **outer** `grad` starts interpreter R1 (level 1) and records `x` as an input: x₁, a level-1 tracer.
- The **inner** `grad` starts R2 (level 2) and records x₁ as _its_ input: x₂, whose value is x₁.
- Running `f(x₂)`: each primitive sees a level-2 tracer, so R2 processes it. R2 lowers x₂ to x₁ and calls `apply`
  again, which reaches R1 (x₁ is level 1). R1 lowers x₁ to the number and computes. **Both** interpreters record the
  operation, each in its own records.
- The **inner** backward sweep runs R2's rules on R2's lowered values, which are level-1 tracers. So every
  multiplication inside a rule is processed by R1 and **recorded by R1**. The inner gradient is therefore a level-1
  tracer: a function of x that R1 knows how to differentiate.
- The **outer** backward sweep differentiates R1's records, which include the inner backward computation, giving the
  second derivative. Its rules run on raw values and record nothing.

Levels also rule out **perturbation confusion** (Siskind and Pearlmutter, 2005). In d/dx [x · d/dy (x + y)], the inner
transform must treat x as a constant even though x is being differentiated by the outer one. x is a level-1 tracer and
the inner transform works at level 2, so to it x is a constant, and the answer is 1, not 2.

Any order of nesting works the same way: `jvp(grad(f))` (the Hessian–vector product), `vmap(grad(f))` (per-example
gradients), `grad` of a function that calls `vmap`, and so on.

## 7. The other interpreters, the transforms, and inspection

### 7.1 Forward mode: dual numbers (`autodiff/forward.ts`)

`jvp(f, x, v)` starts a `ForwardInterpreter` and runs `f` on `ForwardTracer`s carrying the primal `x` and the tangent
`v` (the direction). `process(p, inputs, params)`:

1. **Split** each input into its primal and its tangent (null, a symbolic zero, for constants and lower-level tracers).
2. **Compute the primal** output: `apply(p, primals, params)`.
3. If every tangent is null (or `p` is piecewise constant), return the primal output: its tangent is zero.
4. **Compute the tangent** output with `p.jvp(tangents, primals, output, params)`, and return a tracer of both.

One pass computes f(x) and J·v together. The result's tangent is read off the output tracer; outputs that never met a
tracer get zero tangents.

**The transpose trick** covers a primitive with a vjp but no jvp, for that one application: its vjp u ↦ Jᵀu is linear
in u, so J·t is the gradient in u of ⟨Jᵀu, t⟩. The forward interpreter starts a small reverse interpreter, runs the
primitive's vjp on a traced u, forms ⟨Jᵀu, t⟩ and sweeps back once to u. The answer is exact; the cost is one extra
rule evaluation and one sweep for that node.

### 7.2 Batching: `vmap` (`autodiff/batch.ts`)

`vmap(f)(xs)` starts a `BatchInterpreter` and runs `f` on `BatchTracer`s whose value is `xs` and whose batch axis is
`inAxes` (default 0). The function sees one example: the aval of a batch tracer is its value without the batch axis.
`process(p, inputs, params)`:

1. **Split** each input into its value and its batch axis (null for inputs that are not batched).
2. With a **batching rule**, call `p.batch(values, axes, params, size)`: it applies the primitive once to the whole
   batch and says where the output's batch axis is. Elementwise primitives batch by moving each batch axis to the
   front, inserting axes of length 1 so that examples of different ranks line up, and broadcasting.
3. Without one, **loop**: apply the primitive to each example (the batch axis indexed away) and stack the results.
4. Return a batch tracer of the output.

At the end, every output leaf gets its batch axis moved to `outAxes` (default 0); an output that does not depend on the
batch is broadcast to every example.

### 7.3 The transforms

| Transform                      | How it works                                                                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `valueAndGrad(f)`              | as §5, returning `{ value, grad }` from the same passes                                                                                                                                                         |
| `vjp(f, x)`                    | records one forward pass and returns a `pullback(u)` that runs a backward sweep seeded with `u` instead of 1 (outputs may be pytrees)                                                                           |
| `jvp(f, x, v)`                 | one forward pass with dual numbers (§7.1); x, v and f(x) may be pytrees                                                                                                                                         |
| `linearize(f, x)`              | f(x) and the map v ↦ J·v; each call is one `jvp` pass                                                                                                                                                           |
| `hvp(f, x, v)`                 | forward over reverse: `jvp(grad(f), x, v)`, one forward-mode pass through one gradient evaluation                                                                                                               |
| `jacobian(f, { mode })`        | reverse: `vmap` of the pullback over the one-hot cotangents; forward: `vmap` of `jvp` over the one-hot tangents; `auto` picks forward when x has no more elements than f(x). Pytrees in and out                 |
| `hessian(f)`                   | forward over reverse: the forward-mode Jacobian of `grad(f)`                                                                                                                                                    |
| `vmap(f, { inAxes, outAxes })` | §7.2                                                                                                                                                                                                            |
| `stopGradient(x)`              | a piecewise-constant identity primitive: the value passes, the derivative does not                                                                                                                              |
| `gradCheck(f, x)`              | compares `grad` with central finite differences and reports the error                                                                                                                                           |
| `traceGraph(f, x)`             | runs the forward pass and a backward sweep keeping every cotangent, returning every record with its value, its adjoint and the local partials: the data behind the lab's "Backpropagation, node by node" figure |

**Result types.** A transform's result is raw when nothing it computes on is traced. Inside another transform, or when
`f` closes over a traced value, it is traced by the enclosing transform. The types say so at every level: `grad(f)(x)`
with `x: Tensor` has type `Tensor | Traced` (`Lifted<T>`), and a Jacobian of pytrees is a `TreeOf` trees of values. At
the outermost level, narrow with `as` or read through `unwrap`.

## 8. The interpreters today (design K §4)

Status (2026-10-01): the redesign is complete (phases 2a–2d).

- **Three interpreters** over one primitive registry: _reverse_ (`autodiff/reverse.ts`: records and a backward
  sweep), _forward_ (`autodiff/forward.ts`: dual numbers, so `jvp` is one pass) and _batch_ (`autodiff/batch.ts`:
  `vmap`). They nest by level, as in JAX. Each implements `process(p, inputs, params)`; `apply` hands an application
  to the interpreter of the highest-level tracer among the inputs.
- **Rules.** The primitive spec carries `impl`, `vjp`, `jvp`, `transpose` (linear primitives), `batch` and `shape`.
  Elementwise primitives derive all of them from one derivative per argument; linear primitives derive their vjp and
  jvp from their transpose. Every general primitive has its own jvp, batch and shape rule (phase 2b), including
  Fourier and convolution; the linear-algebra primitives (`cholesky`, `lu` and `luSolve` (under `solve`), `det`,
  `logDet`, `eigh`, `svd`, `qr`, `expm`) have derivative rules of their own (phase 2d). The generated primitive suite checks every rule against
  finite differences and against each other.
- **Transforms.** `hvp` and `hessian` are forward over reverse; `jacobian` takes pytrees in and out and is built on
  `vmap`; `linearize`, `vmap`, custom rules and `checkpoint` (§11) complete the set.
- **The fast paths.** Code that is not being differentiated pays almost nothing for the machinery:
  - `apply` scans the inputs for a tracer; with none it calls `impl` directly, with no interpreter and no record.
  - An elementwise primitive called with plain numbers skips `apply` too: the function `elementwise` returns checks
    `typeof x === 'number'` and calls the scalar rule `f` (numbers are never traced). Only a complex result goes
    through `apply`.
  - `matmul` on concrete, contiguous float64 operands (a matrix times a matrix or a vector) calls the `dense` kernels
    on zero-copy views, skipping the view arithmetic and the primitive dispatch that dominate for the small matrices
    of filters and per-step recursions. Concrete tensors are constants to every transform, so the result is the value
    the primitive would give.
  - `linearCombination(xs, coefficients)` (Σᵢ cᵢ xᵢ) is one variadic linear primitive, so a Runge–Kutta stage or an optimiser
    update is one operation rather than a chain of `add` and `mul`. This is what keeps traced iterative algorithms
    (§11.3) close to the speed of their raw form; `make bench` reports both.

## 9. A worked example, fully worked

f(x) = x · sin x at x = 1. The true derivative is f′(x) = sin x + x cos x, so f′(1) = sin 1 + cos 1 ≈ 0.8415 + 0.5403
= 1.3818, and f″(x) = 2 cos x − x sin x, so f″(1) ≈ 0.2391. We follow exactly what the code does.

```ts
import { grad, mul, sin } from 'aifn' // foundation's common surface; or 'aifn/foundation/autodiff' and '…/tensor'

const f = (x) => mul(x, sin(x))
grad(f)(1) // 1.3818…
grad(grad(f))(1) // 0.2391…
```

### 9.1 The two primitives involved

These are the definitions in `foundation/tensor/elementwise.ts`, lightly abbreviated. Each gives a scalar forward rule
and one derivative per argument, written with primitives:

```ts
sin = elementwise({ id: 'foundation/tensor/sin', f: Math.sin, derivative: [(x) => cos(x)] }) // ∂ sin x / ∂x = cos x
mul = elementwise({ id: 'foundation/tensor/mul', f: (a, b) => a * b, derivative: [(a, b) => b, (a, b) => a] })
```

The vjp rule is derived from the derivatives: given `g`, the cotangent arriving at the primitive's output, input i
receives g·∂ᵢ. So `sin`'s vjp returns [g·cos x] and `mul`'s returns [g·b, g·a]. (When g is the number 1, the seed, the
multiplication is skipped and the derivative itself is the cotangent.)

**Where the "vector–Jacobian product" comes from.** A primitive y = p(a, b, …) has a Jacobian: the matrix of partial
derivatives of its output with respect to each input. For `mul` with scalar inputs it is the 1×2 row
J = [∂y/∂a, ∂y/∂b] = [b, a]; for `sin` it is the 1×1 matrix [cos x]. The **vjp** of a cotangent g is the row vector
gᵀJ: one entry per input, each g times that input's partial. So `mul`'s rule returns [g·b, g·a] and `sin`'s returns
[g·cos x]. The rule never builds J as a matrix; it computes gᵀJ directly (for tensors that matters: see §9.5, where J
would be diagonal and huge).

### 9.2 Forward pass: what each call records

`grad(f)(1)` starts a reverse interpreter R, records the input, and runs `f`. Each row below is one step of the code.

| Step | Code executed  | What happens                                                                                            | Records after the step           | Returned         |
| ---- | -------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------- |
| 1    | `R.input(1)`   | appends an input record                                                                                 | `n0 = input, value 1`            | x = ⟨n0, 1⟩      |
| 2    | `sin(x)`       | `apply` finds a tracer of R → `R.process`: lower x to 1, `apply(sin, [1])` = 0.8415 (fast path), record | `n1 = sin(n0), value 0.8415`     | ⟨n1, 0.8415⟩     |
| 3    | `mul(x, ⟨n1⟩)` | → `R.process`: lower to [1, 0.8415], `apply(mul, …)` = 0.8415, record                                   | `n2 = mul(n0, n1), value 0.8415` | y = ⟨n2, 0.8415⟩ |

(⟨nk, v⟩ is a tracer: record k, value v.) R now holds three records in evaluation order. Each stores its primitive,
its lowered inputs, where each came from (n0, n1, or −1 for a constant) and its lowered output: everything the rule
needs at sweep time.

### 9.3 Backward pass: the sweep, line by line

`y` is a scalar, so the seed is 1. `R.backward([y], [1], [x])`:

1. **Seed:** `cotangent = { n2: 1 }`.
2. **Walk backwards** from n2 to n0:

| Visit | Record           | g arriving | vjp call     | Values plugged in             | Returns     | Accumulate            | cotangent map after   |
| ----- | ---------------- | ---------- | ------------ | ----------------------------- | ----------- | --------------------- | --------------------- |
| a     | n2 = mul(n0, n1) | 1          | `[g·b, g·a]` | a = x = 1, b = sin 1 = 0.8415 | [0.8415, 1] | n0 += 0.8415; n1 += 1 | { n0: 0.8415, n1: 1 } |
| b     | n1 = sin(n0)     | 1          | `[g·cos x]`  | x = 1                         | [0.5403]    | n0 += 0.5403          | { n0: 1.3818, n1: 1 } |
| c     | n0 = input       | 1.3818     | –            | –                             | –           | stop: n0 is the input | –                     |

3. **Collect:** the gradient is `cotangent[n0]` = **1.3818** = sin 1 + cos 1.

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

| Record | op          | value            |
| ------ | ----------- | ---------------- |
| n0     | input       | [1, 2]           |
| n1     | sin(n0)     | [0.8415, 0.9093] |
| n2     | mul(n0, n1) | [0.8415, 1.8186] |
| n3     | sum(n2)     | 2.6601           |

**Backward** (seed 1 at n3):

| Visit            | vjp rule                                                                                                                | Result                                                |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| n3 = sum(n2)     | `sum` is linear; its vjp is its transpose, g broadcast back to the shape of n2: every element contributed with weight 1 | n2 gets [1, 1]                                        |
| n2 = mul(n0, n1) | `[g·b, g·a]`, elementwise                                                                                               | n0 += [1·0.8415, 1·0.9093]; n1 += [1·1, 1·2] = [1, 2] |
| n1 = sin(n0)     | `[g·cos x]`, elementwise                                                                                                | n0 += [1·cos 1, 2·cos 2] = [0.5403, −0.8323]          |
| n0               | stop                                                                                                                    | **[1.3818, 0.0770]**                                  |

Check: f′ elementwise is sin xᵢ + xᵢ cos xᵢ = [0.8415 + 0.5403, 0.9093 − 0.8323]. ✓

The Jacobians here are matrices: J_sum is the 1×2 row [1, 1]; J_mul with respect to a is diag(b), with respect to b
is diag(a); J_sin is diag(cos x). A vjp against a diagonal Jacobian is an elementwise product, so the rules compute
`g * b` rather than building diag(b) and multiplying. For `sum`, gᵀJ_sum = g · [1, 1], which is that broadcast. With a
million elements the Jacobians would be 10⁶ × 10⁶; the rules never form them.

### 9.6 Second derivative: `grad(grad(f))(1)`, two interpreters

The outer `grad` starts R1 (level 1) and records x as its input n0. The inner `grad` starts R2 (level 2) and records
⟨n0⟩ as its own input. Its forward pass is as in §9.2, except that every value R2 records is itself a level-1 tracer,
so R1 records the same `sin` and `mul`. R2's backward sweep runs its rules on those level-1 tracers, so R1 records the
backward computation too. R1's records, in order:

| Record | Recorded while                     | op          | value  | meaning                                                |
| ------ | ---------------------------------- | ----------- | ------ | ------------------------------------------------------ |
| n0     | outer grad, input                  | input       | 1      | x                                                      |
| n1     | inner forward, `sin(x)`            | sin(n0)     | 0.8415 | sin x                                                  |
| n2     | inner forward, `mul(x, sin x)`     | mul(n0, n1) | 0.8415 | f(x)                                                   |
| n3     | inner sweep, `sin`'s rule, cos x   | cos(n0)     | 0.5403 | ∂ sin x / ∂x                                           |
| n4     | inner sweep, `sin`'s rule, g·cos x | mul(n0, n3) | 0.5403 | cotangent for x through sin (g = x, from `mul`'s rule) |
| n5     | inner sweep, accumulation          | add(n1, n4) | 1.3818 | **f′(x)**, as a level-1 tracer                         |

The inner sweep starts with seed 1 on its `mul`, so `mul`'s rule returns its derivatives [b, a] = [⟨n1⟩, ⟨n0⟩]
without multiplying (nothing is recorded); `sin`'s rule then computes g · cos x = ⟨n0⟩ · cos⟨n0⟩ (n3, n4); and the
two cotangents reaching the inner x are added (n5). The inner `grad` returns ⟨n5, 1.3818⟩: the first derivative, but
traced, so R1 knows it as a function of x. The outer `grad` now sweeps R1 from n5 back to n0 (its rules run on raw
values and record nothing):

| Visit | Record           | g                                 | Rule                                                | Contributions         |
| ----- | ---------------- | --------------------------------- | --------------------------------------------------- | --------------------- |
| a     | n5 = add(n1, n4) | 1                                 | `[g, g]`                                            | n1 += 1; n4 += 1      |
| b     | n4 = mul(n0, n3) | 1                                 | `[g·n3, g·n0]`                                      | n0 += 0.5403; n3 += 1 |
| c     | n3 = cos(n0)     | 1                                 | `[−g·sin x]`                                        | n0 += −0.8415         |
| d     | n2 = mul(n0, n1) | none                              | not visited: n5 does not depend on f's value itself | –                     |
| e     | n1 = sin(n0)     | 1                                 | `[g·cos x]`                                         | n0 += 0.5403          |
| f     | n0 = input       | 0.5403 − 0.8415 + 0.5403 = 0.2391 | stop                                                | –                     |

**f″(1) = 0.2391 = 2 cos 1 − sin 1.** ✓ The three contributions to x are the three places x appears in
f′(x) = sin x + x cos x once it is written out as the recorded operations: through `sin` (n1 → cos 1), through the
`x` in `x cos x` (n4 → cos 1), and through the `cos` (n3 → −sin 1).

### 9.7 The same derivative forwards: `jvp(f, 1, 1)`

`jvp` pushes the tangent 1 (the direction) forward alongside the value, one operation at a time. The forward
interpreter F wraps x as the dual number ⟨1, ṫ = 1⟩. Each step computes the value and applies the primitive's jvp,
which for an elementwise primitive is Σᵢ tᵢ·∂ᵢ:

| Step | Code          | Value          | Tangent (jvp rule)                                        |
| ---- | ------------- | -------------- | --------------------------------------------------------- |
| 1    | x             | 1              | 1                                                         |
| 2    | `sin(x)`      | sin 1 = 0.8415 | ṫₓ · cos x = 1 · 0.5403 = 0.5403                          |
| 3    | `mul(x, sin)` | 0.8415         | ṫₓ · sin x + ṫ_sin · x = 1 · 0.8415 + 0.5403 · 1 = 1.3818 |

The result is `{ value: 0.8415, tangent: 1.3818 }` = f(1) and f′(1), in the one forward pass, with nothing recorded.
Forward mode costs one pass per _input_ direction, so it suits functions with few inputs; reverse mode costs one pass
per _output_, so it suits losses. `jacobian` picks between them by comparing the sizes, and `hvp` and `hessian` use
both: forward over reverse.

## 10. Composition: a gradient with no rule of its own

A function built from primitives needs no derivative rule of its own. Nobody writes one for it; the reverse
interpreter records its primitives and the sweep chains their rules. This section shows that on a function with two parameters.

The **negative log-likelihood of one observation under a normal distribution**, dropping the constant ½ log 2π:

L(μ, σ) = ½ ((x − μ) / σ)² + log σ

```ts
import { add, div, grad, log, mul, square, sub } from 'aifn'

// A plain function: not a primitive, no rule, nothing registered.
const nll = (mu, sigma, x) => add(mul(0.5, square(div(sub(x, mu), sigma))), log(sigma))

grad(nll, { argnums: [0, 1] })(1, 2, 3) // [∂L/∂μ, ∂L/∂σ] = [-0.5, 0]
```

The analytic answer, for checking: ∂L/∂μ = −(x − μ)/σ² and ∂L/∂σ = −(x − μ)²/σ³ + 1/σ. At x = 3, μ = 1, σ = 2 these
are −2/4 = −0.5 and −4/8 + 1/2 = 0. (σ = |x − μ| is where the likelihood is maximised in σ, hence the 0.)

**The six primitives it uses, and their rules** (from `foundation/tensor/elementwise.ts`; each declares its
derivatives, and the vjp multiplies the arriving cotangent g by them):

| Primitive   | Forward | Derivatives | vjp rule (derived)  | In words                                  |
| ----------- | ------- | ----------- | ------------------- | ----------------------------------------- |
| `sub(a, b)` | a − b   | [1, −1]     | `[g, −g]`           | a counts +1, b counts −1                  |
| `div(a, b)` | a / b   | [1/b, −y/b] | `[g / b, −g·y / b]` | ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = −a/b² = −y/b |
| `square(x)` | x²      | [2x]        | `[g · 2x]`          |                                           |
| `mul(a, b)` | a·b     | [b, a]      | `[g·b, g·a]`        |                                           |
| `log(x)`    | log x   | [1/x]       | `[g / x]`           |                                           |
| `add(a, b)` | a + b   | [1, 1]      | `[g, g]`            |                                           |

**Forward pass** (x = 3 is a raw constant: it is not an argument being differentiated, so it is never traced):

| Record | op     | inputs  | value                                   |
| ------ | ------ | ------- | --------------------------------------- |
| n0     | input  | –       | μ = 1                                   |
| n1     | input  | –       | σ = 2                                   |
| n2     | sub    | 3, n0   | 3 − 1 = 2                               |
| n3     | div    | n2, n1  | 2 / 2 = 1 (the standardised residual z) |
| n4     | square | n3      | 1                                       |
| n5     | mul    | 0.5, n4 | 0.5                                     |
| n6     | log    | n1      | log 2 = 0.6931                          |
| n7     | add    | n5, n6  | L = 1.1931                              |

**Backward sweep**, seed 1 at n7:

| Visit | Record            | g                 | Rule applied                                                                   | Contributions         |
| ----- | ----------------- | ----------------- | ------------------------------------------------------------------------------ | --------------------- |
| a     | n7 = add(n5, n6)  | 1                 | `[g, g]`                                                                       | n5 += 1; n6 += 1      |
| b     | n6 = log(n1)      | 1                 | `[g / x]`, x = 2                                                               | n1 += 0.5             |
| c     | n5 = mul(0.5, n4) | 1                 | `[g·b, g·a]`: the constant 0.5 is not needed, so its cotangent is not computed | n4 += 1 · 0.5 = 0.5   |
| d     | n4 = square(n3)   | 0.5               | `[g · 2x]`, x = 1                                                              | n3 += 0.5 · 2 = 1     |
| e     | n3 = div(n2, n1)  | 1                 | `[g / b, −g·y / b]`, b = 2, y = 1                                              | n2 += 0.5; n1 += −0.5 |
| f     | n2 = sub(3, n0)   | 0.5               | `[g, −g]`: the constant 3 gets nothing                                         | n0 += −0.5            |
| g     | n1 = input σ      | 0.5 − 0.5 = **0** | stop                                                                           |                       |
| h     | n0 = input μ      | **−0.5**          | stop                                                                           |                       |

The gradient [−0.5, 0] matches the analytic answer, and no one wrote ∂L/∂μ or ∂L/∂σ. σ received two contributions
that cancel: +0.5 through `log σ` (the normaliser pushes σ down) and −0.5 through the residual `(x − μ)/σ` (the fit
term pushes σ up). The sweep found both paths, as the chain rule requires.

**When to stop composing and write a rule instead.** The composite is exact and costs a few records, which is
right here. A hand-written rule for the whole function pays off only for the reasons in the discussion of shortcuts:
far fewer records (a Cholesky factorisation rather than its loops), a stabler formula than the traced one (softplus's
`g · sigmoid(x)` rather than differentiating `log(1 + eˣ)`), or less memory.

aifn's `Normal.logProb` is built the same way: it standardises z = (x − μ)/σ with primitives, applies the
`normalLogPdf` primitive from `special`, and subtracts `log σ`, so it differentiates with respect to both parameters
and the value without a rule of its own.

## 11. Custom rules, checkpointing and implicit differentiation

Some functions should not be differentiated by chaining their primitives' rules. The chain rule may give NaN where the
derivative exists (‖x‖ at 0, through √ of a sum of squares), the derivative wanted may not be the true one (the
straight-through estimator uses the identity as the gradient of a rounding step), the records may cost too much memory
(a long loop), or the function may be a solver whose iterations are irrelevant to the derivative of its answer. The
functions in `foundation/autodiff/custom.ts`, `numerics/implicit/` and `foundation/trace/differentiate.ts` cover these
cases.

| Function                                       | What it does                                                                                                   | Under each interpreter                                                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `customVjp(f, fwd, bwd)`                       | f with a reverse rule: `fwd` returns the output and residuals, `bwd(residuals, ct)` one cotangent per argument | reverse: one record per output leaf whose rule calls bwd; forward: J·t from bwd by the transpose trick (§7.1); batch: one call per example |
| `customJvp(f, jvp)`                            | f with a forward rule `jvp(primals, tangents) → [out, tangent]`                                                | forward: the rule; reverse: its transpose, one reverse sweep over the rule in the tangents; batch: per example                             |
| `checkpoint(f)`                                | f's forward pass kept off the tape, recomputed in the backward sweep                                           | reverse: one record per output leaf; forward and batch: f as it is                                                                         |
| `implicitFixedPoint(solver, F)`                | the fixed point x⋆ = F(p, x⋆) found by any solver, differentiable in p                                         | custom vjp and jvp from the implicit function theorem                                                                                      |
| `implicitRoot(solver, r)`                      | the root of r(p, x) = 0 found by any solver                                                                    | as above                                                                                                                                   |
| `unrolled(alg, start, n, { checkpointEvery })` | an `Algorithm`'s steps run on tracers                                                                          | differentiated through every step; each segment of k steps is a `checkpoint`                                                               |
| `atConvergence(make, r, { start, select })`    | an `Algorithm` run to convergence on raw values                                                                | `implicitRoot` on the converged state                                                                                                      |

None of these is a primitive. Each looks at the highest-level tracer among its arguments, as `apply` does, and acts
for that interpreter. It lowers the arguments one level and calls itself on them, so the levels below also use the
custom rule. Higher derivatives differentiate the rule: `bwd` runs on lowered values, so an enclosing transform
records it. A rule must therefore be written with primitives, and `bwd` (or the tangent of `jvp`) must be linear in
its cotangent (tangent).

**Checkpointing.** In the forward pass `checkpoint(f)` runs f once under a throwaway reverse level that sees only the
arguments the active transform traces. That level's records say which outputs depend on those arguments, and are then
dropped. The active transform keeps one record per dependent output, whose rule reruns f under a fresh level and sweeps
it. Outputs that do not depend on traced values (a step counter `t`) stay plain numbers. Gradient descent for 100 steps
on ½‖x‖² from a traced 2-vector leaves 504 records (as counted by `traceGraph`); with
`unrolled(..., { checkpointEvery: 10 })` it leaves 44 (one per traced leaf of each segment's output state, plus the
input), and each segment is recomputed once during the backward sweep. f must receive every traced value as an argument: a closure over a tracer of
the transform being checkpointed is an error, because the recomputation could not see it. That is why `unrolled`
accepts a factory `(params) => Algorithm` with `options.params`.

**Implicit differentiation.** If x⋆ solves r(p, x⋆) = 0 and ∂r/∂x is invertible there, differentiating the identity
r(p, x⋆(p)) = 0 gives ∂r/∂p + (∂r/∂x)(∂x⋆/∂p) = 0, so

$$\frac{\partial x^\star}{\partial p} = -\Big(\frac{\partial r}{\partial x}\Big)^{-1} \frac{\partial r}{\partial p}.$$

The reverse rule solves (∂r/∂x)ᵀ w = x̄ and returns p̄ = −(∂r/∂p)ᵀ w. The forward rule solves (∂r/∂x) ẋ = −(∂r/∂p) ṗ.
Neither looks at the solver, which runs on raw values and may be anything: Newton's method, L-BFGS, a `dense` loop.
The linear system is solved densely when x has at most 64 elements (the Jacobian by `jacobian`, then
`aifn/numerics/linalg`'s `solve`, which is why these functions live in `aifn/numerics/implicit`), and otherwise with vjp or jvp products only. `implicitFixedPoint` uses the iteration
w ← (∂F/∂x)ᵀ w + x̄, which converges when the fixed-point iteration contracts; `implicitRoot` uses BiCGSTAB. Before
differentiating, both check that the answer satisfies its equation. A failed check, or an iterative solve that does
not converge, raises `NumericalError('not-converged')` rather than returning a gradient at the wrong point. The
forward rule is written out rather than obtained by the transpose trick. The trick evaluates bwd at a zero cotangent,
and an iterative solve that stops on the size of its residual would stop at once.

### 11.1 Worked example: √p by Newton's method

```ts
const newton = (p: number) => {
  let x = Math.max(p, 1)
  for (let i = 0; i < 60; i++) x = 0.5 * (x + p / x)
  return x
}
const root = implicitRoot(newton, (p: Value, x: Value) => sub(mul(x, x), p)) // r(p, x) = x² − p
grad(root)(2) // 0.35355 = 1/(2√2)
grad(grad(root))(2) // −0.088388 = −1/(4·2^1.5)
```

`grad(root)(2)` starts R1 and calls `root` on the tracer ⟨p⟩. The arguments lowered one level are the raw number 2,
so `implicitRoot` runs Newton's method on 2, checks |x² − 2| ≤ 10⁻⁶(1 + x), and records one node:

| Record | op           | inputs | value   | rule                                     |
| ------ | ------------ | ------ | ------- | ---------------------------------------- |
| n0     | input        |        | 2       | p                                        |
| n1     | implicitRoot | n0     | 1.41421 | solve (∂r/∂x)ᵀ w = x̄, return −(∂r/∂p)ᵀ w |

The sixty Newton iterations leave no records. The backward sweep, seed x̄ = 1 at n1:

| Step | Computation                             | Value                   |
| ---- | --------------------------------------- | ----------------------- |
| 1    | ∂r/∂x = 2x, by `jacobian` of x ↦ x² − p | 2.82843                 |
| 2    | w = x̄ / (∂r/∂x), the 1 × 1 dense solve  | 0.35355                 |
| 3    | ∂r/∂p = −1, by `vjp` of p ↦ x² − p      | −1                      |
| 4    | p̄ = −(∂r/∂p)ᵀ w                         | **0.35355** = 1/(2√2) ✓ |

For `grad(grad(root))(2)` the outer transform R1 is level 1 and the inner R2 is level 2. The inner call lowers its
argument to the level-1 tracer ⟨p⟩ and calls `root` on it, which records n1 on R1 exactly as above. R2's sweep then
runs steps 1–4 on the level-1 tracer ⟨x⋆⟩, so R1 records w = 1/(2x⋆) as a function of x⋆. The outer sweep pulls back
through that and then through n1's own rule:

| Visit | Record                | g             | Contribution                               |
| ----- | --------------------- | ------------- | ------------------------------------------ |
| a     | w = 1/(2x⋆)           | 1             | x⋆ += −1/(2x⋆²) = −0.25                    |
| b     | n1 = implicitRoot(n0) | −0.25         | p += −0.25 · 0.35355 (n1's rule, as above) |
| c     | n0 = input            | **−0.088388** | stop                                       |

f″(2) = −1/(4·2^1.5) = −0.088388 ✓. The second derivative is the derivative of the implicit rule: the solver was
never differentiated at either level.

### 11.2 Where implicit differentiation lives

`implicitFixedPoint`, `implicitRoot` and `atConvergence` are in `aifn/numerics/implicit` (`implicit.ts` and
`convergence.ts`), not in `aifn/foundation/autodiff`. The dense adjoint solve needs `aifn/numerics/linalg`'s `solve`,
and foundation may import nothing above itself. They are built on `defineCustomVjp` from `aifn/foundation/autodiff`,
with a jvp rule of their own (§11). Phase 2c first placed them in `autodiff/implicit.ts` with a private differentiable
dense solve; the move removed that copy.

Solvers elsewhere in core use the module rather than repeat it. `atConvergence` differentiates any `Algorithm` at its
converged state. A hyperparameter gradient through `aifn/optim`'s `minimize` (L-BFGS, say) is `implicitRoot` on the
minimiser's gradient condition.

### 11.3 Differentiating through unrolled optimisers

`unrolled(make, start, n, { params, checkpointEvery })` runs an `Algorithm` for n steps on tracers and returns its
final state. The first-order optimisers (`gradientDescent`, `momentum`, `rmsprop`, `adam`, …) and the explicit
Runge–Kutta and Euler–Maruyama steps are written with primitives, so this differentiates the computed iterate exactly,
step by step:

- **With respect to the step size.** `StepSize` is a `Value` or a schedule, so a traced learning rate flows into every
  update. The learning-rate hypergradient of gradient descent on f(x) = ½ Σᵢ aᵢ xᵢ² after n steps has the closed form
  Σᵢ aᵢ x_{n,i} · n(1 − ηaᵢ)ⁿ⁻¹(−aᵢ) x₀ᵢ, since x_{n,i} = (1 − ηaᵢ)ⁿ x₀ᵢ; `grad` of `unrolled` matches it to 12 digits
  (`core/test/optim/first-order/hypergradient.test.ts`).
- **With respect to the start.** A traced `x0` is kept in the initial state, and an `Objective` is passed through
  `objectiveFn` so that its value and gradient are traced too (a nested reverse pass).
- **Through a simulation.** `rungeKutta` stages and the Euler–Maruyama step accept traced states and parameters, so
  ∂L/∂x₀ and ∂L/∂θ of a solution, or a pathwise gradient of an SDE expectation with common random numbers, come from
  the same call.

The make-function form `(params) => Algorithm` with `options.params` is required with `checkpointEvery` (§11). An
algorithm whose step leaves the trace (a raw `dense` loop) gives `NotDifferentiableError` rather than a zero gradient.
Memory grows linearly in n without checkpoints. Unrolling gives the exact gradient of the discrete computation;
`atConvergence` (§11.2) gives the gradient of the limit and ignores the path.

### 11.4 The ODE adjoint: `odeAdjoint`

`odeAdjoint(f, [t0, t1], options)` in `aifn/dynamics/ode` returns the solution map (x₀, θ) ↦ x(t₁) of
x′ = f(t, x, θ), with f written with primitives. It is a `defineCustomVjp`: the forward pass solves on raw values
and keeps only x(t₁); the reverse rule integrates the augmented system backwards from t₁ (Pontryagin et al., 1962;
Chen et al., 2018):

- x′ = f(t, x, θ), recomputing the state;
- a′ = −aᵀ ∂f/∂x, with a(t₁) = x̄, the adjoint;
- g′ = −aᵀ ∂f/∂θ, with g(t₁) = 0.

Then ∂L/∂x₀ = a(t₀) and ∂L/∂θ = g(t₀). The two vector–Jacobian products come from one `vjp` of f per evaluation.
Memory is constant in the number of steps. `checkpoints: k` keeps x at k segment starts and restarts the backward
reconstruction from each, which bounds the error when x is integrated backwards along an unstable direction. The
method is any of `euler`, `heun`, `midpoint`, `rk4` (default) or `dormand-prince`, the same for both solves.

```ts
const flow = odeAdjoint((t, x, k) => mul(neg(k), x), [0, 1], { stepSize: 0.01 })
grad((k: Value) => sum(flow(tensor([1]), k)))(2) // −0.135335…, the exact −e⁻² to 10 digits
```

The adjoint is optimise-then-discretise: it gives the gradient of the exact flow to the solver's accuracy. `unrolled`
over `rungeKutta` gives the exact gradient of the discrete solution, at memory linear in the steps. Forward mode gets
J·t from the reverse rule by the transpose trick.

## 12. Complex numbers

A complex128 tensor stores each element as two doubles (re, im), interleaved (design K §8.1). Autodiff treats a
complex value as a **pair of reals**: z = x + iy is the point (x, y) of ℝ², and every derivative is the ordinary real
derivative of that pair. Nothing else changes: the same interpreters, the same records, the same sweep.

**The convention.** Losses are real. The cotangent of z is **z̄ = x̄ + iȳ**, the gradient of the loss with respect to
x and to y packed back into one complex number. So `grad(L)(z)` has the dtype and shape of z, gradient descent is
z − η·z̄ exactly as for reals, and a finite-difference check perturbs x and y separately. The Wirtinger derivative
∂L/∂z̄ used in the adaptive-filter literature is ½(x̄ + iȳ). `grad` of a function with a complex output throws
`DTypeError` (as it throws `ShapeError` for a non-scalar one): take `realPart`, `abs` or a squared modulus first.

**What each kind of primitive does.**

| Primitive                                                                                     | Rule under the convention                                                                               |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| holomorphic elementwise (add, sub, mul, div, exp, log, sqrt, pow, square, neg, expj, complex) | vjp ḡ·conj(f′(z)), jvp ż·f′(z); `elementwise({ …, holomorphic: true })` applies the conjugate           |
| real-valued functions of z (`abs` → `complexAbs`, `angle`)                                    | with w = ∂f/∂x + i ∂f/∂y: vjp ḡ·w, jvp Re(conj(w)·ż); for \|z\|, w = z/\|z\|; for arg z, w = i·z/\|z\|² |
| ℝ-linear maps (`conj`, `realPart`, `imagPart`, structure, sum, cumsum)                        | the transpose is the ℝ² adjoint: conj ↦ conj, Re ↦ (x̄ + 0i), Im ↦ (0 + iȳ), structural maps unchanged   |
| complex-linear products (`matmul`, `einsum`)                                                  | the adjoint conjugates the other operand: Ā = C̄ Bᴴ, B̄ = Aᴴ C̄                                            |
| a real input that meets a complex value (`mul(x, z)`)                                         | the cotangent is projected: x̄ = Re(ḡ·conj(z)); a real output's tangent is its real part                 |

The projection is done in one place for each direction: `sumLike` and `fitTo` (which every derived rule uses), the
vjp derived from a transpose, the reverse sweep when a cotangent reaches a real record, and the forward interpreter
when a tangent belongs to a real output. `conj` and `realPart` return real values unchanged without recording
anything, so real code pays nothing for the convention. The transpose trick (§7.1) uses the ℝ² inner product
Re Σ conj(c)·t, and `vmap` works on complex values like any other (the batching rules are structural).

### 12.1 Worked example: the gradient of |z|² = z z̄

Take `L = (z) => realPart(mul(z, conj(z)))` at z = 3 − 4i. As a real function, L = x² + y², so the answer must be
∂L/∂x + i ∂L/∂y = 6 − 8i = 2z.

Forward pass, `grad(L)(z)`:

| Record | op       | inputs | value   |
| ------ | -------- | ------ | ------- |
| n0     | input    |        | 3 − 4i  |
| n1     | conj     | n0     | 3 + 4i  |
| n2     | mul      | n0, n1 | 25 + 0i |
| n3     | realPart | n2     | 25      |

Backward sweep, seed 1 at n3:

| Visit | Record           | g          | Contribution                                                       |
| ----- | ---------------- | ---------- | ------------------------------------------------------------------ |
| a     | n3 = realPart    | 1          | n2 += 1 + 0i (the adjoint of Re)                                   |
| b     | n2 = mul(n0, n1) | 1 + 0i     | n0 += g·conj(n1) = 3 − 4i; n1 += g·conj(n0) = 3 + 4i (holomorphic) |
| c     | n1 = conj(n0)    | 3 + 4i     | n0 += conj(3 + 4i) = 3 − 4i                                        |
| d     | n0 = input       | **6 − 8i** | stop                                                               |

The two contributions to n0 are the two paths from z to L, through z and through z̄. Forward mode gives the same
numbers one direction at a time: `jvp(L, z, 1)` pushes ż = 1 (a step in x) through n1 (ż̄ = 1), n2 (1·(3 + 4i) +
1·(3 − 4i) = 6) and n3 (6) = ∂L/∂x; `jvp(L, z, i)` pushes ż = i through n1 (−i), n2 (i(3 + 4i) − i(3 − 4i) = −8)
and n3 (−8) = ∂L/∂y. Writing the loss as `square(abs(z))` gives the same 6 − 8i through `complexAbs`'s rule
(w = z/|z| = (3 − 4i)/5, times ḡ = 2|z| = 10).

### 12.2 Fourier transforms

`fft`, `ifft`, `rfft` and `irfft` (`aifn/foundation/fourier`) are linear primitives, so each declares only its
transpose, the ℝ² adjoint (the conjugate transpose), and gets its vjp and jvp from it. With F the unnormalised DFT
matrix (Fⱼₖ = e^{−2πijk/n}, symmetric) and `dual` swapping the norms 'backward' and 'forward':

| Primitive                          | Map                            | Transpose                                                   |
| ---------------------------------- | ------------------------------ | ----------------------------------------------------------- |
| `fft(norm)`                        | s·F x                          | `ifft(dual(norm))` (= s·Fᴴ)                                 |
| `ifft(norm)`                       | s·Fᴴ x                         | `fft(dual(norm))`                                           |
| `rfft(norm)` (real n → ⌊n/2⌋ + 1)  | first bins of s·F x            | `realPart(ifft(dual)(pad to n))`                            |
| `irfft(norm)` (⌊n/2⌋ + 1 → real n) | Hermitian extension, then s·Fᴴ | c ⊙ `rfft(dual)`, cₖ = 1 at DC and (n even) Nyquist, else 2 |

The irfft weights c count the two copies of each interior bin in the Hermitian extension; the registry's dot-product
test fails without them. The axis is stored as a negative index, so a batch axis moved to the front leaves the
parameters unchanged and `vmap` costs nothing.
