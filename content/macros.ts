/**
 * Maths macros for every note: the site's equivalent of a LaTeX `definitions.sty`.
 *
 * Passed to KaTeX at build time (vite.config.ts → rehype-katex) and checked against every formula by
 * `npm run check:content`. The /notation page renders this file, so readers see the same conventions.
 *
 * Conventions: bold lowercase is a vector (`\xvec`), bold uppercase a matrix (`\Xmat`), calligraphic a set or family
 * (`\Dcal`), blackboard a number system or operator (`\reals`, `\expect`). Arguments use `#1`, `#2`.
 */

export type MacroGroup = {
  title: string
  description?: string
  macros: Record<string, string>
  /**
   * How to demonstrate a macro that cannot render on its own, e.g. `\\LP` needs a matching `\\RP`. Shown on the
   * /notation page and checked by `npm run check:content`.
   */
  examples?: Record<string, string>
}

const LOWER = 'abcdefghijklmnopqrstuvwxyz'.split('')
const UPPER = LOWER.map((c) => c.toUpperCase())
const GREEK_LOWER = [
  'alpha',
  'beta',
  'gamma',
  'delta',
  'epsilon',
  'varepsilon',
  'zeta',
  'eta',
  'theta',
  'vartheta',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'xi',
  'pi',
  'rho',
  'sigma',
  'tau',
  'upsilon',
  'phi',
  'varphi',
  'chi',
  'psi',
  'omega',
]
const GREEK_UPPER = ['Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega']

const family = (names: string[], name: (s: string) => string, body: (s: string) => string) =>
  Object.fromEntries(names.map((s) => [`\\${name(s)}`, body(s)]))

export const macroGroups: MacroGroup[] = [
  {
    title: 'Vectors',
    description: 'Bold lowercase. Greek vectors also have a short form: \\mub for \\muvec.',
    macros: {
      ...family(
        LOWER,
        (c) => `${c}vec`,
        (c) => `\\mathbf{${c}}`,
      ),
      ...family(
        GREEK_LOWER,
        (g) => `${g}vec`,
        (g) => `\\boldsymbol{\\${g}}`,
      ),
      ...family(
        GREEK_LOWER,
        (g) => `${g}b`,
        (g) => `\\boldsymbol{\\${g}}`,
      ),
      '\\ones': '\\mathbf{1}',
      '\\zeros': '\\mathbf{0}',
    },
  },
  {
    title: 'Matrices',
    description: 'Bold uppercase. Greek capitals that look Latin (A, B, E, …) use the Latin form: \\Amat.',
    macros: {
      ...family(
        UPPER,
        (c) => `${c}mat`,
        (c) => `\\mathbf{${c}}`,
      ),
      ...family(
        GREEK_UPPER,
        (g) => `${g}mat`,
        (g) => `\\boldsymbol{\\${g}}`,
      ),
      ...family(
        GREEK_UPPER,
        (g) => `${g}b`,
        (g) => `\\boldsymbol{\\${g}}`,
      ),
      '\\eye': '\\mathbf{I}',
      '\\I': '\\mathbf{I}',
      '\\transpose': '^{\\mathsf{\\top}}',
      '\\tr': '\\top',
    },
  },
  {
    title: 'Sets and families',
    description: 'Calligraphic capitals.',
    macros: family(
      UPPER,
      (c) => `${c}cal`,
      (c) => `\\mathcal{${c}}`,
    ),
  },
  {
    title: 'Number systems and probability',
    macros: {
      '\\reals': '\\mathbb{R}',
      '\\complex': '\\mathbb{C}',
      '\\rationals': '\\mathbb{Q}',
      '\\integers': '\\mathbb{Z}',
      '\\naturals': '\\mathbb{N}',
      '\\expect': '\\mathbb{E}',
      '\\prob': '\\mathbb{P}',
      '\\pr': '\\operatorname{Pr}',
      '\\indicator': '\\mathbb{I}',
      '\\iid': '\\mathrel{\\overset{\\text{iid}}{\\sim}}',
      '\\eqdef': '\\stackrel{\\mathrm{def}}{=}',
    },
  },
  {
    title: 'Operators',
    description: 'Upright names with operator spacing. Starred forms take limits underneath in display.',
    macros: {
      '\\argmin': '\\operatorname*{arg\\,min}',
      '\\argmax': '\\operatorname*{arg\\,max}',
      '\\trace': '\\operatorname{tr}',
      '\\diag': '\\operatorname{diag}',
      '\\rank': '\\operatorname{rank}',
      '\\Span': '\\operatorname{span}',
      '\\sgn': '\\operatorname{sgn}',
      '\\var': '\\operatorname{var}',
      '\\cov': '\\operatorname{cov}',
      '\\corr': '\\operatorname{corr}',
      '\\KL': '\\operatorname{KL}',
      '\\entropy': '\\operatorname{H}',
      '\\AIC': '\\operatorname{AIC}',
      '\\BIC': '\\operatorname{BIC}',
    },
  },
  {
    title: 'Brackets, norms and inner products',
    description: 'Delimiters size themselves to their contents.',
    macros: {
      '\\abs': '\\left\\lvert #1 \\right\\rvert',
      '\\norm': '\\left\\lVert #1 \\right\\rVert',
      '\\inner': '\\left\\langle #1,\\, #2 \\right\\rangle',
      '\\set': '\\left\\{ #1 \\right\\}',
      '\\paren': '\\left( #1 \\right)',
      '\\brack': '\\left[ #1 \\right]',
      '\\LP': '\\left(',
      '\\RP': '\\right)',
      '\\LB': '\\left[',
      '\\RB': '\\right]',
      '\\LC': '\\left\\{',
      '\\RC': '\\right\\}',
      '\\LA': '\\left\\langle',
      '\\RA': '\\right\\rangle',
    },
    examples: {
      '\\LP': '\\LP \\frac{a}{b} \\RP',
      '\\RP': '\\LP \\frac{a}{b} \\RP',
      '\\LB': '\\LB \\frac{a}{b} \\RB',
      '\\RB': '\\LB \\frac{a}{b} \\RB',
      '\\LC': '\\LC \\frac{a}{b} \\RC',
      '\\RC': '\\LC \\frac{a}{b} \\RC',
      '\\LA': '\\LA \\frac{a}{b} \\RA',
      '\\RA': '\\LA \\frac{a}{b} \\RA',
    },
  },
  {
    title: 'Distributions',
    macros: {
      '\\Gauss': '\\mathcal{N}',
      '\\Bern': '\\operatorname{Bern}',
      '\\Binom': '\\operatorname{Binom}',
      '\\Cat': '\\operatorname{Cat}',
      '\\Mult': '\\operatorname{Mult}',
      '\\Poisson': '\\operatorname{Poisson}',
      '\\Geom': '\\operatorname{Geom}',
      '\\Unif': '\\operatorname{Unif}',
      '\\Exp': '\\operatorname{Exp}',
      '\\GammaD': '\\operatorname{Gamma}',
      '\\Beta': '\\operatorname{Beta}',
      '\\Dir': '\\operatorname{Dir}',
      '\\Laplace': '\\operatorname{Laplace}',
      '\\Wishart': '\\operatorname{Wishart}',
      '\\ChiSq': '\\chi^2',
    },
  },
]

/** Arity of a macro: the highest #n in its body. */
const arity = (body: string) => Math.max(0, ...[...body.matchAll(/#(\d)/g)].map((m) => Number(m[1])))

/** A renderable demonstration of every macro: its example if it has one, else the macro with placeholder arguments. */
export function macroExample(group: MacroGroup, name: string): string {
  const body = group.macros[name]
  return (
    group.examples?.[name] ??
    name +
      ['x', 'y']
        .slice(0, arity(body))
        .map((p) => `{${p}}`)
        .join('')
  )
}

/** Flat table passed to KaTeX. */
export const macros: Record<string, string> = Object.assign({}, ...macroGroups.map((g) => g.macros))
