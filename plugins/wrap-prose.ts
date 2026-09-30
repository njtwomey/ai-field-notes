/**
 * Prose wrapping for MDX notes. Every paragraph with a line longer than WIDTH is reflowed to WIDTH, breaking only at
 * whitespace that belongs to text: never inside inline maths, inline code, a JSX tag or a link destination. Headings,
 * tables, display maths, code blocks, JSX blocks and frontmatter are not paragraphs and are never touched.
 *
 * Two checks guard every rewrapped note. `wrapProse` parses the result again: its syntax tree must equal the
 * original's, ignoring positions and treating any run of whitespace as one space. `sameRender` then renders both
 * versions through the site's own remark and rehype plugins (plugins/mdx-options.ts): smartypants, KaTeX, first-use
 * <Gloss>, heading slugs. The HTML trees, JSX elements included, must match up to whitespace, which HTML does not
 * render. The wrapper never breaks inside maths or code, where whitespace would matter.
 */
import remarkFrontmatter from 'remark-frontmatter'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkMdx from 'remark-mdx'
import remarkParse from 'remark-parse'
import remarkRehype, { type Options as RemarkRehypeOptions } from 'remark-rehype'
import { unified } from 'unified'
import { mdxOptions } from './mdx-options.ts'

export const WIDTH = 120

type Node = {
  type: string
  children?: Node[]
  value?: string
  position?: { start: { offset?: number }; end: { offset?: number } }
  [key: string]: unknown
}

const parser = unified().use(remarkParse).use(remarkFrontmatter).use(remarkMdx).use(remarkGfm).use(remarkMath)

export function parseMdx(text: string): Node {
  return parser.parse(text) as unknown as Node
}

/** A word may not start a line when it would open a block there: a list item, quote, heading, setext underline,
 * display maths or a fence. */
const BAD_START = /^(?:(?:[-*+>]|#{1,6}|\d+[.)])$|[-*+>](?=\s)|=+$|-+$|\$\$|```|~~~)/

export type WrapResult = { text: string; paragraphs: number; failure?: string }

/** Reflow the paragraphs of `text` that have a line longer than `width`. */
export function wrapProse(text: string, width = WIDTH): WrapResult {
  const tree = parseMdx(text)
  const edits: { from: number; to: number; block: string }[] = []
  visit(tree, [], (node, ancestors) => {
    if (node.type !== 'paragraph' || ancestors.some((a) => a.type === 'blockquote')) return
    const edit = reflow(text, node, width)
    if (edit) edits.push(edit)
  })
  if (edits.length === 0) return { text, paragraphs: 0 }
  let out = text
  for (const e of edits.sort((a, b) => b.from - a.from)) out = out.slice(0, e.from) + e.block + out.slice(e.to)
  const failure = compare(tree, parseMdx(out))
  return failure ? { text, paragraphs: 0, failure } : { text: out, paragraphs: edits.length }
}

function visit(node: Node, ancestors: Node[], f: (n: Node, ancestors: Node[]) => void) {
  f(node, ancestors)
  for (const child of node.children ?? []) visit(child, [...ancestors, node], f)
}

function reflow(text: string, para: Node, width: number): { from: number; to: number; block: string } | undefined {
  const start = para.position?.start.offset
  const end = para.position?.end.offset
  if (start === undefined || end === undefined) return undefined
  const lineStart = text.lastIndexOf('\n', start - 1) + 1
  const lineEnd = text.indexOf('\n', end) === -1 ? text.length : text.indexOf('\n', end)
  const lines = text.slice(lineStart, lineEnd).split('\n')
  if (!lines.some((l) => l.length > width)) return undefined
  // The paragraph must own its lines: only indentation or list markers before it, nothing after it.
  const prefix = text.slice(lineStart, start)
  if (!/^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)*$/.test(prefix) || text.slice(end, lineEnd).trim() !== '') return undefined
  const indent = /^[ \t]*$/.test(prefix) ? prefix : ' '.repeat(prefix.length)

  // Characters that belong to text (not maths, code, tags or link destinations) may be broken at.
  const inText = new Uint8Array(end - start)
  visit(para, [], (n) => {
    const s = n.position?.start.offset
    const e = n.position?.end.offset
    if (n.type === 'text' && s !== undefined && e !== undefined) inText.fill(1, s - start, e - start)
  })
  if (hasHardBreak(para)) return undefined

  // Split at whitespace runs that begin inside text. Other whitespace stays inside its word, verbatim.
  const body = text.slice(start, end)
  const words: string[] = []
  let word = ''
  for (let i = 0; i < body.length;) {
    if (/\s/.test(body[i]) && inText[i]) {
      let j = i
      while (j < body.length && /\s/.test(body[j])) j++
      if (word) words.push(word)
      word = ''
      i = j
    } else word += body[i++]
  }
  if (word) words.push(word)

  // A line holding only JSX tags is a block element in MDX, so never start a line when only tags remain.
  const isTag = (w: string) => /^<[\s\S]*>$/.test(w)
  const onlyTagsFrom = words.map((_, i) => words.slice(i).every(isTag))
  const out: string[] = []
  let line = prefix + words[0]
  for (const [i, w] of words.entries()) {
    if (i === 0) continue
    const lastSegment = (line + ' ' + w).split('\n').pop()!
    if (lastSegment.length <= width || BAD_START.test(w) || onlyTagsFrom[i]) line += ' ' + w
    else {
      out.push(line)
      line = indent + w
    }
  }
  out.push(line)
  const block = out.join('\n')
  const old = text.slice(lineStart, end)
  return block === old ? undefined : { from: lineStart, to: end, block }
}

function hasHardBreak(node: Node): boolean {
  return node.type === 'break' || (node.children ?? []).some(hasHardBreak)
}

/**
 * Syntax trees without positions, with every whitespace run in a string collapsed to one space. The `estree` of a JSX
 * expression is dropped: it repeats the expression's source, which is compared as its `value`, with offsets.
 */
function normalise(node: unknown): unknown {
  if (typeof node === 'string') return node.replace(/\s+/g, ' ')
  if (Array.isArray(node)) return node.map(normalise)
  if (node && typeof node === 'object')
    return Object.fromEntries(
      Object.entries(node)
        .filter(([k]) => k !== 'position' && k !== 'estree')
        .map(([k, v]) => [k, normalise(v)]),
    )
  return node
}

function compare(before: Node, after: Node): string | undefined {
  const a = JSON.stringify(normalise(before))
  const b = JSON.stringify(normalise(after))
  if (a === b) return undefined
  let i = 0
  while (i < a.length && a[i] === b[i]) i++
  return `the rewrapped note parses differently near …${a.slice(Math.max(0, i - 60), i + 60)}…`
}

/** MDX nodes that remark-rehype passes through to the HTML tree, as @mdx-js/mdx does. */
const MDX_NODES = [
  'mdxFlowExpression',
  'mdxJsxFlowElement',
  'mdxJsxTextElement',
  'mdxTextExpression',
  'mdxjsEsm',
] as const

const renderer = unified()
  .use(remarkParse)
  .use(remarkMdx)
  .use(mdxOptions.remarkPlugins)
  .use(remarkRehype, { passThrough: [...MDX_NODES], allowDangerousHtml: true } satisfies RemarkRehypeOptions)
  .use(mdxOptions.rehypePlugins)

/** The HTML tree the site renders from `text`, before JSX components run. */
export async function renderTree(text: string): Promise<Node> {
  return (await renderer.run(renderer.parse(text))) as unknown as Node
}

/** Undefined when both versions render the same, up to whitespace; otherwise where they first differ. */
export async function sameRender(before: string, after: string): Promise<string | undefined> {
  const failure = compare(await renderTree(before), await renderTree(after))
  return failure?.replace('parses differently', 'renders differently')
}
