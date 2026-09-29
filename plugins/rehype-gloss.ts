/**
 * Marks the first `<Gloss>` of each glossary entry in a note with a `first` attribute, as the LaTeX acronym packages
 * do: the first use renders "Long Form (SHORT)", later uses the short form. Done at compile time, in document order,
 * so the result does not depend on React's render order.
 *
 * Aliases resolve to their entry, so `<Gloss name="auc" />` followed by `<Gloss name="auroc" />` expands only once. A use
 * with an explicit `form` renders as asked and does not count as the first use. Unknown names are left alone; the
 * content index reports them with the file named.
 */
import { loadGlossary } from './glossary.ts'

type Attribute = { type: string; name?: string; value?: unknown }
type Node = { type: string; name?: string | null; attributes?: Attribute[]; children?: Node[] }

export function rehypeGlossFirstUse({ glossaryFile }: { glossaryFile: string }) {
  return (tree: Node) => {
    let names: Map<string, string>
    try {
      names = loadGlossary(glossaryFile).names
    } catch {
      // A malformed glossary is reported by the content index; the note still compiles.
      names = new Map()
    }
    const seen = new Set<string>()
    const walk = (node: Node) => {
      if ((node.type === 'mdxJsxTextElement' || node.type === 'mdxJsxFlowElement') && node.name === 'Gloss') {
        const attrs = node.attributes ?? []
        const attr = (n: string) => attrs.find((a) => a.type === 'mdxJsxAttribute' && a.name === n)
        const name = attr('name')?.value
        if (typeof name === 'string' && !attr('form') && !attr('first')) {
          const key = names.get(name) ?? name
          if (!seen.has(key)) {
            seen.add(key)
            // A valueless attribute is `first={true}` in the compiled JSX.
            node.attributes = [...attrs, { type: 'mdxJsxAttribute', name: 'first', value: null }]
          }
        }
      }
      node.children?.forEach(walk)
    }
    walk(tree)
  }
}
