/**
 * Transforms pandoc-style citations `[@key]` or `[@key1, @key2]` into `<Cite id="..." />` elements in MDX.
 */
import type { Node } from 'unist'
import { visit } from 'unist-util-visit'

type MdxAttribute = { type: 'mdxJsxAttribute'; name: string; value: string }
type MdxJsxElement = {
  type: 'mdxJsxTextElement'
  name: 'Cite'
  attributes: MdxAttribute[]
  children: unknown[]
}
type TextNode = Node & { type: 'text'; value: string }
type ParentNode = Node & { type: string; children: unknown[] }

const CITE_PATTERN = /(?:\[@|\(@)([a-zA-Z0-9_\-]+(?:\s*[,;]\s*@[a-zA-Z0-9_\-]+)*)[\]\)]/g

export function remarkPandocCite() {
  return (tree: Node) => {
    visit(tree, 'text', (node: TextNode, index: number | undefined, parent: ParentNode | undefined) => {
      if (!node.value || (!node.value.includes('[@') && !node.value.includes('(@')) || index === undefined || !parent)
        return

      CITE_PATTERN.lastIndex = 0
      if (!CITE_PATTERN.test(node.value)) return

      CITE_PATTERN.lastIndex = 0
      const replacement: (TextNode | MdxJsxElement)[] = []
      let lastIndex = 0
      let match: RegExpExecArray | null

      while ((match = CITE_PATTERN.exec(node.value)) !== null) {
        if (match.index > lastIndex) {
          replacement.push({ type: 'text', value: node.value.slice(lastIndex, match.index) })
        }
        const keys = match[1]
          .split(/[,;]/)
          .map((k) => k.trim().replace(/^@/, ''))
          .filter(Boolean)

        replacement.push({
          type: 'mdxJsxTextElement',
          name: 'Cite',
          attributes: [{ type: 'mdxJsxAttribute', name: 'id', value: keys.join(', ') }],
          children: [],
        })
        lastIndex = CITE_PATTERN.lastIndex
      }

      if (lastIndex < node.value.length) {
        replacement.push({ type: 'text', value: node.value.slice(lastIndex) })
      }

      parent.children.splice(index, 1, ...replacement)
      return index + replacement.length
    })
  }
}
