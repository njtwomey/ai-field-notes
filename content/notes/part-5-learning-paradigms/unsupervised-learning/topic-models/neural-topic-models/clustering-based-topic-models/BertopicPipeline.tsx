import { Diagram, Figure } from 'aifn-render'
import { bertopicSpec } from '../_shared/specs'

/** The BERTopic pipeline: embed, reduce, cluster, then describe each cluster with class-based TF-IDF. */
export function BertopicPipeline() {
  return (
    <Figure
      title="The BERTopic pipeline"
      caption="Documents are embedded by a pretrained sentence encoder, reduced with UMAP and clustered with HDBSCAN. The embeddings decide which documents form a topic. Separately, each cluster's documents are concatenated into one bag of words, and class-based TF-IDF picks the words that describe the cluster. Top2Vec follows the same first three steps but describes a cluster by the word vectors nearest its centroid."
    >
      <Diagram
        spec={bertopicSpec}
        ariaLabel="BERTopic: documents to SBERT embeddings to UMAP to HDBSCAN clusters to c-TF-IDF to topic words"
      />
    </Figure>
  )
}
