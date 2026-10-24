import re

text = """SynapseAI uses ChromaDB as its persistent vector database. ChromaDB enables semantic search over text chunks. Knowledge Graph implements NetworkX DiGraph for topological retrieval."""
entities = [
    ('SynapseAI', 'ent_technology_synapseai'),
    ('ChromaDB', 'ent_technology_chromadb'),
    ('vector database', 'ent_technology_vector_database'),
    ('semantic search', 'ent_concept_semantic_search'),
    ('Knowledge Graph', 'ent_technology_knowledge_graph'),
    ('NetworkX DiGraph', 'ent_technology_networkx_digraph')
]

for sent in re.split(r'(?<=[.!?])\s+', text):
    sent_lower = sent.lower()
    matches = []
    for name, eid in entities:
        for m in re.finditer(rf'\b{re.escape(name.lower())}\b', sent_lower):
            matches.append((m.start(), m.end(), name, eid))
    matches.sort()
    print('Sentence:', sent)
    for i in range(len(matches)):
        for j in range(i + 1, min(i + 3, len(matches))):
            m1, m2 = matches[i], matches[j]
            between = sent[m1[1]:m2[0]].strip()
            print(f'  Pair: {m1[2]} -> {m2[2]}, between: "{between}"')
