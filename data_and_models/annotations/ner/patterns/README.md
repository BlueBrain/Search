# Description
- These pattern files are sometimes used to train NER models to provide a first guess.
- This is particularly necessary when no basis model can be found (e.g. SciSpaCy models) to provide good first
 guesses for the entity type of interest. 

# Content
## `patterns/patterns.jsonl`
- Contains all entities that Emmanuelle identified in Ontology v3 (it then 
pre-annotates those entities in the prodigy GUI).

## `patterns/pathway_patterns.jsonl`
- Contains a list of entities that Emmanuelle considers as a good starting point
for the entity type PATHWAY. 
- The file was generated using `prodigy terms.teach`.