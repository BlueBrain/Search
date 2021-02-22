<!---
Blue Brain Search is a text mining toolbox focused on scientific use cases.

Copyright (C) 2020  Blue Brain Project, EPFL.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
-->

# Description
- Annotations collected in order to train or evaluate sentence embedding models. 

# Content

## `cord19_v47_sentences_pre.txt`
- Unannotated file of sentences (one line per sentence) from cord-19.
- 20,510,932 total sentences.
- Can be used to train unsupervised nlp models.

## `sentence_similarity_cord19.csv`
- Sentences pairs with similarity scores annotated by Emmanuelle Logette. 
- 40 sentences pairs in total:
  - 20 pairs (those with `sentence_id` starting by `A-`) are generically
  extracted from the CORD-19 dataset
  - 20 pairs (those with `sentence_id` starting by `B-`) are also extracted from
  the CORD-19 dataset but are focused on "COVID-19" and "glucose" topics.
- The scoring system is the one used in Soğancıoğlu G. et al. "BIOSSES: a semantic sentence
 similarity estimation system for the biomedical domain." Bioinformatics 33.14 (2017): i49-i58.
	
| Score | Comment |
| --- | --- |
| 0 | The two sentences are on different topics. |
| 1 | The two sentences are not equivalent, but are on the same topic. |
| 2 | The two sentences are not equivalent, but share some details. |
| 3 | The two sentences are roughly equivalent, but some important information differs/missing. |
| 4 | The two sentences are completely or mostly equivalent, as they mean the same thing. |
