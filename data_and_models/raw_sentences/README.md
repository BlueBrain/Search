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
This directory contains collections of raw sentences without any annotations.
They can be used to feed prodigy during an annotation process, see the `annotations/` directory.

# Content
## `raw1_2020-06-10_cord19_TestSet.jsonl`
- 334 tot sentences
- All sentences chosen from grep of entities (after lemma) identified by Emmanuelle (in Ontology v3) for the 8 first labels.

## `raw2_2020-06-29_cord19_Disease.jsonl`
- 200 tot sentences
  - 100 sentences chosen with grep of the entities that Emmanuelle identified as DISEASE
  - 100 sentences chosen randomly
- Sentences from `raw1` were excluded
- All sentences are shuffled

## `raw3_2020-06-30_cord19_Disease.jsonl`
- 2100 tot sentences
  - 1000 sentences chosen with grep of the entities that Emmanuelle identified as DISEASE
  - 1000 sentences chosen randomly
  - 100 sentences chosen with grep of 'covid-19' (from Emmanuelle's remarks that covid-19 was not always identified as
 disease)
 - Sentences from `raw1` and `raw2` were excluded
 - All sentences are shuffled

## `raw4_2020-07-02_cord19_ChemicalOrganism.jsonl`
- 1000 tot sentences
  - 500 sentences chosen with grep of the entities that Emmanuelle identified as CHEMICAL and ORGANISM
  - 500 sentences chosen randomly
 - Sentences from `raw1` were excluded
- All sentences are shuffled

## `raw5_2020-07-08_cord19_Drug_TestSet.jsonl`
-  39 tot sentences	
- Goal: Complete Test Set with sentences more focused on DRUG entity type. (After Emmanuelle finished annotations7)
- All sentences chosen with grep of the entities that Emmanuelle identified as DRUG (anakinra, insulin, azithromycin
, antibiotics, aciclovir, omeprazol, tamiflu, cortisone, chemotherapy, 5-FU, etoposide, anti-il1, anti-il6, chloroquine, indinavir, tocilizumab, metformin, corticosteroids, treatment, remdesivir, vaccines)

## `raw6_2020-07-08_cord19_CelltypeProtein.jsonl`
- 1000 tot sentences	
  - 500 sentences chosen with grep of the entities that Emmanuelle identified as CELL_TYPE and PROTEIN (because jnlpba
  is currently the best model for those entity types)
  - 500 sentences chosen randomly
- Sentences from `raw1` were excluded
- All sentences are shuffled

## `raw7_2020-09-01_cord19v35_CellCompartment.jsonl`
- 23 tot sentences	
- Sentences focused on CellCompartment to add sentences with more entities CELL_COMPARTMENT in the test set

## `raw8_2020-09-02_cord19v35_CellCompartmentDrugOrgan.jsonl`
- 1000 tot sentences	
  - 500 sentences chosen with grep of the entities that Emmanuelle identified as CELL_COMPARTMENT, DRUG and ORGAN (for
  `bionlp13` model)
  - 500 sentences chosen randomly
- Sentences from `raw1` and `raw7` were excluded. (considered as the test set)
- All sentences are shuffled and come from the CORD19_v35

## `raw9_2020-09-02_cord19v35_Pathway.jsonl`
- 1000 tot sentences 	
  - 500 sentences chosen with grep of the entities that Emmanuelle identified as PATHWAY
  - 500 sentences chosen randomly
- Sentences from `raw1` and `raw7` were excluded
- All sentences are shuffled and come from the CORD19_v35
