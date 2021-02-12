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
- Annotations collected with `prodigy` in order to train or evaluate NER models.
- Input files given to the annotators can be either raw sentences 
(see `raw_sentences/`) or pre-annotated files that require corrections or adding new entity types.

# Content
 
## `annotations1_EmmanuelleLogette_2020-06-19_raw1_8FirstLabels.jsonl`	
```shell script
prodigy ner.manual \
    annotations1_EmmanuelleLogette_2020-06-19_raw1_8FirstLabels \
    blank:en \
    raw1_2020-06-10_cord19_TestSet.jsonl \
    --patterns patterns.jsonl 
    -l 'DISEASE,ORGAN,ORGANISM,PATHWAY,CONDITION,PROTEIN,CHEMICAL,CELL_TYPE'
```
- 289 tot sentences
- Annotated by Emmanuelle Logette
- This file contains inconsistency concerning the entity types (disease and DISEASE, organ and ORGANS, ...)
- Name before naming convention: cord19

## `annotations2_CharlotteLorin_2020-06-19_8FirstLabels.jsonl`
```shell script
prodigy ner.manual \
    annotations2_CharlotteLorin_2020-06-19_8FirstLabels \
    blank:en \
    raw1_2020-06-10_cord19_TestSet.jsonl \
    --patterns patterns.jsonl \
    -l 'DISEASE,ORGAN,ORGANISM,PATHWAY,CONDITION,PROTEIN,CHEMICAL,CELL_TYPE'
```
- 314 tot sentences
- Annotated by Charlotte Lorin
- Exactly the same command line as annotations1
- This file contains inconsistency concerning the entity types (disease and DISEASE, organ and ORGANS, ...)
- Name before naming convention: cord19_Charlotte

## `annotations3_EmmanuelleLogette_2020-07-06_raw1_8FirstLabels.jsonl`
- 289 tot sentences
- Same dataset as annotations1 with the inconsistency corrected
- Name before naming convention: cord19_corrected

## `annotations4_CharlotteLorin_2020-07-02_raw1_8FirstLabels.jsonl`
- 314 tot sentences
- Same dataset as annotations2 with the inconsistency corrected
- Name before naming convention: cord19_Charlotte_corrected``

## `annotations5_EmmanuelleLogette_2020-06-30_raw2_Disease.jsonl`
```shell script
prodigy ner.correct \
    annotations5_EmmanuelleLogette_2020-06-30_raw2_Disease \
    en_ner_bc5cdr_md \
    raw2_2020-06-29_cord19_Disease.jsonl \
    -l 'DISEASE'
```
- 207 tot sentences
- Annotated by Emmanuelle Logette
- Comments from Emmanuelle:
   - "Assez fiable dans l’ensemble"
   - "COVID_19" is often missed.
   - "La plupart des erreurs sont des phrases "nichées" ou des termes qui ne 
   sont pas vraiment des maladies genre "toxicity"
   - 30-40 mins to annotate this dataset
- Name before naming convention: ner_correct_disease_bc5

## `annotations6_EmmanuelleLogette_2020-07-07_raw4_TaxonChebi.jsonl`
```shell script
prodigy ner.correct \
    annotations6_EmmanuelleLogette_2020-07-07_raw4_TaxonChebi \
    en_ner_craft_md \
    raw4_2020-07-02_cord19_ChemicalOrganism.jsonl  \
    -l 'TAXON,CHEBI'
```
- 634 tot sentences
- Annotated by Emmanuelle Logette
- Two iterations for those annotations (first one with pre-annotation from en_ner_craft_md, and second one with pre
-annotation from model2)
  - 380 sentences for first iteration (+/- one hour to annotate)
  - 254 sentences for the second iteration
- TAXON = ORGANISM
- CHEBI = CHEMICAL
- Comments from Emmanuelle:
  - Missed in TAXON : SARS-CoV (many times) ; MERS-CoV (many times), staphylococcus (once), coronaviruses ((many times
 ), dog (many times) BUT dogS is recognized; insect, bats, pathogen
  - removed from chebi : molecular, protein, protein sequence, DNA sequence, compounds, drug …….
  - missed in chebi : sugar
- Name before naming convention: ner_correct_organism_chemical

## `annotations7_EmmanuelleLogette_2020-07-06_raw1_9EntityTypes.jsonl`
```shell script
prodigy ner.manual \
    annotations7_EmmanuelleLogette_2020-07-06_raw1_9EntityTypes \
    blank:en \
    annotations3_EmmanuelleLogette_2020-07-06_raw1_8FirstLabels.jsonl \ 
    -l 'DISEASE,CHEMICAL,ORGAN,ORGANISM,PROTEIN,PATHWAY,CONDITION,CELL_TYPE,DRUG'
```
- 298 tot sentences
- Annotated by Emmanuelle Logette
- Complete Test set annotation with the new entity type: DRUG
- Feedback from Emmanuelle: not really suited for DRUG entity type.

## `annotations8_EmmanuelleLogette_2020-07-08_raw5_9EntityTypes.jsonl`
```shell script
prodigy ner.manual \
    annotations8_EmmanuelleLogette_2020-07-08_raw5_9EntityTypes \
    blank:en \
    raw5_2020-07-08_cord19_Drug_TestSet.jsonl \
    -l 'DISEASE,CHEMICAL,ORGAN,ORGANISM,PROTEIN,PATHWAY,CONDITION
    ,CELL_TYPE,DRUG'
```
- 30 tot sentences
- Annotated by Emmanuelle Logette
- New sentences in response to the feedback on 
`annotations7_EmmanuelleLogette_2020-07-06_raw1_9EntityTypes.jsonl`

## `annotations9_EmmanuelleLogette_2020-07-08_raw6_CelltypeProtein.jsonl`
```shell script
prodigy ner.correct \
    annotations9_EmmanuelleLogette_2020-07-08_raw6_CelltypeProtein \
    en_ner_jnlpba_md \
    raw6_2020-07-08_cord19_CelltypeProtein.jsonl \
    -l 'PROTEIN,CELL_TYPE'
```
- 131 tot sentences
- Annotated by Emmanuelle Logette
- Feedback from Emmanuelle:
  - Cellular compartment is cruelly missing in the entity types
  - Sentences 7873305 and 7043422 good for DRUGS annotations
  - Neuron is not detected as CELL_TYPE
  - Some false recognition for PROTEIN
  - It is performing good in general

## `annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes.jsonl`
```shell script
prodigy ner.manual \
    annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes \
    blank:en \
    $(cat annotations7_EmmanuelleLogette_2020-07-06_raw1_9EntityTypes.jsonl \
          annotations8_EmmanuelleLogette_2020-07-08_raw5_9EntityTypes.jsonl) \
    -l 'DISEASE,CHEMICAL,ORGAN,ORGANISM,PROTEIN,PATHWAY,CONDITION,CELL_TYPE,DRUG,CELL_COMPARTMENT'
```
- 309 tot sentences 
- Annotated by Emmanuelle Logette
- Complete Test set annotation with the new entity type: CELL_COMPARTMENT
= Emmanuelle's feedback: not enough entities --> creation of new dataset of 23 sentences (raw7)

## `annotations11_CharlotteLorin_2020-08-28_raw1_10EntityTypes.jsonl`
```shell script
prodigy ner.manual \
    annotations11_CharlotteLorin_2020-08-28_raw1_10EntityTypes \
    blank:en \
    annotations4_CharlotteLorin_2020-07-02_raw1_8FirstLabels.jsonl \
    -l 'DISEASE,CHEMICAL,ORGAN,ORGANISM,PROTEIN,PATHWAY,CONDITION,CELL_TYPE,DRUG,CELL_COMPARTMENT'
```
- 314 tot sentences
- Annotated by Charlotte Lorin
- Complete Test set annotation with the new entity types: DRUG and CELL_COMPARTMENT

## `annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes.jsonl`
```shell script
prodigy ner.manual \
    annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes \
    blank:en \
    raw7_2020-09-01_cord19v35_CellCompartment.jsonl \
    -l 'DISEASE,CHEMICAL,ORGAN,ORGANISM,PROTEIN,PATHWAY,CONDITION,CELL_TYPE,DRUG,CELL_COMPARTMENT'
```
- 23 tot sentences
- Annotated by Emmanuelle Logette
- Complete Test set for CELL_COMPARTMENT and allow to choose the best model for this entity type.

## `annotations13_CharlotteLorin_2020-09-02_raw7_10EntityTypes.jsonl`
```shell script
prodigy ner.manual \
    annotations13_CharlotteLorin_2020-09-02_raw7_10EntityTypes \
    blank:en \
    raw7_2020-09-01_cord19v35_CellCompartment.jsonl \
    -l 'DISEASE,CHEMICAL,ORGAN,ORGANISM,PROTEIN,PATHWAY,CONDITION,CELL_TYPE,DRUG,CELL_COMPARTMENT'
```
- 23 tot sentences
- Annotated by Charlotte Lorin
- Complete Test set for CELL_COMPARTMENT

## `annotations14_EmmanuelleLogette_2020-09-02_raw8_CellCompartmentDrugOrgan.jsonl`
```shell script
prodigy ner.correct \
    annotations14_EmmanuelleLogette_2020-09-02_raw8_CellCompartmentDrugOrgan \
    en_ner_bionlp13cg_md \
    raw8_2020-09-02_cord19v35_CellCompartmentDrugOrgan.jsonl \
    -l 'ORGAN,SIMPLE_CHEMICAL,CELLULAR_COMPONENT'
```
- 383 tot sentences
- Annotated by EmmanuelleLogette
- Training set for classes CELL_COMPARTMENT, DRUG, ORGAN

## `annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.jsonl`
```shell script
prodigy ner.manual \
    annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway \
    en_core_web_lg \
    raw9_2020-09-02_cord19v35_Pathway.jsonl \
    --patterns pathway_patterns.jsonl \
    -l 'PATHWAY'
```
- 151 tot sentences
- Annotated by EmmanuelleLogette
- Training set for class PATHWAY

## `rule_based_patterns.jsonl`
- Collection of patterns for rule based entity recognition with `EntityRuler`
- We use it in the `add_er_*` stages to append an `EntityRuler`
