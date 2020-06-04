"""Module for the gold standard creation of NER."""
from matplotlib import cm
import numpy as np
import pandas as pd
import spacy


class GoldStandardDataset():
    """Creation of a Gold Standard dataset based on a database and entities.

    Parameters
    ----------
    database: sqlite3.Cursor
        Cursor to the database with 'sentences' table.
    entities_dict: dict
        Dictionary containing entity_types as key and the corresponding entities
        to entity_type. Those entities can be specified through:
         - a string (comma separated list)
         - a list of string.
    baseline_dataset: pd.DataFrame or list
        Either, List of sentence ids if already identified or
        Either, pd.DataFrame containing sentence_ids and the sentence directly.

    Attributes
    ----------
    sentences: pd.DataFrame
        Dataframe containing all the sentences of the database
    baseline_dataset: pd.DataFrame or list
        Either, List of sentence ids if already identified or
        Either, pd.DataFrame containing sentence_ids and the sentence directly.
    ground_truth_annotation: pd.DataFrame
        DataFrame containing all the annotations.
    """

    def __init__(self,
                 database,
                 entities_dict,
                 baseline_dataset=None):

        self.db = database

        print('Loading sentences ....')
        self.sentences = pd.read_sql("""SELECT * FROM sentences""", self.db)
        print('Sentences loaded!')

        print('Loading spacy model ...')
        self.nlp = spacy.load("en_core_web_sm")
        print('Spacy model loaded!')

        self.entity_types_to_entity = entities_dict
        if not isinstance(list(self.entity_types_to_entity.keys())[0], set) \
                and not isinstance(list(self.entity_types_to_entity.keys())[0], list):
            for entity_type, entities in self.entity_types_to_entity.items():
                entities = entities.split(',')
                entities = set([word.strip() for word in entities])
                self.entity_types_to_entity[entity_type] = entities

        self.entity_to_entity_type = dict()
        self.words_to_find = list()
        for entity_type, entities in self.entity_types_to_entity.items():
            self.words_to_find += entities
            for word in entities:
                self.entity_to_entity_type[word] = entity_type

        self.all_words = dict()
        for entity in self.entity_to_entity_type.keys():
            self.all_words[entity] = [word.lemma_ for word in self.nlp(entity)]

        self.baseline_dataset = baseline_dataset
        self.ground_truth_annotation = None

    def construct(self):
        """Construct the gold standard dataset."""
        if self.baseline_dataset is None:
            self._create_baseline_dataset()
        else:
            self._create_ground_truth()

    @staticmethod
    def find_tokens(sentence, entity):
        """Give start and end of entity if exists in the sentence.

        Parameters
        ----------
        sentence: doc
            Sentence to find entity
        entity: list
            List of strings representing the entity to look at

        Returns
        -------
        start_chars: list
            Starting characters where to find the entity
        end_chars: list
            Ending characters where to find the entity

        Raises
        ------
        ValueError
            If the entity is not found in the sentence, ValueError is raised.
        """
        num_words = len(entity)
        start_chars, end_chars = list(), list()
        for i in range(len(sentence) - num_words):
            if [word.lemma_ for word in sentence[i:i + num_words]] == entity:
                start_chars += [sentence[i:i + 1].start_char]
                end_chars += [sentence[i + num_words - 1:i + num_words].end_char]

        if not start_chars:
            raise ValueError('We did not find the entity in this sentence')

        return start_chars, end_chars

    def _create_ground_truth(self):
        """Create ground_truth annotations only."""
        if isinstance(self.baseline_dataset, list):
            self.baseline_dataset = \
                self.sentences[self.sentences['sentence_id'].isin(self.baseline_dataset)].sort_values(['sentence_id'])

        ground_truth = list()
        for index, row in self.baseline_dataset.iterrows():
            sentence, sentence_id = row['text'], row['sentence_id']
            nlp_sentence = self.nlp(sentence)
            for word_ in self.all_words.keys():
                try:
                    start_chars, end_chars = self.find_tokens(nlp_sentence, self.all_words[word_])
                    for start, end in zip(start_chars, end_chars):
                        new_line = [{'sentence_id': sentence_id,
                                     'entity': word_,
                                     'start_char': start,
                                     'end_char': end,
                                     'entity_type': self.entity_to_entity_type[word_].upper()}]
                        ground_truth += new_line

                except ValueError:
                    continue

        self.ground_truth_annotation = pd.DataFrame(ground_truth)

    def _create_baseline_dataset(self, max_iter=1000000):
        """Create baseline_dataset and ground_truth annotations.

        Parameters
        ----------
        max_iter: int
            Number of sentences to go through to find the words_to_find
        """
        ground_truth = list()
        words_to_find = list(self.all_words.keys())
        baseline_dataset_ids = set()
        index = 0

        while words_to_find and index < max_iter:

            sentence_infos = self.sentences.iloc[index]
            sentence, sentence_id = sentence_infos['text'], sentence_infos['sentence_id']
            nlp_sentence = self.nlp(sentence)

            for word in words_to_find:
                try:
                    _, _ = self.find_tokens(nlp_sentence, self.all_words[word])
                    words_to_find.remove(word)
                    baseline_dataset_ids.add(sentence_id)

                    for word_ in self.all_words.keys():
                        try:
                            start_chars, end_chars = self.find_tokens(nlp_sentence, self.all_words[word_])
                            for start, end in zip(start_chars, end_chars):
                                new_line = [{'sentence_id': sentence_id,
                                             'entity': word_,
                                             'start_char': start,
                                             'end_char': end,
                                             'entity_type': self.entity_to_entity_type[word_].upper()}]
                                ground_truth += new_line

                        except ValueError:
                            continue
                    break

                except ValueError:
                    continue

            index += 1
            if index % 100000 == 0:
                print(index, ':', len(words_to_find), 'words to find / total is', len(self.all_words))

        self.baseline_dataset = \
            self.sentences[self.sentences['sentence_id'].isin(baseline_dataset_ids)].sort_values(['sentence_id'])
        self.baseline_dataset = self.baseline_dataset[['sentence_id', 'text']]
        self.ground_truth_annotation = pd.DataFrame(ground_truth)

    @staticmethod
    def highlight_sentences(sentence, annotations, entity_color_dict):
        """Highlight the entities in a sentence.

        Parameters
        ----------
        sentence: str
            Sentence to highlight
        annotations: list
            List of tuples (start_char, end_char, entity_type) corresponding
            to the sentence.
        entity_color_dict: dict
            Dictionary containing entity types as keys and their corresponding color
            as values.

        Returns
        -------
        annotated_text: str
            HTML format string with highlighted extracted entities
        """
        annotated_text = ''
        last_idx = 0

        sorted_annotations = sorted(annotations, key=lambda x: x[0])

        for start, end, entity_type in sorted_annotations:
            if start >= last_idx:
                color = entity_color_dict[entity_type]
                annotated_text += sentence[
                                  last_idx:start] + f"<span style='background: rgb({color[0]}, " \
                                                    f"{color[1]},{color[2]});" \
                                                    f" display: inline-block; padding: 0.2em 0.5em ; " \
                                                    f"border-radius: 7px'>{sentence[start:end]}</span>"
                last_idx = end
        annotated_text += sentence[last_idx:]
        return annotated_text

    def highlight_dataset(self):
        """Annotate the entire baseline_dataset.

        The method assumes that the number of entity types is less
        than 20 for the coloring purpose.

        Returns
        -------
        complete_text: str
            HTML format string with all the sentences of the dataset
            highlighted with the corresponding entities.
        """
        complete_text = ''

        color_dict = cm.get_cmap('tab20')
        color_list = color_dict.colors
        color_list = (np.array(color_list) * 255).astype(int)
        entity_types = self.ground_truth_annotation['entity_type'].unique()
        entity_color_dict = {entity_type: color_list[index, :] for index, entity_type in enumerate(entity_types)}

        ground_truth_df = self.ground_truth_annotation.sort_values(['sentence_id', 'start_char'])

        # Just to show the color of each entity types
        for entity_type, color in entity_color_dict.items():
            complete_text += f"<span style='background: rgb({color[0]}, {color[1]}, " \
                             f"{color[2]}); display: inline-block; padding: 0.2em 0.5em; " \
                             f"border-radius: 7px;'>{entity_type}</span>" + '<br>'

        # Annotation of all the sentences of the baseline dataset with the entities identified
        for sentence_num in range(len(self.baseline_dataset)):
            sentence_id, sentence = self.baseline_dataset.iloc[sentence_num]['sentence_id'], \
                                    self.baseline_dataset.iloc[sentence_num]['text']
            annotations = list()
            for index, entity in ground_truth_df[ground_truth_df['sentence_id'] == sentence_id].iterrows():
                annotations += [(entity['start_char'], entity['end_char'], entity['entity_type'])]
            output = self.highlight_sentences(sentence, annotations, entity_color_dict) + '<br>'
            complete_text += f'[{sentence_num} - sentence_id {sentence_id}] :  '
            complete_text += output

        return complete_text
