import re

from matplotlib import cm
import numpy as np
import pandas as pd


def create_baseline_dataset(db, words_to_find, entity_to_type, max_iter=1000000):
    """Create baseline_dataset and ground_truth annotations.

    Parameters
    ----------
    db: sqlite3.Cursor
        Database with the assumption that database has 'sentences' table
    words_to_find: list
        List with all the entities to find
    entity_to_type: dict
        Dictionary with keys corresponding to the different words_to_find
        and the values corresponding to their entity type.
    max_iter: int
        Number of sentences to go through to find the words_to_find

    Returns
    -------
    baseline_dataset_df: pd.DataFrame
        DataFrame containing sentences and sentence_id
    ground_truth_df: pd.DataFrame
        DataFrame containing all the entities extracted with their entity
        types, start_char, end_char and the corresponding sentence_id.
    """
    baseline_dataset = list()
    ground_truth = list()
    all_words = words_to_find.copy()
    print('Loading sentences ...')
    sentences = pd.read_sql("""SELECT * FROM sentences""", db)
    print('Sentences loaded. ')
    index = 0

    while words_to_find and index < max_iter:
        sentence_infos = sentences.iloc[index]
        sentence, sentence_id = sentence_infos['text'], sentence_infos['sentence_id']
        for word in words_to_find:
            if word in sentence:
                words_to_find.remove(word)
                new_sentence = [{'sentence_id': sentence_id,
                                 'text': sentence}]
                baseline_dataset += new_sentence

                # Second, check if other words are contained in the sentence
                for word in all_words:
                    start_position = [m.start() for m in re.finditer(word, sentence)]
                    for pos in start_position:
                        new_line = [{'sentence_id': sentence_id,
                                     'entity': word,
                                     'start_char': pos + 1,
                                     'end_char': pos + len(word) - 1,
                                     'entity_type': entity_to_type[word].upper()}]
                        ground_truth += new_line
                break

        index += 1
        if index % 100000 == 0:
            print(index)

        baseline_dataset_df = pd.DataFrame(baseline_dataset)
        ground_truth_df = pd.DataFrame(ground_truth)

        return baseline_dataset_df, ground_truth_df


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

    for start, end, entity_type in annotations:
        if start >= last_idx:
            color = entity_color_dict[entity_type]
            annotated_text += sentence[
                              last_idx:start] + f"<span style='color: rgb({color[0]}, {color[1]}, " \
                                                f"{color[2]}); font-size: 46px;'>{sentence[start:end]}</span>"
            last_idx = end
    annotated_text += sentence[last_idx:]
    return annotated_text


def highlight_dataset(baseline_dataset_df, ground_truth_df):
    """Annotate the entire baseline_dataset.

    The method assumes that the number of entity types is less
    than 20 for the coloring purpose.
    Parameters
    ----------
    baseline_dataset_df: pd.DataFrame
        DataFrame containing sentences and sentence_id
    ground_truth_df: pd.DataFrame
        DataFrame containing all the entities extracted with their entity
        types, start_char, end_char and the corresponding sentence_id.
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
    entity_types = ground_truth_df['entity_type'].unique()
    entity_color_dict = {entity_type: color_list[index, :] for index, entity_type in enumerate(entity_types)}

    for entity_type, color in entity_color_dict.items():
        complete_text += f"<span style='color: rgb({color[0]}, {color[1]}, " \
                         f"{color[2]}); font-size: 46px;'>{entity_type}</span>" + '<br>'
    for sentence_num in range(len(baseline_dataset_df)):
        sentence_id, sentence = baseline_dataset_df.iloc[sentence_num]['sentence_id'], \
                                baseline_dataset_df.iloc[sentence_num]['text']
        annotations = list()
        for index, entity in ground_truth_df[ground_truth_df['sentence_id'] == sentence_id].iterrows():
            annotations += [(entity['start_char'], entity['end_char'], entity['entity_type'])]
        output = highlight_sentences(sentence, annotations) + '<br>'
        complete_text += output

    return complete_text
