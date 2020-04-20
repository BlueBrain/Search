import hashlib
import json
import logging
import sqlite3

import numpy as np
import nltk
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AllData:

    def __init__(self, data_path, assets_path, cord_path=None, databases_path=None,
                 embeddings_path=None):
        nltk.download('punkt')
        nltk.download('stopwords')

        # Data paths
        assert data_path.exists()
        assert assets_path.exists()

        self.data_path = data_path
        self.assets_path = assets_path
        self.cord_path = cord_path or data_path / "CORD-19-research-challenge"
        self.databases_path = databases_path or data_path / "databases"
        self.embeddings_path = embeddings_path or data_path / "embeddings"

        assert self.cord_path.exists()
        assert self.databases_path.exists()
        assert self.embeddings_path.exists()

        # SQL Database
        self.db = sqlite3.connect(str(self.databases_path / "articles.sqlite"))

        # Metadata CSV file
        self.df_metadata_original = pd.read_csv(self.cord_path / "metadata.csv")

        # Remove rows with both no title and no SHA
        mask_useless = self.df_metadata_original['title'].isna()
        mask_useless &= self.df_metadata_original['sha'].isna()
        self.df_metadata = self.df_metadata_original[~mask_useless]

        # Generate fake SHAs for entries that do not have full-text
        mask = self.df_metadata['sha'].isna()
        self.df_metadata.loc[mask, 'sha'] = self.df_metadata.loc[mask, 'title'].apply(
            lambda text: hashlib.sha1(str(text).encode("utf-8")).hexdigest())
        self.df_metadata.head(2)

        # Load JSON Files
        self.n_json = len(list(data_path.rglob("*.json")))
        self.json_files = []

        for f in tqdm(data_path.rglob("*.json"), total=self.n_json):
            self.json_files.append(json.load(open(f)))

        # Fill in missing titles from the metadata
        for json_file in tqdm(self.json_files):
            if json_file['metadata']['title'] == '':
                sha = json_file['paper_id']
                idx = np.where(self.df_metadata['sha'] == sha)[0]
                if len(idx) > 0:
                    new_title = self.df_metadata['title'].iloc[idx[0]]
                    json_file['metadata']['title'] = new_title

        # Create a dictionary with JSON files based on their SHAs
        self.json_files_d = {
            json_file['paper_id']: json_file
            for json_file in self.json_files
        }

    def find_paragraph(self, uid, sentence):
        """Find the paragraph corresponding to the given sentece

        Parameters
        ----------
        uid : int
            The identifier of the given sentence
        sentence: str
            The sentence to highlight
        db: sqlite3.Connection
            The database connection

        Returns
        -------
        formatted_paragraph : str
            The paragraph containing `sentence`
        """

        sha, where_from = \
            self.db.execute(f'SELECT Article, Name FROM sections WHERE Id = {uid}').fetchall()[0]
        logger.debug(f"uid = {uid}")
        logger.debug(f"sha = {sha}")
        logger.debug(f"where_from = {where_from}")
        logger.debug(f"sentence = {sentence}")
        if sha in list(self.df_metadata['sha']) and where_from in ['TITLE', 'ABSTRACT']:
            df_row = self.df_metadata[self.df_metadata['sha'] == sha].iloc[0]
            if sentence in df_row['title']:
                paragraph = df_row['title']
            elif sentence in df_row['abstract']:
                paragraph = df_row['abstract']
            else:
                raise ValueError("Sentence not found in title nor in abstract")
        elif sha in self.json_files_d:
            json_file = self.json_files_d[sha]
            if sentence in json_file['metadata']['title']:
                paragraph = json_file['metadata']['title']
            else:
                for text_chunk in json_file['abstract'] + json_file['body_text']:
                    paragraph = text_chunk['text']
                    if sentence in paragraph:
                        break
                else:
                    raise ValueError("sentence not found in body_text and abstract")
        else:
            raise ValueError("SHA not found")

        return paragraph

    def highlight_in_paragraph(self, paragraph, sentence, width=80, indent=0):
        """Highlight a given sentence in the paragraph.

        Parameters
        ----------
        uid : int
            The identifier of the given sentence.
        sentence: str
            The sentence to highlight.
        width : int
            The width to which to wrapt the returned paragraph.
        indent : int
            The indentation for the lines in the returned apragraph.

        Returns
        -------
        formatted_paragraph : str
            The paragraph containing `sentence` with the sentence highlighted
            in color
        """
        COLOR_TEXT = '#222222'
        COLOR_HIGHLIGHT = '#000000'

        start = paragraph.index(sentence)
        end = start + len(sentence)
        hightlighted_paragraph = f'''
            <p style="font-size:13px; color:{COLOR_TEXT}">
            {paragraph[:start]}
            <b style="color:{COLOR_HIGHLIGHT}"> {paragraph[start:end]} </b>
            {paragraph[end:]}
            </p>
            '''
        #     wrapped_lines = textwrap.wrap(hightlighted_paragraph, width=width)
        #     wrapped_lines = [' ' * indent + line for line in wrapped_lines]
        #     formatted_paragraph = '\n'.join(wrapped_lines)

        return hightlighted_paragraph
