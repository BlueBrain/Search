import hashlib
import json
import logging
import sqlite3

import nltk
import pandas as pd

logger = logging.getLogger(__name__)


class AllData:

    def __init__(self, data_path, assets_path,
                 cord_path=None, databases_path=None, embeddings_path=None):
        logger.info("Setting data paths...")
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

        logger.info("Connecting to the SQLite database...")
        self.db = sqlite3.connect(str(self.databases_path / "articles.sqlite"))

        logger.info("Reading the metadata.csv file...")
        self.df_metadata_original = pd.read_csv(self.cord_path / "metadata.csv")

        logger.info("Removing rows with both no title and no SHA...")
        mask_useless = self.df_metadata_original['title'].isna()
        mask_useless &= self.df_metadata_original['sha'].isna()
        self.df_metadata = self.df_metadata_original[~mask_useless]

        logger.info("Generate fake SHAs for entries that do not have full-text...")
        mask = self.df_metadata['sha'].isna()
        self.df_metadata.loc[mask, 'sha'] = self.df_metadata.loc[mask, 'title'].apply(
            lambda text: hashlib.sha1(str(text).encode("utf-8")).hexdigest())
        self.df_metadata.head(2)

        logger.info("Loading the JSON file paths...")
        self.json_paths = {json_path.stem: str(json_path)
                           for json_path in data_path.rglob("*.json")}

    def find_paragraph(self, uid, sentence):
        """Find the paragraph corresponding to the given sentence

        Parameters
        ----------
        uid : int
            The identifier of the given sentence
        sentence: str
            The sentence to highlight

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
        elif sha in self.json_paths:
            json_path = self.json_paths[sha]
            with open(json_path, 'r') as f:
                json_file = json.load(f)
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

    @staticmethod
    def highlight_in_paragraph(paragraph, sentence):
        """Highlight a given sentence in the paragraph.

        Parameters
        ----------
        paragraph : str
            The paragraph in which to highlight the sentence.
        sentence: str
            The sentence to highlight.

        Returns
        -------
        formatted_paragraph : str
            The paragraph containing `sentence` with the sentence highlighted
            in color
        """
        color_text = '#222222'
        color_highlight = '#000000'

        start = paragraph.index(sentence)
        end = start + len(sentence)
        highlighted_paragraph = f"""
            <p style="font-size:13px; color:{color_text}">
            {paragraph[:start]}
            <b style="color:{color_highlight}"> {paragraph[start:end]} </b>
            {paragraph[end:]}
            </p>
            """

        return highlighted_paragraph
