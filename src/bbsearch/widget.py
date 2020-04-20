from collections import OrderedDict
import datetime
import logging
import pdfkit
import textwrap
import time

import ipywidgets as widgets
from IPython.display import HTML, display
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .sql import ArticleConditioner, SentenceConditioner
from .sql import get_ids_by_condition

logger = logging.getLogger(__name__)


class Widget:

    def __init__(self, all_data, all_models):
        self.all_data = all_data
        self.all_models = all_models
        self.report = None
        self.my_widgets = OrderedDict()

        self.initialize_widgets()

    def initialize_widgets(self):
        # Select model to compute Sentence Embeddings
        self.my_widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['USE', 'SBERT', 'BSV', 'SBIOBERT'],
            description='Model for Sentence Embedding',
            tooltips=['Universal Sentence Encoder', 'Sentence BERT', 'BioSentVec',
                      'Sentence BioBERT'],
        )

        # Select n. of top results to return
        self.my_widgets['top_results'] = widgets.widgets.IntSlider(
            value=10,
            min=0,
            max=100,
            description='Top N results'
        )

        # Choose whether to merge synonyms or not
        self.my_widgets['merge_synonyms'] = widgets.Checkbox(
            value=False,
            description='Merge synonyms'
        )

        # Choose whether to print whole paragraph containing sentence highlighted, or just the
        # sentence
        self.my_widgets['print_paragraph'] = widgets.Checkbox(
            value=True,
            description='Show whole paragraph'
        )

        # Enter Query
        self.my_widgets['query_text'] = widgets.Textarea(
            value='Glucose is a risk factor for COVID-19',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Query'
        )

        # Filtering widgets
        # self.my_widgets['has_title'] = widgets.Checkbox(
        #     description="Require Title",
        #     value=True)
        # self.my_widgets['has_authors'] = widgets.Checkbox(
        #     description="Require Authors",
        #     value=True)
        # self.my_widgets['has_abstract'] = widgets.Checkbox(
        #     description="Require Abstract",
        #     value=False)
        self.my_widgets['has_journal'] = widgets.Checkbox(
            description="Require Journal",
            value=False)
        # self.my_widgets['has_doi'] = widgets.Checkbox(
        #     description="Require DOI",
        #     value=False)
        self.my_widgets['date_range'] = widgets.IntRangeSlider(
            description="Date Range:",
            continuous_update=False,
            min=1900,
            max=2020,
            value=(1900, 2020),
            layout=widgets.Layout(width='80ch'))

        # Enter Deprioritization Query
        self.my_widgets['deprioritize_text'] = widgets.Textarea(
            value='',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Deprioritize'
        )

        # Select Deprioritization Strength
        self.my_widgets['deprioritize_strength'] = widgets.ToggleButtons(
            options=['None', 'Weak', 'Mild', 'Strong', 'Stronger'],
            disabled=False,
            button_style='info',
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength',
        )

        # Enter Substrings Exclusions
        self.my_widgets['exclusion_text'] = widgets.Textarea(
            layout=widgets.Layout(width='90%', height='80px'),
            value='',
            style={'description_width': 'initial'},
            description='Substring Exclusion (newline separated): '
        )

        # Click to run Information Retrieval!
        self.my_widgets['investigate_button'] = widgets.Button(description='Investigate!')

        # Click to run Generate Report!
        self.my_widgets['report_button'] = widgets.Button(description='Generate PDF Report!',
                                                          layout=widgets.Layout(width='25%'))

        # Output Area
        self.my_widgets['out'] = widgets.Output(layout={'border': '1px solid black'})

        # Callbacks
        self.my_widgets['investigate_button'].on_click(self.investigate_on_click)
        self.my_widgets['report_button'].on_click(self.report_on_click)

    def investigate_on_click(self, b):
        self.my_widgets['out'].clear_output()
        with self.my_widgets['out']:
            self.report = ''
            print()
            t0 = time.time()

            sentence_embedder_name = self.my_widgets['sent_embedder'].value
            merge_synonyms = self.my_widgets['merge_synonyms'].value
            top_n_results = self.my_widgets['top_results'].value
            print_whole_paragraph = self.my_widgets['print_paragraph'].value
            query_text = self.my_widgets['query_text'].value
            deprioritize_text = self.my_widgets['deprioritize_text'].value
            deprioritize_strength = self.my_widgets['deprioritize_strength'].value
            exclusion_text = self.my_widgets['exclusion_text'].value

            if merge_synonyms:
                query_text = self.all_models.sent_preprocessing(
                    [query_text], self.all_models.synonyms_index)
                deprioritize_text = self.all_models.sent_preprocessing(
                    [deprioritize_text], self.all_models.synonyms_index)
            else:
                query_text = [query_text]
                deprioritize_text = [deprioritize_text]

            print('Embedding query...    ', end=' ')
            embedding_query = self.all_models.embed_sentences(
                query_text,
                sentence_embedder_name,
                getattr(self.all_models, sentence_embedder_name.lower())
            )
            print(f'{time.time() - t0:.2f} s.')

            if deprioritize_text[0]:
                print('Embedding deprioritization...', end=' ')
                embedding_exclu = self.all_models.embed_sentences(
                    deprioritize_text,
                    sentence_embedder_name,
                    getattr(self.all_models, sentence_embedder_name.lower())
                )
                print(f'{time.time() - t0:.2f} s.')

            # Process date range and has-journal filtering
            has_journal = self.my_widgets['has_journal'].value
            date_range = self.my_widgets['date_range'].value

            # Apply article conditions
            article_conditions = [
                ArticleConditioner.get_date_range_condition(date_range)]
            if has_journal:
                article_conditions.append(ArticleConditioner.get_has_journal_condition())
            restricted_article_ids = get_ids_by_condition(
                article_conditions,
                'articles',
                self.all_data.db)

            # Apply sentence conditions
            all_aticle_ids_str = ', '.join([f"'{sha}'" for sha in restricted_article_ids])
            sentence_conditions = [
                f"Article IN ({all_aticle_ids_str})",
                SentenceConditioner.get_restrict_to_tag_condition("COVID-19")
            ]
            excluded_words = [x for x in exclusion_text.lower().split('\n')
                              if x]  # remove empty strings
            sentence_conditions += [
                SentenceConditioner.get_word_exclusion_condition(word)
                for word in excluded_words]
            restricted_sentence_ids = get_ids_by_condition(
                sentence_conditions,
                'sections',
                self.all_data.db)

            #             n_articles = db.execute("SELECT COUNT(*) FROM articles").fetchone()
            #             n_sentences = db.execute("SELECT COUNT(*) FROM sections").fetchone()
            #             n_articles = n_articles[0]
            #             n_sentences = n_sentences[0]
            #             frac_articles = len(restricted_article_ids) / n_articles
            #             frac_sentences = len(restricted_sentence_ids) / n_sentences
            #             print(f"Selected {len(restricted_article_ids)} of {n_articles} "
            #                   f"articles ({frac_articles * 100:.1f}%).")
            #             print(f"Selected {len(restricted_sentence_ids)} of {n_sentences} "
            #                   f"sentences ({frac_sentences * 100:.1f})%.")

            # Load sentence embedding from the npz file
            if merge_synonyms:
                arr = self.all_models.embeddings_syns[sentence_embedder_name]
            else:
                arr = self.all_models.embeddings[sentence_embedder_name]

            # Apply date-range and has-journal filtering to arr
            idx_col = arr[:, 0]
            mask = np.isin(idx_col, restricted_sentence_ids)
            arr = arr[mask]
            if len(arr) == 0:
                print("No documents left after filtering!")
                return

            # Compute similarities
            print('Computing similarities...', end=' ')
            sentence_ids, embeddings_corpus = arr[:, 0], arr[:, 1:]
            similarities_query = cosine_similarity(X=embedding_query,
                                                   Y=embeddings_corpus).squeeze()

            if deprioritize_text[0]:
                similarities_exclu = cosine_similarity(X=embedding_exclu,
                                                       Y=embeddings_corpus).squeeze()
            else:
                similarities_exclu = np.zeros_like(similarities_query)

            deprioritizations = {
                'None': (1, 0),
                'Weak': (0.9, 0.1),
                'Mild': (0.8, 0.3),
                'Strong': (0.5, 0.5),
                'Stronger': (0.5, 0.7),
            }
            # now: maximize L = a1 * cos(x, query) - a2 * cos(x, exclusions)
            alpha_1, alpha_2 = deprioritizations[deprioritize_strength]
            similarities = alpha_1 * similarities_query - alpha_2 * similarities_exclu

            print(f'{time.time() - t0:.2f} s.')

            print('Ranking documents...     ', end=' ')
            indices = np.argsort(-similarities)
            indices = indices[:top_n_results]

            print(f'{time.time() - t0:.2f} s.')

            print(f'\nInvestigating: {query_text[0]}\n')

            for i, (sentence_id_, sim_) in enumerate(zip(sentence_ids[indices],
                                                         similarities[indices])):
                article_sha, section_name, text = \
                    self.all_data.db.execute(
                        'SELECT Article, Name, Text FROM sections WHERE Id = ?',
                        [sentence_id_]).fetchall()[0]
                article_auth, article_title, date, ref = self.all_data.db.execute(
                    'SELECT Authors, Title, Published, Reference FROM articles WHERE Id = ?',
                    [article_sha]).fetchall()[0]
                article_auth = article_auth.split(';')[0] + ' et al.'
                date = date.split()[0]
                ref = ref if ref else ''
                section_name = section_name if section_name else ''

                width = 80
                if print_whole_paragraph:
                    logger.debug(f"UID={sentence_id_}")
                    try:
                        paragraph = self.all_data.find_paragraph(sentence_id_, text)
                        formatted_output = self.all_data.highlight_in_paragraph(
                            paragraph, text, width=width, indent=2)
                    except:
                        formatted_output = "<there was a problem retrieving the paragraph, " \
                                           "the original sentence is:>\n"
                        formatted_output += text
                else:
                    formatted_output = textwrap.fill(text, width=width)

                COLOR_TITLE = '#1A0DAB'
                COLOR_METADATA = '#006621'
                article_metadata = f"""
                <a href="{ref}" style="color:{COLOR_TITLE}; font-size:17px"> 
                    {article_title}
                </a>
                <br>
                <p style="color:{COLOR_METADATA}; font-size:13px"> 
                    {article_auth} &#183; {section_name.lower().title()}
                </p>
                """

                display(HTML(article_metadata))
                display(HTML(formatted_output))
                print()

                self.report += article_metadata + formatted_output + "<br>"

    def report_on_click(self, b):
        print("Saving results to a pdf file.")

        COLOR_HYPERPARAMETERS = '#222222'

        hyperparameters_section = f'<h1> Search Parameters </h1>' + \
                                  f'<ul style="font-size:13; color:{COLOR_HYPERPARAMETERS}">' + \
                                  '<li>' + '</li> <li>'.join(['<b>' +
                                                              ' '.join(k.split('_')).title() +
                                                              '</b>' +
                                                              f': {repr(v.value)}'
                                                              for k, v in self.my_widgets.items()
                                                              if hasattr(v, 'value')]) + '</li>' + \
                                  f'</ul>'

        results_section = f"<h1> Results </h1> {self.report}"
        pdfkit.from_string(hyperparameters_section + results_section,
                           f"report_{datetime.datetime.now()}.pdf")

    def display(self):
        display(widgets.VBox(list(self.my_widgets.values())))

