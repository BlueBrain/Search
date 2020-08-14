"""Parallel mining of named entities."""
import argparse
import collections
import io
import logging
import multiprocessing as mp
import queue
import time
import traceback

import pandas as pd
import spacy
import sqlalchemy
import sqlalchemy.sql

from bbsearch.mining import run_pipeline


class Miner:
    """Multiprocessing worker class for mining named entities.

    Parameters
    ----------
    name : str
        The name of this worker.
    engine_url : str
        The URL to the text database. Should be a valid argument for
        the `sqlalchemy.create_engine` function. The database should
        contain tables `articles` and `sentences`.
    model_path : str
        The path for loading the spacy model that will perform the
        named entity extraction.
    entity_map : dict[str, str]
        A map from entity types produced by the model to new
        entity types that should appear in the cached results.
    task_queue : multiprocessing.Queue
        The queue with tasks for this worker
    can_finish : multiprocessing.Event
        A flag to indicate that the worker can stop waiting for new
        tasks. Unless this flag is set, the worker will continue
        polling the task queue for new tasks.
    """

    def __init__(
        self,
        name,
        engine_url,
        model_path,
        entity_map,
        target_table,
        task_queue,
        can_finish,
    ):
        self.name = name
        self.engine = sqlalchemy.create_engine(engine_url)
        self.model_path = model_path
        self.entity_map = entity_map
        self.target_table_name = target_table
        self.task_queue = task_queue
        self.can_finish = can_finish

        self.n_tasks_done = 0

        self.logger = logging.getLogger(str(self))

        self.logger.info("Loading the NLP model...")
        self.model = spacy.load(self.model_path)
        self.model_meta = {
            "mining_model": self.model_path,
            # "mining_model_name": self.model.meta["name"],
            # "mining_model_version": self.model.meta["version"],
            # "spacy_version": self.model.meta["spacy_version"],
            # "spacy_git_version": self.model.meta["spacy_git_version"]
        }

        self.logger.info("Starting mining...")
        self._work_loop()

        self.logger.info("Finished mining, cleaning up...")
        self._clean_up()

    def _log_exception(self, article_id):
        error_trace = io.StringIO()
        traceback.print_exc(file=error_trace)

        error_message = f"\nArticle ID : {article_id}\n" + error_trace.getvalue()
        self.logger.error(error_message)

    def _work_loop(self):
        """Do the work loop."""
        finished = False
        while not finished:
            try:
                # this gets stuck if `block=False` isn't set, no idea why
                article_id = self.task_queue.get(block=False)
                try:
                    self._mine(article_id)
                    self.n_tasks_done += 1
                except Exception:
                    self._log_exception(article_id)
            except queue.Empty:
                self.logger.debug("Queue empty")
                if self.can_finish.is_set():
                    finished = True
                else:
                    time.sleep(1)

    def _fetch_all_paragraphs(self, article_id):
        query = """
                SELECT paragraph_pos_in_article, text
                FROM sentences
                WHERE article_id = :article_id
                ORDER BY paragraph_pos_in_article, sentence_pos_in_paragraph
                """
        result_proxy = self.engine.execute(
            sqlalchemy.sql.text(query), article_id=article_id
        )

        all_paragraphs = collections.defaultdict(list)
        for paragraph_pos, sentence in result_proxy:
            all_paragraphs[paragraph_pos].append(sentence)

        return all_paragraphs

    def _mine(self, article_id):
        """Perform one mining task.

        Parameters
        ----------
        article_id : int
            The article ID for mining.
        """
        self.logger.info(f"Processing article_id = {article_id}")

        self.logger.debug("Getting all sentences from the database...")
        all_paragraphs = self._fetch_all_paragraphs(article_id)

        self.logger.debug("Constructing texts...")
        texts = []
        for paragraph_pos in sorted(all_paragraphs):
            sentences = all_paragraphs[paragraph_pos]
            paragraph = " ".join(sentences)
            metadata = {
                "article_id": article_id,
                "paragraph_pos_in_article": paragraph_pos,
                **self.model_meta,
            }
            texts.append((paragraph, metadata))

        self.logger.debug("Running the pipeline...")
        df_results = run_pipeline(
            texts=texts, model_entities=self.model, models_relations={}, debug=True
        )

        self.logger.debug("Filtering entity types...")
        # Keep only the entity types we care about
        rows_to_keep = df_results["entity_type"].isin(self.entity_map)
        df_results = df_results[rows_to_keep]

        # Map model's names for entity types to desired entity types
        df_results["entity_type"] = df_results["entity_type"].apply(
            lambda entity_type: self.entity_map[entity_type]
        )

        self.logger.debug("Writing results to the SQL database...")
        df_results.to_sql(
            self.target_table_name, con=self.engine, if_exists="append", index=False
        )

        self.logger.info(f"Mined {len(df_results)} entities.")

    def _clean_up(self):
        """Clean up after task processing has been finished."""
        self.logger.info(f"I'm proud to have done {self.n_tasks_done} tasks!")

    def __str__(self):
        """Represent self as string.

        Returns
        -------
        str
            The string representation of self.
        """
        return self.__repr__()

    def __repr__(self):
        """Represent self.

        Returns
        -------
        str
            The representation of self.
        """
        return f"{self.__class__.__name__}[{self.name}]"


class MinerMaster:
    """The master class for creating the mining cache.

    Parameters
    ----------
    workers_per_model : int
        The number of workers per model.
    database_url : str
        URL to the MySQL CORD-19 database.
    model_schemas : dict
        The models to be used for mining. The keys of the dictionary
        are arbitrary names for the models. The values are model
        paths or names that will be used in `spacy.load(...)`.
    target_table : str
        The SQL table for writing the mining results.
    """

    def __init__(self, workers_per_model, database_url, model_schemas, target_table):
        self.workers_per_model = workers_per_model
        self.database_url = database_url
        self.model_schemas = model_schemas
        self.target_table = target_table
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_tasks(self, task_queues):
        """Create tasks for the mining workers.

        Parameters
        ----------
        task_queues : dict[str or pathlib.Path, multiprocessing.Queue]
            Task queues for different models. The keys are the model
            paths and the values are the actual queues.

        """
        self.logger.info("Getting all article IDs...")
        engine = sqlalchemy.create_engine(self.database_url)
        result_proxy = engine.execute(
            "SELECT article_id FROM articles ORDER BY article_id"
        )
        all_article_ids = [row["article_id"] for row in result_proxy]

        # We got some new tasks, put them in the task queues.
        self.logger.info("Adding new tasks...")
        for model_path in task_queues:
            for article_id in all_article_ids:
                task_queues[model_path].put(article_id)

    def do_mining(self):
        """Do the parallelized mining."""
        self.logger.info(
            f"Starting mining with {self.workers_per_model} workers per model."
        )

        # A flag to let the workers know there won't be any new tasks.
        can_finish = mp.Event()

        # Prepare the task queues for the workers - one task queue per model.
        task_queues = {model_name: mp.Queue() for model_name in self.model_schemas}

        # Spawn the workers according to `workers_per_model`.
        self.logger.info("Spawning the worker processes...")
        worker_processes = []
        for model_name, model_schema in self.model_schemas.items():
            model_path = model_schema["model_path"]
            entity_map = model_schema["entity_map"]
            for i in range(self.workers_per_model):
                worker_name = f"{model_name}_{i}"
                worker_process = mp.Process(
                    name=worker_name,
                    target=Miner,
                    kwargs={
                        "name": worker_name,
                        "engine_url": self.database_url,
                        "model_path": model_path,
                        "entity_map": entity_map,
                        "target_table": self.target_table,
                        "task_queue": task_queues[model_name],
                        "can_finish": can_finish,
                    },
                )
                worker_process.start()
                worker_processes.append(worker_process)

        # Create tasks
        self.logger.info("Creating tasks...")
        self.create_tasks(task_queues)

        # Wait for the processes to finish.
        can_finish.set()
        self.logger.info("No more new tasks, just waiting for the workers to finish...")
        for process in worker_processes:
            process.join()

        # Evaluate workers' exit codes.
        for process in worker_processes:
            if process.exitcode != 0:
                self.logger.error(
                    f"Worker {process.name} terminated with exit code {process.exitcode}!"
                )

        self.logger.info("Finished.")


def get_cord19_db_url():
    """Construct the URL to the MySQL CORD-19 database.

    Returns
    -------
    url : str
        The MySQL CORD-19 database URL.
    """
    protocol = "mysql+pymysql"
    host = "dgx1.bbp.epfl.ch"
    port = 8853
    user = "stan"
    pw = "letmein"
    db = "cord19_v35"

    url = f"{protocol}://{user}:{pw}@{host}:{port}/{db}"

    return url


def load_model_schemas(schema_path):
    """Load the model schemas from a file.

    Parameters
    ----------
    schema_path : str or pathlib.Path
        The path to the CSV file containing the model schemas.
        It should contain the following three columns:
            1. Public entity type
            2. Model path
            3. Model's internal entity type

    Returns
    -------
    model_schemas : dict
        The model schemas in a dictionary of the following form:
            model_schemas = {
                "model1_bc5cdr_annotations5_spacy23": {
                    "model_path": "/path/to/model",
                    "entity_map": {
                        "model_entity_type_1": "public_entity_type_1",
                        "model_entity_type_2": "public_entity_type_2",
                    },
                },
                "en_ner_bionlp13cg_md": {...},
            }
        The keys of this dictionary are model names produces form the
        model paths by taking all characters that follow the last "/"
        character, or the whole model path if there is no "/" character
        in the model path.
    """
    schema_df = pd.read_csv(schema_path)
    model_schemas = dict()
    for entity_type_to, model_path, entity_type_from in schema_df.itertuples(
        index=False
    ):
        _, _, model_name = model_path.rpartition("/")
        if model_name not in model_schemas:
            model_schemas[model_name] = dict()
            model_schemas[model_name]["model_path"] = model_path
            model_schemas[model_name]["entity_map"] = dict()

        model_schemas[model_name]["entity_map"][entity_type_from] = entity_type_to

    return model_schemas


def filter_model_schemas(model_schemas, available_models):
    """Filter model schemas and replace model paths.

    If only a subset of models in model schemas is available,
    and/or the local model paths are different to those in
    model_schemas then this function can be used to select
    a subset of models from the model schemas and to adjust
    their paths to the local paths.

    The keys in both `model_schemas` and `available_models`
    are model names constructed as described in `load_model_schemas`.

    Parameters
    ----------
    model_schemas : dict[str, dict]
        The model schemas as returned by `load_model_schemas`.
    available_models : dict[str, str or pathlib.Path]
        A set of available models that should be selected from
        the model schemas. It has the following form:
            available_models = {
                "model_name_1": "path_to_model_1",
                "model_name_2": "path_to_model_2",
            }

    Returns
    -------
    filtered_model_schemas : dict[str, dict]
        The updated version of `model_schemas`.
    """
    filtered_model_schemas = dict()
    for model_name in model_schemas:
        if model_name in available_models:
            filtered_model_schemas[model_name] = model_schemas[model_name]
            filtered_model_schemas[model_name]["model_path"] = available_models[
                model_name
            ]

    return filtered_model_schemas


def main(workers_per_model, verbose, log_file_name=None):
    """Run the main entrypoint.

    Parameters
    ----------
    workers_per_model : int
        The number of workers per model.
    verbose : int
        The logging level, -v corresponds to INFO, -vv to DEBUG.
    log_file_name : str
        The file for the logs. If not provided the stdout will be used.
    """
    # Set logging level
    logging.basicConfig(
        filename=log_file_name,
        format="%(asctime)s :: %(levelname)-8s :: %(name)s | %(message)s",
    )
    if verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)

    # Prepare the model schemas
    available_models = {
        "model1_bc5cdr_annotations5_spacy23": "assets/model1_bc5cdr_annotations5_spacy23",
        "en_ner_bionlp13cg_md": "en_ner_bionlp13cg_md",
    }
    # schema_path = "/raid/sync/proj115/bbs_data/models_libraries/ee_models_library.csv"
    schema_path = "assets/ee_models_library.csv"
    target_table = "mining_cache_temp"

    model_schemas = load_model_schemas(schema_path)
    model_schemas = filter_model_schemas(model_schemas, available_models)

    # Launch the mining
    miner_master = MinerMaster(
        workers_per_model=workers_per_model,
        database_url=get_cord19_db_url(),
        model_schemas=model_schemas,
        target_table=target_table,
    )
    miner_master.do_mining()


parser = argparse.ArgumentParser(description="Parallel mining.")
parser.add_argument(
    "--workers_per_model",
    "-w",
    type=int,
    default=mp.cpu_count(),
    help="The number of worker processes to spawn for each mining model.",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="count",
    default=0,
    help="The logging level, -v correspond to INFO, -vv to DEBUG",
)
parser.add_argument(
    "--log_file",
    "-l",
    type=str,
    default=None,
    help="The file for the logs. If not provided the stdout will be used.",
)
args = parser.parse_args()
if __name__ == "__main__":
    main(args.workers_per_model, args.verbose, args.log_file)
