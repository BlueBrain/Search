"""Parallel mining of named entities."""
import argparse
import logging
import multiprocessing as mp
import queue
import time

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
    task_queue : multiprocessing.Queue
        The queue with tasks for this worker
    can_finish : multiprocessing.Value
        An integer shared value to indicate that the worker can finish
        waiting. Unless `can_finish` is equal to `1`, the worker will
        continue polling the task queue for new tasks.
    logging_level : int, optional
        The logging level for the internal logger
    """

    def __init__(
        self, name, engine_url, model_path, task_queue, can_finish, logging_level=None
    ):
        self.name = name
        self.create_engine = sqlalchemy.create_engine(engine_url)
        self.engine = self.create_engine
        self.model_path = model_path
        self.work_queue = task_queue
        self.can_finish = can_finish

        self.n_tasks_done = 0

        self.logger = logging.getLogger(str(self))
        if logging_level is not None:
            self.logger.setLevel(logging_level)

        self.logger.info("loading the NLP model...")
        self.ee_model = spacy.load(model_path)

        self.logger.info("starting mining...")
        self._work_loop()

        self.logger.info("finished mining, cleaning up...")
        self._clean_up()

        self.logger.info("all done.")

    def _work_loop(self):
        """Do the work loop."""
        finished = False
        while not finished:
            try:
                # this gets stuck if `block=False` isn't set, no idea why
                article_id = self.work_queue.get(block=False)
                self._mine(article_id)
                self.n_tasks_done += 1
            except queue.Empty:
                self.logger.info("Queue empty")
                if self.can_finish.value == 1:
                    finished = True
                else:
                    time.sleep(1)

    def _mine(self, article_id):
        """Perform one mining task.

        Parameters
        ----------
        article_id : int
            The article ID for mining.
        """
        self.logger.info(f"Processing article_id = {article_id}")

        self.logger.info("Getting all sentences from the database...")
        query = """
        SELECT paragraph_pos_in_article, sentence_pos_in_paragraph, text
        FROM sentences
        WHERE article_id = :article_id
        """
        safe_query = sqlalchemy.sql.text(query)
        result_proxy = self.engine.execute(safe_query, article_id=article_id)

        all_sentences = dict()
        for paragraph_pos, sentence_pos, text in result_proxy:
            all_sentences[(paragraph_pos, sentence_pos)] = text

        texts = [(all_sentences[key], {}) for key in sorted(all_sentences)]

        self.logger.info("Running the pipeline...")
        df_results = run_pipeline(
            texts=texts, model_entities=self.ee_model, models_relations={}, debug=False
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


def get_engine_url():
    """Get the URL for the MySQL CORD-19 database.

    Returns
    -------
    engine_url : str
        The database URL.
    """
    mysql_host = "dgx1.bbp.epfl.ch"
    mysql_port = 8853
    database_name = "cord19_v35"
    engine_url = (
        f"mysql+mysqldb://guest:guest@{mysql_host}:{mysql_port}/{database_name}"
    )

    return engine_url


def create_tasks(task_queues):
    """Create tasks for the mining workers.

    Parameters
    ----------
    task_queues : dict[str or pathlib.Path, multiprocessing.Queue]
        Task queues for different models. The keys are the model
        paths and the values are the actual queues.

    """
    print("Getting all article IDs...")
    engine = sqlalchemy.create_engine(get_engine_url())
    result = engine.execute("select article_id from articles")
    all_article_ids = sorted([row[0] for row in result.fetchall()])

    # # Pretend we're doing something else while the workers are working.
    # print("Waiting a bit...")
    # time.sleep(15)

    # We got some new tasks, put them in the task queues.
    print("Adding new tasks...")
    for model_path in task_queues:
        for article_id in all_article_ids[:100]:
            task_queues[model_path].put(article_id)

    # # Again pretend we're busy.
    # print("Wait again...")
    # time.sleep(15)


def do_mining(models, workers_per_model):
    """Do the parallelized mining.

    Parameters
    ----------
    models : dict[str, str]
        The models to be used for mining. The keys of the dictionary
        are arbitrary names for the models. The values are model
        paths or names that will be used in `spacy.load(...)`.
    workers_per_model : int
        The number of workers to spawn for each model.
    """
    # Prepare as many workers as necessary according to `workers_per_model`.
    # Also instantiate one task queue per model.
    print("Preparing the workers...")
    task_queues = {model_name: mp.Queue() for model_name in models}
    workers_info = []
    for model_name, model_path in models.items():
        for i in range(workers_per_model):
            worker_name = f"{model_name}_{i}"
            workers_info.append((worker_name, model_path, task_queues[model_name]))

    # A shared integer value to let the workers know when it's finished.
    # The workers will never write to this value, so we don't need locks.
    can_finish = mp.Value("i", 0, lock=False)

    # Create the worker processes.
    print("Spawning the worker processes...")
    worker_processes = []
    for worker_name, model_path, task_queue in workers_info:
        worker_process = mp.Process(
            name=worker_name,
            target=Miner,
            args=(worker_name, get_engine_url(), model_path, task_queue, can_finish,),
        )
        worker_process.start()
        worker_processes.append(worker_process)

    # Create tasks
    print("Creating tasks...")
    create_tasks(task_queues)

    # This operation should be atomic, so don't need a lock.
    can_finish.value = 1

    # Wait for the processes to finish.
    print("No more new tasks, just waiting for the workers to finish...")
    for process in worker_processes:
        process.join()
        if process.exitcode != 0:
            logging.warning(
                f"Worker {process.name} terminated with exit code {process.exitcode}!"
            )

    print("Finished.")


def main(workers_per_model, verbose=False):
    """Run main.

    Parameters
    ----------
    workers_per_model : int
        The number of workers per model.
    verbose : bool
        If true then the logging level is set to `logging.INFO`.
    """
    logging.basicConfig()
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    models = {
        "bc5cdr_annotations5": "assets/model1_bc5cdr_annotations5_spacy23",
        "en_ner_bionlp13cg_md": "en_ner_bionlp13cg_md",
    }
    do_mining(models, workers_per_model=workers_per_model)


parser = argparse.ArgumentParser(description="Parallel mining.")
parser.add_argument("--workers_per_model", "-w", type=int, default=mp.cpu_count())
parser.add_argument("--verbose", "-v", action="store_true", default=False)
args = parser.parse_args()
if __name__ == "__main__":
    print(f"Running with {args.workers_per_model} workers per model.")
    main(args.workers_per_model, verbose=args.verbose)
