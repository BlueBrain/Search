"""Module for the Database Creation."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import io
import logging
import multiprocessing as mp
import queue
import traceback
from typing import Dict, List

import sqlalchemy

from ..mining.pipeline import run_pipeline
from ..sql import retrieve_articles
from ..utils import load_spacy_model


class Miner:
    """Multiprocessing worker class for mining named entities.

    Parameters
    ----------
    database_url : str
        URL of a database already containing tables `articles` and `sentences`.
        The URL should indicate database dialect and connection argument, e.g.
        `database_url = "postgresql://scott:tiger@localhost/test"`.
    model_path : str
        The path for loading the spacy model that will perform the
        named entity extraction.
    target_table : str
        The target table name for the mining results.
    task_queue : multiprocessing.Queue
        The queue with tasks for this worker
    can_finish : multiprocessing.Event
        A flag to indicate that the worker can stop waiting for new
        tasks. Unless this flag is set, the worker will continue
        polling the task queue for new tasks.
    """

    def __init__(
        self,
        database_url,
        model_path,
        target_table,
        task_queue,
        can_finish,
    ):
        self.name = mp.current_process().name
        self.engine = sqlalchemy.create_engine(database_url)
        self.model_path = model_path
        self.target_table_name = target_table
        self.task_queue = task_queue
        self.can_finish = can_finish

        self.n_tasks_done = 0

        self.logger = logging.getLogger(str(self))

        self.logger.info("Disposing of existing connections in engine")
        # This is important for multiprocessing
        self.engine.dispose()

        self.logger.info("Loading the NLP model")
        self.model = load_spacy_model(self.model_path)

    @classmethod
    def create_and_mine(
        cls,
        database_url,
        model_path,
        target_table,
        task_queue,
        can_finish,
    ):
        """Create a miner instance and start the mining loop.

        Parameters
        ----------
        database_url : str
            URL of a database already containing tables `articles` and `sentences`.
            The URL should indicate database dialect and connection argument, e.g.
            `database_url = "postgresql://scott:tiger@localhost/test"`.
        model_path : str
            The path for loading the spacy model that will perform the
            named entity extraction.
        target_table : str
            The target table name for the mining results.
        task_queue : multiprocessing.Queue
            The queue with tasks for this worker
        can_finish : multiprocessing.Event
            A flag to indicate that the worker can stop waiting for new
            tasks. Unless this flag is set, the worker will continue
            polling the task queue for new tasks.
        """
        miner = cls(
            database_url=database_url,
            model_path=model_path,
            target_table=target_table,
            task_queue=task_queue,
            can_finish=can_finish,
        )

        miner.work_loop()
        miner.clean_up()

    def _log_exception(self, article_id):
        """Log any unhandled exception raised during mining."""
        error_trace = io.StringIO()
        traceback.print_exc(file=error_trace)

        error_message = f"\nArticle ID: {article_id}\n" + error_trace.getvalue()
        self.logger.error(error_message)

    def work_loop(self):
        """Do the mining work loop."""
        self.logger.info("Starting mining")
        while not self.can_finish.is_set():
            # Just get new tasks until the main thread sets `can_finish`
            try:
                article_id = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                # This doesn't always mean that the queue is empty,
                # and is raised when queue.get() times out.
                self.logger.debug("queue.Empty raised")
            else:
                try:
                    self._mine(article_id)
                    self.n_tasks_done += 1
                except Exception:
                    self._log_exception(article_id)

    def _generate_texts_with_metadata(self, article_ids):
        """Return a generator of (text, metadata_dict) for nlp.pipe.

        Parameters
        ----------
        article_ids : int or list of int
            Article(s) to mine.

        Yields
        ------
        text : str
            The text to mine
        metadata : dict
            The metadata for the text.
        """
        if isinstance(article_ids, int):
            article_ids = [article_ids]
        df_articles = retrieve_articles(article_ids, self.engine)

        for _, row in df_articles.iterrows():
            text = row["text"]
            article_id = row["article_id"]
            section_name = row["section_name"]
            paragraph_pos = row["paragraph_pos_in_article"]

            metadata = {
                "article_id": article_id,
                "paragraph_pos_in_article": paragraph_pos,
                "paper_id": f"{article_id}:{section_name}:{paragraph_pos}",
            }

            yield text, metadata

    def _mine(self, article_id):
        """Perform one mining task.

        Parameters
        ----------
        article_id : int
            The article ID for mining.
        """
        self.logger.info(f"Processing article_id = {article_id}")

        self.logger.debug("Getting all texts for the article")
        texts = self._generate_texts_with_metadata(article_id)

        self.logger.debug("Running the pipeline")
        df_results = run_pipeline(
            texts=texts, model_entities=self.model, models_relations={}, debug=True
        )

        df_results["mining_model_version"] = self.model.meta["version"]
        df_results["spacy_version"] = self.model.meta["spacy_version"]

        self.logger.debug("Writing results to the SQL database")
        with self.engine.begin() as con:
            df_results.to_sql(
                self.target_table_name, con=con, if_exists="append", index=False
            )

        self.logger.info(f"Mined {len(df_results)} entities.")

    def clean_up(self):
        """Clean up after task processing has been finished."""
        self.logger.info("Finished mining, cleaning up")
        self.logger.info(f"I'm proud to have done {self.n_tasks_done} tasks!")

    def __str__(self):
        """Represent self as string.

        Returns
        -------
        str
            The string representation of self.
        """
        return f"{self.__class__.__name__}[{self.name}]"


class CreateMiningCache:
    """Create SQL database to save results of mining into a cache.

    Parameters
    ----------
    database_engine : sqlalchemy.engine.Engine
        Connection to the CORD-19 database.
    ee_models_paths : dict[str, pathlib.Path]
        Dictionary mapping entity type to model path detecting it.
    target_table_name : str
        The target table name for the mining results.
    workers_per_model : int, optional
        Number of max processes to spawn to run text mining and table
        population in parallel.
    """

    def __init__(
        self,
        database_engine,
        ee_models_paths,
        target_table_name,
        workers_per_model=1,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        required_tables = ["articles", "sentences"]
        for table_name in required_tables:
            if not database_engine.has_table(table_name):
                raise ValueError(
                    f"Database at {database_engine.url} does not "
                    f"contain required table {table_name}."
                )

        self.engine = database_engine
        self.target_table = target_table_name
        self.ee_models_paths = ee_models_paths
        self.workers_per_model = workers_per_model

    def construct(self):
        """Construct and populate the cache of mined results."""
        self.logger.info("Creating target table schema")
        self._schema_creation()

        self.logger.info("Deleting rows that will be re-populated")
        self._delete_rows()

        self.logger.info("Starting mining")
        self.do_mining()

        self.logger.info("Mining complete")

    def _delete_rows(self):
        """Delete rows in the target table that will be re-populated."""
        for etype in self.ee_models_paths:
            # Reformatted due to this bandit bug in python3.8:
            # https://github.com/PyCQA/bandit/issues/658
            query = (  # nosec
                f"DELETE FROM {self.target_table} WHERE entity_type = :etype"
            )
            self.engine.execute(
                sqlalchemy.sql.text(query),
                etype=etype,
            )

    def _schema_creation(self):
        """Create the schemas of the different tables in the database."""
        metadata = sqlalchemy.MetaData()

        if self.engine.has_table(self.target_table):
            self.mining_cache_table = sqlalchemy.Table(
                self.target_table, metadata, autoload=True, autoload_with=self.engine
            )
            return

        articles_table = sqlalchemy.Table(
            "articles", metadata, autoload=True, autoload_with=self.engine
        )

        self.mining_cache_table = sqlalchemy.Table(
            self.target_table,
            metadata,
            sqlalchemy.Column("entity", sqlalchemy.Text()),
            sqlalchemy.Column("entity_type", sqlalchemy.Text()),
            sqlalchemy.Column("property", sqlalchemy.Text()),
            sqlalchemy.Column("property_value", sqlalchemy.Text()),
            sqlalchemy.Column("property_type", sqlalchemy.Text()),
            sqlalchemy.Column("property_value_type", sqlalchemy.Text()),
            sqlalchemy.Column("ontology_source", sqlalchemy.Text()),
            sqlalchemy.Column("paper_id", sqlalchemy.Text()),
            sqlalchemy.Column("start_char", sqlalchemy.Integer()),
            sqlalchemy.Column("end_char", sqlalchemy.Integer()),
            sqlalchemy.Column(
                "article_id",
                sqlalchemy.Integer(),
                sqlalchemy.ForeignKey(articles_table.c.article_id),
                nullable=False,
            ),
            sqlalchemy.Column(
                "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
            ),
            sqlalchemy.Column(
                "mining_model_version", sqlalchemy.Text(), nullable=False
            ),
            sqlalchemy.Column("spacy_version", sqlalchemy.Text(), nullable=False),
        )

        with self.engine.begin() as connection:
            metadata.create_all(connection)

    def create_tasks(self, task_queues, workers_by_queue):
        """Create tasks for the mining workers.

        Parameters
        ----------
        task_queues : dict[str or pathlib.Path, multiprocessing.Queue]
            Task queues for different models. The keys are the model
            paths and the values are the actual queues.
        workers_by_queue : dict[str]
            All worker processes working on tasks from a given queue.
        """
        self.logger.info("Getting all article IDs")
        result_proxy = self.engine.execute(
            "SELECT article_id FROM articles ORDER BY article_id"
        )
        all_article_ids = [row["article_id"] for row in result_proxy]

        current_task_ids = {queue_name: 0 for queue_name in task_queues}

        # We got some new tasks, put them in the task queues.
        self.logger.info("Adding new tasks")
        # As long as there are any tasks keep trying to add them to the queues
        while any(
            task_idx < len(all_article_ids) for task_idx in current_task_ids.values()
        ):
            for queue_name, task_queue in task_queues.items():
                # Check if still task available for the current queue
                current_task_idx = current_task_ids[queue_name]
                if current_task_idx == len(all_article_ids):
                    self.logger.debug(
                        f"All tasks for the {queue_name} queue have already been added."
                    )
                    continue

                # Check if there are still workers working on this queue
                if not any(
                    worker.is_alive() for worker in workers_by_queue[queue_name]
                ):
                    self.logger.debug("No workers left working on this queue")
                    current_task_ids[queue_name] = len(all_article_ids)
                    continue

                # Try adding the task to the queue.
                article_id = all_article_ids[current_task_idx]
                self.logger.debug(
                    f"Adding article ID {article_id} to the {queue_name} queue"
                )
                try:
                    task_queue.put(article_id, timeout=0.5)
                    current_task_ids[queue_name] += 1
                except queue.Full:
                    self.logger.debug("Queue full, will try next time")

    def do_mining(self):
        """Do the parallelized mining."""
        self.logger.info(
            f"Starting mining with {self.workers_per_model} workers per model."
        )

        # Flags to let the workers know there won't be any new tasks.
        can_finish: Dict[str, mp.synchronize.Event] = {
            etype: mp.Event() for etype in self.ee_models_paths
        }

        # Prepare the task queues for the workers - one task queue per model.
        task_queues: Dict[str, mp.Queue] = {
            etype: mp.Queue() for etype in self.ee_models_paths
        }

        # Spawn the workers according to `workers_per_model`.
        self.logger.info("Spawning the worker processes")
        worker_processes = []
        workers_by_queue: Dict[str, List[mp.Process]] = {
            queue_name: [] for queue_name in task_queues
        }
        for etype, model_path in self.ee_models_paths.items():
            for i in range(self.workers_per_model):
                worker_name = f"{etype}_{i}"
                worker_process = mp.Process(
                    name=worker_name,
                    target=Miner.create_and_mine,
                    kwargs={
                        "database_url": self.engine.url,
                        "model_path": model_path,
                        "target_table": self.target_table,
                        "task_queue": task_queues[etype],
                        "can_finish": can_finish[etype],
                    },
                )
                worker_process.start()
                worker_processes.append(worker_process)
                workers_by_queue[etype].append(worker_process)

        # Create tasks
        self.logger.info("Creating tasks")
        self.create_tasks(task_queues, workers_by_queue)

        # Monitor the queues and the workers to decide when we're finished.
        # For a given model the work is finished when the corresponding queue
        # is empty. But it can be that all workers stop/crash before all
        # tasks are done. Therefore we need to check if anyone is still
        # working on a given queue, and if not empty we will empty it.
        while not all(flag.is_set() for flag in can_finish.values()):
            for queue_name, task_queue in task_queues.items():
                if can_finish[queue_name].is_set():
                    # This queue is already empty we've let the workers know
                    continue
                if not any(
                    worker.is_alive() for worker in workers_by_queue[queue_name]
                ):
                    self.logger.debug(f"Emptying the {queue_name} queue")
                    while not task_queue.empty():
                        article_id = task_queue.get(timeout=1)
                        self.logger.debug(f"Got non-done task {article_id}")
                if task_queue.empty():
                    self.logger.debug(
                        f"Setting the can finish flag for the {queue_name} queue."
                    )
                    can_finish[queue_name].set()

        self.logger.info("Closing all task queues")
        for queue_name, task_queue in task_queues.items():
            self.logger.debug(f"Closing the reading end of the queue {queue_name}")
            # Note that this is only safe when the queue is empty. This is
            # because there's a background thread putting buffered data
            # in the queue. If the queue is not empty it might be that the
            # background thread is still transferring the data from the
            # buffer. Closing the reading end of the internal pipe actually
            # also closes the writing end, and therefore the background
            # thread will throw a BrokenPipeError as it will fail to write
            # to the closed pipe.
            task_queue.close()
            self.logger.debug(f"Joining the buffering thread of queue {queue_name}")
            task_queue.join_thread()

        # Wait for the processes to finish.
        self.logger.info("No more new tasks, just waiting for the workers to finish")
        # We'll transfer finished workers from `worker_processes`
        # to `finished_workers`. We're done when `worker_processes` is empty.
        finished_workers: List[mp.Process] = []
        while len(worker_processes) > 0:
            self.logger.debug(
                f"Status: {len(worker_processes)} workers still alive, "
                f"{len(finished_workers)} finished."
            )
            # Loop through all living workers and try to join
            for process in worker_processes:
                # Don't need to wait forever - others might finish before
                process.join(timeout=1.0)
                # If the current process did finish then put it in the
                # `finished_workers` queue and do some cleaning up.
                if not process.is_alive():
                    self.logger.info(f"Worker {process.name} finished.")
                    finished_workers.append(process)
                    if process.exitcode != 0:
                        self.logger.error(
                            f"Worker {process.name} terminated with exit "
                            f"code {process.exitcode}!"
                        )

            # Remove all workers that are already in the `finished_workers`
            # list from the `worker_processes` list.
            for process in finished_workers:
                if process in worker_processes:
                    worker_processes.remove(process)

        self.logger.info("Finished mining.")

        # Create index on (article_id, paragraph_pos_in_article, start_char)
        # to speed up ORDER BY clause.
        self.logger.info("Start creating index on (par, art, char)...")
        sqlalchemy.Index(
            "index_art_par_char",
            self.mining_cache_table.c.article_id,
            self.mining_cache_table.c.paragraph_pos_in_article,
            self.mining_cache_table.c.start_char,
        ).create(bind=self.engine)
        self.logger.info("Done creating index on (par, art, char).")
