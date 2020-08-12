import logging
import pathlib
import queue
import time
from multiprocessing import Process, Queue, Value

import numpy as np
import spacy
import sqlalchemy

from bbsearch.mining import run_pipeline


class Miner:
    def __init__(
        self, name, engine_url, model_path, work_queue, finished, logging_level=None
    ):
        self.name = name
        self.engine = sqlalchemy.create_engine(engine_url)
        self.model_path = model_path
        self.work_queue = work_queue
        self.finished = finished

        self.n_tasks_done = 0

        self.logger = logging.getLogger(str(self))
        if logging_level is not None:
            self.logger.setLevel(logging_level)

        self.logger.info("loading the NLP model...")
        self.ee_model = spacy.load(model_path)

        self.logger.info("starting mining...")
        self.work_loop()

        self.logger.info("finished mining, cleaning up...")
        self.clean_up()

        self.logger.info("all done.")

    def work_loop(self):
        finished = False
        while not finished:
            try:
                article_id = self.work_queue.get(block=False)
                self.mine(article_id)
                self.n_tasks_done += 1
            except queue.Empty:
                self.logger.info(f"Queue empty")
                if self.finished.value == 1:
                    finished = True
                else:
                    time.sleep(1)

    def mine(self, article_id):
        self.logger.info(f"Processing article_id = {article_id}")

        self.logger.info("Getting all sentences from the database...")
        query = f"""
        select paragraph_pos_in_article, sentence_pos_in_paragraph, text
        from sentences
        where article_id = '{article_id}'
        """
        all_sentences = dict()
        result = self.engine.execute(query)
        for (
            paragraph_pos_in_article,
            sentence_pos_in_paragraph,
            text,
        ) in result.fetchall():
            all_sentences[(paragraph_pos_in_article, sentence_pos_in_paragraph)] = text

        texts = [(all_sentences[key], {}) for key in sorted(all_sentences)]

        self.logger.info("Running the pipeline...")
        df_results = run_pipeline(
            texts=texts, model_entities=self.ee_model, models_relations={}, debug=False
        )

        self.logger.info(f"Mined {len(df_results)} entities.")

    def clean_up(self):
        self.logger.info(f"I'm proud to have done {self.n_tasks_done} tasks!")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]"


def get_engine_url():
    mysql_host = "dgx1.bbp.epfl.ch"
    mysql_port = 8853
    database_name = "cord19_v35"
    engine_url = (
        f"mysql+mysqldb://guest:guest@{mysql_host}:{mysql_port}/{database_name}"
    )

    return engine_url


def create_tasks(task_queues):
    print("Getting all article IDs...")
    engine = sqlalchemy.create_engine(get_engine_url())
    result = engine.execute("select article_id from articles")
    all_article_ids = [row[0] for row in result.fetchall()]

    # Pretend we're doing something else while the workers are working.
    print("Waiting a bit...")
    time.sleep(15)

    # We got some new tasks, put them in the task queues.
    print("Adding new tasks...")
    for model_path in task_queues:
        for i in range(100):
            article_id = np.random.choice(all_article_ids)
            task_queues[model_path].put(article_id)

    # Again pretend we're busy.
    print("Wait again...")
    time.sleep(15)


def do_mining(model_paths, workers_per_model):
    # Prepare as many workers as necessary according to `workers_per_model`.
    # Also instantiate one task queue per model.
    print("Preparing the workers...")
    workers_info = []
    task_queues = dict()
    for model_path in model_paths:
        task_queues[model_path] = Queue()
        for i in range(workers_per_model):
            worker_name = f"{model_path.name}_{i}"
            workers_info.append((worker_name, model_path))

    # A shared integer value to let the workers know when it's finished.
    finished = Value("i", 0)

    # Create the worker processes.
    print("Spawning the worker processes...")
    worker_processes = []
    for worker_name, model_path in workers_info:
        worker_process = Process(
            target=Miner,
            args=(
                worker_name,
                get_engine_url(),
                model_path,
                task_queues[model_path],
                finished,
            ),
            kwargs=dict(logging_level=logging.INFO),
        )
        worker_process.start()
        worker_processes.append(worker_process)

    # Create tasks
    print("Creating tasks...")
    create_tasks(task_queues)

    # Wait for the processes to finish.
    print("No more new tasks, just wait for the workers to finish...")
    finished.value = 1
    for proc in worker_processes:
        proc.join()

    print("Finished.")


def main():
    logging.basicConfig()

    model_paths = {
        pathlib.Path("assets/model1_bc5cdr_annotations5_spacy23"),
    }
    workers_per_model = 4
    do_mining(model_paths, workers_per_model)


if __name__ == "__main__":
    main()
