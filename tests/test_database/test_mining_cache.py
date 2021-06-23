"""Tests covering mining cache."""

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

import logging
import multiprocessing as mp
import pathlib
import queue
import time

import pandas as pd
import pytest
import sqlalchemy

from bluesearch.database import CreateMiningCache
from bluesearch.database.mining_cache import Miner


class TestMiner:
    @pytest.fixture
    def miner_env(self, fake_sqlalchemy_engine, monkeypatch, model_entities):
        # Re-use the "en_core_web_sm" model in the model_entities fixture
        monkeypatch.setattr(
            "bluesearch.database.mining_cache.load_spacy_model",
            lambda model_path: model_entities,
        )

        task_queue: mp.Queue = mp.Queue()
        can_finish = mp.Event()
        miner = Miner(
            database_url=fake_sqlalchemy_engine.url,
            model_path="en_core_web_sm",
            target_table="mining_cache_temporary",
            task_queue=task_queue,
            can_finish=can_finish,
        )

        return miner, task_queue, can_finish

    def test_create_and_mine(self, fake_sqlalchemy_engine, monkeypatch, model_entities):
        task_queue: mp.Queue = mp.Queue()
        can_finish = mp.Event()
        can_finish.set()

        # Re-use the "en_core_web_sm" model in the model_entities fixture
        monkeypatch.setattr(
            "bluesearch.database.mining_cache.load_spacy_model",
            lambda model_path: model_entities,
        )

        Miner.create_and_mine(
            database_url=fake_sqlalchemy_engine.url,
            model_path="en_core_web_sm",
            target_table="",
            task_queue=task_queue,
            can_finish=can_finish,
        )

    def test_log_exception(self, miner_env, caplog):
        miner, _, _ = miner_env
        article_id = 5

        try:
            _ = 1 / 0
        except ZeroDivisionError:
            miner._log_exception(article_id=article_id)

        assert len(caplog.record_tuples) == 1
        logger_name, level, text = caplog.record_tuples[0]
        assert logger_name == str(miner)
        assert level == logging.ERROR
        assert f"Article ID: {article_id}" in text
        assert "Traceback" in text
        assert "ZeroDivisionError" in text

    def test_work_loop_and_mine(self, miner_env, monkeypatch):
        miner, task_queue, can_finish = miner_env
        article_id = 0

        fake_result = {
            "article_id": article_id,
            "paper_id": "my_paper",
            "entity_type": "ORGAN",
        }

        def run_pipeline(*args, **kwargs):
            can_finish.set()
            return pd.DataFrame([fake_result])

        monkeypatch.setattr(
            "bluesearch.database.mining_cache.run_pipeline", run_pipeline
        )

        # Work loop with queue with one article ID
        task_queue.put(article_id)
        miner.work_loop()

        df_result = pd.read_sql(
            f"select * from {miner.target_table_name}", con=miner.engine
        )
        assert len(df_result) == 1
        assert df_result.iloc[0]["article_id"] == fake_result["article_id"]
        assert df_result.iloc[0]["paper_id"] == fake_result["paper_id"]
        assert df_result.iloc[0]["entity_type"] == fake_result["entity_type"]
        assert "mining_model_version" in df_result.columns

        miner.engine.execute(f"drop table {miner.target_table_name}")

    def test_generate_texts_with_metadata(self, miner_env, test_parameters):
        miner, task_queue, can_finish = miner_env
        article_id = 1

        results = list(miner._generate_texts_with_metadata(article_ids=article_id))

        assert len(results) == test_parameters["n_sections_per_article"]

        (text_1, meta_1), (text_2, meta_2) = results

        assert text_1 == (
            "I am a sentence 0 in section 0 in article 1. I am a sentence 1 in "
            "section 0 in article 1. I am a sentence 2 in section 0 in article 1."
        )
        assert meta_1 == {
            "article_id": 1,
            "paragraph_pos_in_article": 0,
            "paper_id": "1:section_0:0",
        }
        assert text_2 == (
            "I am a sentence 0 in section 1 in article 1. I am a sentence 1 in "
            "section 1 in article 1. I am a sentence 2 in section 1 in article 1."
        )
        assert meta_2 == {
            "article_id": 1,
            "paragraph_pos_in_article": 1,
            "paper_id": "1:section_1:1",
        }

    def test_clean_up(self, miner_env, caplog):
        miner, _, _ = miner_env
        miner.clean_up()

        for logger_name, level, _text in caplog.record_tuples:
            assert logger_name == str(miner)
            assert level == logging.INFO

    def test_str(self, miner_env):
        miner, _, _ = miner_env
        assert str(miner) == f"Miner[{mp.current_process().name}]"


class TestCreateMiningCache:
    @pytest.fixture
    def cache_creator(self, fake_sqlalchemy_engine):
        ee_models_paths = {"type_1_public": pathlib.Path("/my/model/path")}
        cache_creator_instance = CreateMiningCache(
            database_engine=fake_sqlalchemy_engine,
            ee_models_paths=ee_models_paths,
            target_table_name="mining_cache_temporary",
            workers_per_model=2,
        )

        return cache_creator_instance

    def test_delete_rows(self, cache_creator):
        table_name = cache_creator.target_table
        engine = cache_creator.engine
        ee_models_paths = cache_creator.ee_models_paths

        # Create a table with two rows, one with good model, one with bad
        assert len(ee_models_paths) >= 1
        valid_etype = list(ee_models_paths.keys())[0]
        df = pd.DataFrame(
            [
                {"entity_type": valid_etype},
                {"entity_type": valid_etype + "_not_valid"},
            ]
        )
        with engine.begin() as con:
            df.to_sql(table_name, con)

        # Check table has two rows
        df_table = pd.read_sql(f"select * from {table_name}", engine)
        assert len(df_table) == 2

        # Call `_delete_rows`, it show only delete the row with the good model
        cache_creator._delete_rows()

        # Check that only one row is left
        df_table = pd.read_sql(f"select * from {table_name}", engine)
        assert len(df_table) == 1

        # Clean up by deleting the table just created
        engine.execute(f"drop table {table_name}")

    def test_schema_creation(self, cache_creator):
        cache_creator._schema_creation()
        df_new_table = pd.read_sql(
            f"select * from {cache_creator.target_table}", con=cache_creator.engine
        )

        assert len(df_new_table) == 0
        assert set(df_new_table.columns) == {
            "entity",
            "entity_type",
            "property",
            "property_value",
            "property_type",
            "property_value_type",
            "ontology_source",
            "paper_id",
            "start_char",
            "end_char",
            "article_id",
            "paragraph_pos_in_article",
            "mining_model_version",
            "spacy_version",
        }

        # Test calling with the table already existing.
        cache_creator._schema_creation()

        # Clean up by deleting the table just created
        cache_creator.engine.execute(f"drop table {cache_creator.target_table}")

    @staticmethod
    def fake_wait_miner(stop_now):
        while not stop_now.is_set():
            time.sleep(0.01)

    def test_create_tasks(self, cache_creator):
        my_queue: mp.Queue = mp.Queue()
        queue_name = "my_queue"
        task_queues = {queue_name: my_queue}

        stop_event = mp.Event()
        worker_proc = mp.Process(target=self.fake_wait_miner, args=(stop_event,))
        worker_proc.start()
        workers_by_queue = {queue_name: [worker_proc]}

        assert worker_proc.is_alive()
        cache_creator.create_tasks(task_queues, workers_by_queue)

        # Ugly workaround. This gives the background thread a bit of time
        # to flush the buffer into the pipe. Otherwise the following assert
        # might fail if none of the buffer has been transferred yet.
        while len(my_queue._buffer) > 0:  # type: ignore
            time.sleep(0.1)

        assert not my_queue.empty()

        stop_event.set()
        worker_proc.join()

        # Test adding tasks to a queue where all workers are dead
        cache_creator.create_tasks(task_queues, workers_by_queue)

    @staticmethod
    def fake_queue_miner(task_queue=None, can_finish=None, **kwargs):
        while not can_finish.is_set():
            try:
                _ = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

    def test_do_mining(self, cache_creator, monkeypatch):
        cache_creator._schema_creation()

        monkeypatch.setattr(
            "bluesearch.database.mining_cache.Miner.create_and_mine",
            self.fake_queue_miner,
        )
        cache_creator.do_mining()
        cache_creator.engine.execute(f"drop table {cache_creator.target_table}")
        inspector = sqlalchemy.inspect(cache_creator.engine)
        indexes = inspector.get_indexes("mining_cache")
        assert len(indexes) == 1

    def test_construct(self, cache_creator, monkeypatch):
        monkeypatch.setattr(
            "bluesearch.database.mining_cache.Miner.create_and_mine",
            self.fake_queue_miner,
        )

        cache_creator.construct()
        cache_creator.engine.execute(f"drop table {cache_creator.target_table}")
