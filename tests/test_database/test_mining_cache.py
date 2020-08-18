import logging
import multiprocessing as mp
import time

import pandas as pd
import pytest
import sqlalchemy

from bbsearch.database.mining_cache import Miner


class TestMiner:

    @pytest.fixture(scope="session")
    def miner_env(self, fake_sqlalchemy_engine):
        task_queue = mp.Queue()
        can_finish = mp.Event()
        miner = Miner(
            database_engine=fake_sqlalchemy_engine,
            model_path="en_core_web_sm",
            entity_map={"ORGAN": "ORGAN_ENTITY"},
            target_table="mining_cache_temporary",
            task_queue=task_queue,
            can_finish=can_finish,
        )

        return miner, task_queue, can_finish

    def test_create_and_mine(self, fake_sqlalchemy_engine):
        task_queue = mp.Queue()
        can_finish = mp.Event()
        can_finish.set()

        Miner.create_and_mine(
            database_engine=fake_sqlalchemy_engine,
            model_path="en_core_web_sm",
            entity_map={},
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
            "entity_type": "ORGAN"
        }

        def run_pipeline(*args, **kwargs):
            can_finish.set()
            return pd.DataFrame([fake_result])

        monkeypatch.setattr("bbsearch.database.mining_cache.run_pipeline", run_pipeline)

        # Work loop with queue with one article ID
        task_queue.put(article_id)
        miner.work_loop()

        df_result = pd.read_sql("select * from mining_cache_temporary", con=miner.engine)
        assert len(df_result) == 1
        assert df_result.iloc[0]["article_id"] == fake_result["article_id"]
        assert df_result.iloc[0]["paper_id"] == fake_result["paper_id"]
        assert df_result.iloc[0]["entity_type"] == miner.entity_map[fake_result["entity_type"]]
        assert "mining_model" in df_result.columns
        assert "mining_model_version" in df_result.columns

        miner.engine.execute(f"drop table {miner.target_table_name}")

    def test_generate_texts_with_metadata(self, miner_env):
        miner, task_queue, can_finish = miner_env
        article_id = 1

        results = list(miner._generate_texts_with_metadata(article_ids=article_id))

        assert len(results) == 2

        (text_1, meta_1), (text_2, meta_2) = results

        assert text_1 == (
            'I am a sentence 0 in section 0 in article 1. I am a sentence 1 in '
            'section 0 in article 1. I am a sentence 2 in section 0 in article 1.'
        )
        assert meta_1 == {
            'article_id': 1,
            'paragraph_pos_in_article': 0,
            'paper_id': '1:section_0:0',
        }
        assert text_2 == (
            'I am a sentence 0 in section 1 in article 1. I am a sentence 1 in '
            'section 1 in article 1. I am a sentence 2 in section 1 in article 1.'
        )
        assert meta_2 == {
            'article_id': 1,
            'paragraph_pos_in_article': 1,
            'paper_id': '1:section_1:1',
        }

    def test_clean_up(self, miner_env, caplog):
        miner, _, _ = miner_env
        miner.clean_up()

        for logger_name, level, text in caplog.record_tuples:
            assert logger_name == str(miner)
            assert level == logging.INFO

    def test_str(self, miner_env):
        miner, _, _ = miner_env
        assert str(miner) == repr(miner)

    def test_repr(self, miner_env):
        miner, _, _ = miner_env
        assert repr(miner) == f"Miner[{mp.current_process().name}]"


class TestCreateMiningCache:

    @pytest.fixture
    def cache_creator(self, fake_sqlalchemy_engine):
        ee_models_library = pd.DataFrame([
            {
                "entity_type": "type_1",
                "model": "model_1",
                "entity_type_name": "type_1_public"
            }
        ])
        cache_creator_instance = CreateMiningCache(
            database_engine=fake_sqlalchemy_engine,
            ee_models_library=ee_models_library,
            target_table_name="mining_cache_temporary",
            restrict_to_models=["model_1"],
            workers_per_model=2,
        )

        return cache_creator_instance

    def test_delete_rows(self, cache_creator):
        table_name = cache_creator.target_table
        engine = cache_creator.engine
        model_schemas = cache_creator.model_schemas

        # Create a table with two rows, one with good model, one with bad
        assert len(model_schemas) >= 1
        valid_model = list(model_schemas.keys())[0]
        df = pd.DataFrame([
            {"mining_model": valid_model},
            {"mining_model": valid_model + "_not_valid"}
        ])
        df.to_sql(table_name, engine)

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
            f"select * from {cache_creator.target_table}",
            con=cache_creator.engine
        )
        
        assert len(df_new_table) == 0
        assert set(df_new_table.columns) == {
            'entity', 'entity_type', 'property', 'property_value',
            'property_type', 'property_value_type', 'ontology_source',
            'paper_id', 'start_char', 'end_char', 'article_id',
            'paragraph_pos_in_article', 'mining_model', 'mining_model_version'
        }

        # Test calling with the table already existing.
        cache_creator._schema_creation()

        # Clean up by deleting the table just created
        cache_creator.engine.execute(f"drop table {cache_creator.table_name}")

    def test_load_model_schemas(self):
        ...

    def test_create_tasks(self):
        queue = mp.Queue()

        def worker(stop_event):
            while not stop_event.is_set():
                time.sleep(0.01)

    def test_do_mining(self):
        ...

    def test_construct(self):
        ...
