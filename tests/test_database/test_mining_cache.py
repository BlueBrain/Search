import logging
import multiprocessing as mp

import pytest

from bbsearch.database.mining_cache import Miner


class TestMiner:

    @pytest.fixture(scope="session")
    def miner_env(self, fake_sqlalchemy_engine):
        database_engine = fake_sqlalchemy_engine
        # model_path = "en_ner_bionlp13cg_md"
        model_path = "en_core_web_sm"
        entity_map = {
            "ORGAN": "ORGAN_ENTITY"
        }
        target_table = "mining_cache_temp"
        task_queue = mp.Queue()
        can_finish = mp.Event()

        can_finish.set()

        miner = Miner(
            database_engine=database_engine,
            model_path=model_path,
            entity_map=entity_map,
            target_table=target_table,
            task_queue=task_queue,
            can_finish=can_finish,
        )

        return miner, task_queue, can_finish

    def test_create_and_mine(self):
        ...

    def test_log_exception(self):
        ...

    def test_work_loop(self):
        ...

    def test_fetch_all_paragraphs(self):
        ...

    def test_generate_texts_with_metadata(self):
        ...

    def test_mine(self):
        ...

    def test_clean_up(self, miner_env, caplog):
        miner, _, _ = miner_env
        article_id = 5

        miner._log_exception(article_id=article_id)

        assert len(caplog.record_tuples) == 1
        logger_name, level, text = caplog.record_tuples[0]
        assert logger_name.startswith("Miner")
        assert level == logging.ERROR
        assert f"Article ID: {article_id}" in text

    def test_str(self):
        ...

    def test_repr(self):
        ...


class TestCreateMiningCache:
    ...
