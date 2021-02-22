import argparse
from typing import Dict, Sequence

import pytest

from bluesearch.entrypoint._helper import parse_args_or_environment


def test_parse_args_or_environment(monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-arg")
    parser.add_argument("--env-arg", default=argparse.SUPPRESS)
    argv_value = "5"
    env_value = "6"

    # --env-arg not provided at all
    argv: Sequence[str] = []
    env_variable_names: Dict[str, str] = {}
    args = parse_args_or_environment(parser, env_variable_names, argv)
    assert "normal_arg" in args.__dict__
    assert "env_arg" not in args.__dict__

    # --env-arg provided through the CLI
    argv = ["--env-arg", argv_value]
    env_variable_names = {}
    args = parse_args_or_environment(parser, env_variable_names, argv)
    assert "normal_arg" in args.__dict__
    assert "env_arg" in args.__dict__
    assert args.env_arg == argv_value

    # --env-arg provided through the environment
    argv = []
    environ = {
        "ENV_ARG": env_value,
    }
    monkeypatch.setattr("bluesearch.entrypoint._helper.os.environ", environ)
    env_variable_names = {
        "env_arg": "ENV_ARG",
    }
    args = parse_args_or_environment(parser, env_variable_names, argv)
    assert "normal_arg" in args.__dict__
    assert "env_arg" in args.__dict__
    assert args.env_arg == env_value

    # Check that CLI argument have precedence over environment variables
    argv = ["--env-arg", argv_value]
    environ = {
        "ENV_ARG": env_value,
    }
    monkeypatch.setattr("bluesearch.entrypoint._helper.os.environ", environ)
    env_variable_names = {
        "env_arg": "ENV_ARG",
    }
    args = parse_args_or_environment(parser, env_variable_names, argv)
    assert "normal_arg" in args.__dict__
    assert "env_arg" in args.__dict__
    assert args.env_arg == argv_value

    # Value not specified through the CLI, nor through environment
    argv = []
    environ = {}
    monkeypatch.setattr("bluesearch.entrypoint._helper.os.environ", environ)
    env_variable_names = {
        "env_arg": "ENV_ARG",
    }
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        parse_args_or_environment(parser, env_variable_names, argv)
    assert pytest_wrapped_e.value.code == 1
