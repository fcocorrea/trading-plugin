from trading_engine.cli import main


def test_inverted_fetch_range_is_clean_error(binance_asset, capsys):
    assert main(["fetch", binance_asset, "--start", "2025-01-02", "--end", "2025-01-01"]) == 1
    assert "before start" in capsys.readouterr().err


def test_check_data_without_data(binance_asset, capsys):
    assert main(["check-data", binance_asset, "--stream", "trades"]) == 0
    assert f"no data for {binance_asset}" in capsys.readouterr().out


def test_assets_command_without_assets(capsys):
    assert main(["assets"]) == 0
    assert "no assets declared" in capsys.readouterr().out


def test_record_requires_symbols(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["record"])


def test_pending_commands_exit_with_code_2(capsys):
    assert main(["compare", "some/path"]) == 2
    assert "pending" in capsys.readouterr().err


def test_runner_imports():
    import trading_engine.runner  # noqa: F401
