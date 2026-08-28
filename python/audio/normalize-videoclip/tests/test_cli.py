from __future__ import annotations

import json

from normalize_videoclip.cli import main


def test_main_missing_input_dir_returns_exit_code_4(tmp_path, capsys):
    code = main([str(tmp_path / "missing"), str(tmp_path / "out")])
    assert code == 4
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_main_dry_run_json_reports_planned_work(tmp_path, capsys):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "clip.mp4").touch()
    (in_dir / "old.webm").touch()

    code = main([str(in_dir), str(tmp_path / "out"), "--dry-run", "--json"])
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 2
    assert payload["dry_run"] == 2
    assert payload["failed"] == 0
    assert not (tmp_path / "out").exists()


def test_main_no_files_found_message(tmp_path, capsys):
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    code = main([str(in_dir), str(tmp_path / "out")])
    assert code == 0
    assert "No matching video files found" in capsys.readouterr().out


def test_main_bad_config_key_returns_exit_code_2(tmp_path, capsys):
    config_path = tmp_path / "bad.toml"
    config_path.write_text("not_a_real_field = 1\n", encoding="utf-8")
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    code = main([str(in_dir), str(tmp_path / "out"), "--config", str(config_path)])
    assert code == 2
    assert "Configuration error" in capsys.readouterr().err


def test_main_rejects_out_of_range_crf(tmp_path, capsys):
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    code = main([str(in_dir), str(tmp_path / "out"), "--crf", "999"])
    assert code == 2
