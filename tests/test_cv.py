from src.cv import assert_group_isolation, fold_assignments, observation_id


def test_observation_id_uses_physical_gong_name():
    assert observation_id("010401-20160920230134Lh.jpg") == "20160920230134Lh"
    assert observation_id("20160920230134Lh.jpeg") == "20160920230134Lh"


def test_duplicate_annotator_records_never_split():
    records = []
    for i in range(10):
        name = f"20260{(i%9)+1:01d}{(i%27)+1:02d}120000Bh.jpeg"
        records.extend([{"file_name": name}, {"file_name": name}])
    folds = fold_assignments(records, n_folds=5, group_mode="observation")
    assert assert_group_isolation(records, folds, "observation")
    for i in range(0, len(records), 2):
        assert folds[i] == folds[i + 1]


def test_month_grouping_is_supported():
    records = []
    for month in range(1, 11):
        records.append({"file_name": f"2025{month:02d}01120000Bh.jpeg"})
        records.append({"file_name": f"2025{month:02d}15120000Ch.jpeg"})
    folds = fold_assignments(records, n_folds=5, group_mode="month")
    assert assert_group_isolation(records, folds, "month")
    for i in range(0, len(records), 2):
        assert folds[i] == folds[i + 1]


def test_day_grouping_keeps_same_day_sites_together():
    records = []
    for day in range(1, 21):
        # the same Sun observed by two GONG sites a few minutes apart, one with 2 annotators
        records.append({"file_name": f"201503{day:02d}172714Mh.jpg"})
        records.append({"file_name": f"201503{day:02d}172714Mh.jpg"})
        records.append({"file_name": f"201503{day:02d}175020Bh.jpg"})
    folds = fold_assignments(records, n_folds=5, group_mode="day")
    assert assert_group_isolation(records, folds, "day")
    for i in range(0, len(records), 3):
        assert folds[i] == folds[i + 1] == folds[i + 2]
    assert len(set(folds.tolist())) == 5


def _day_records():
    records = []
    for i in range(40):
        name = f"2016{(i % 12) + 1:02d}{(i % 27) + 1:02d}{i % 24:02d}0134{'BLMCTU'[i % 6]}h.jpg"
        records += [{"file_name": name}] * (1 + i % 3)  # 1-3 annotator records
    return records


def test_fold_file_round_trip_and_priority(tmp_path):
    from src.cv import read_fold_file, write_fold_file

    records = _day_records()
    computed = fold_assignments(records, 5, "day")
    frozen = (computed + 1) % 5  # a different, still group-consistent assignment
    path = tmp_path / "folds.csv"
    write_fold_file(records, frozen, path, source="test")
    assert path.read_text().startswith("# test\nstem,fold\n")
    assert (read_fold_file(records, path, 5) == frozen).all()
    # the frozen file wins over GroupKFold
    assert (fold_assignments(records, 5, "day", fold_file=path) == frozen).all()


def test_fold_file_rejects_incomplete_or_wrong_files(tmp_path):
    import pytest

    from src.cv import read_fold_file, write_fold_file

    records = _day_records()
    folds = fold_assignments(records, 5, "day")
    path = tmp_path / "folds.csv"
    write_fold_file(records[:-1], folds[:-1], path)
    if len({r["file_name"] for r in records[:-1]}) < len({r["file_name"] for r in records}):
        with pytest.raises(ValueError, match="without a fold"):
            read_fold_file(records, path, 5)
    write_fold_file(records, folds, path)
    with pytest.raises(ValueError, match="n_folds"):
        read_fold_file(records, path, 4)


def test_folds_from_oof_cache_layout(tmp_path):
    from pathlib import Path

    from src.cv import folds_from_oof_cache

    records = _day_records()
    folds = fold_assignments(records, 5, "day")
    for r, f in zip(records, folds):
        d = tmp_path / "oof" / f"fold{f}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{Path(r['file_name']).stem}.npz").write_bytes(b"x")
    (tmp_path / "oof" / "manifest.json").write_text("{}")
    assert (folds_from_oof_cache(records, tmp_path / "oof") == folds).all()
    # incomplete cache -> no assignment
    next((tmp_path / "oof").glob("fold0/*.npz")).unlink()
    assert folds_from_oof_cache(records, tmp_path / "oof") is None
