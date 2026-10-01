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
