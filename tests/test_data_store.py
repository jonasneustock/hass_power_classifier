import time

from app.data_store import DataStore


def test_samples_and_sensor_samples(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))

    store.add_sample(1, 10.0)
    store.add_sample(2, 20.0)
    latest = store.get_latest_sample()
    assert latest["value"] == 20.0

    recent = store.get_recent_samples()
    assert [s["value"] for s in recent] == [10.0, 20.0]

    store.add_sensor_sample(1, "s1", 5.0)
    store.add_sensor_sample(2, "s1", 6.0)
    store.add_sensor_sample(2, "s2", 7.0)
    latest_sensors = store.get_latest_sensor_samples()
    assert latest_sensors["s1"]["value"] == 6.0
    assert latest_sensors["s2"]["value"] == 7.0

    recent_s1 = store.get_recent_sensor_samples("s1")
    assert [s["value"] for s in recent_s1] == [5.0, 6.0]


def test_segments_and_labels(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    now = int(time.time())

    seg_id = store.add_segment(
        {
            "start_ts": now,
            "end_ts": now + 10,
            "mean": 1.0,
            "std": 0.1,
            "max": 1.1,
            "min": 0.9,
            "duration": 10,
            "slope": 0.0,
            "change_score": 0.2,
            "candidate": True,
            "created_ts": now,
        }
    )
    segments = store.list_segments()
    assert segments[0]["id"] == seg_id
    assert segments[0]["candidate"] == 1

    store.update_segment_prediction(seg_id, "toaster", "start")
    store.update_segment_label(seg_id, "toaster", "start")
    labeled = store.get_labeled_segments()
    assert labeled[0]["label_appliance"] == "toaster"
    assert store.get_label_counts_by_appliance()["toaster"] == 1

    store.clear_segment_prediction(seg_id)
    cleared = store.get_segment(seg_id)
    assert cleared["predicted_appliance"] is None


def test_delete_segment(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    now = int(time.time())
    seg_id = store.add_segment(
        {
            "start_ts": now,
            "end_ts": now + 5,
            "mean": 1,
            "std": 0,
            "max": 1,
            "min": 1,
            "duration": 5,
            "slope": 0,
            "change_score": 0,
            "candidate": True,
            "flank": "positive",
            "created_ts": now,
        }
    )
    assert store.get_segment(seg_id) is not None
    store.delete_segment(seg_id)
    assert store.get_segment(seg_id) is None


def test_list_segments_filters(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    now = int(time.time())
    seg1 = store.add_segment(
        {
            "start_ts": now,
            "end_ts": now + 1,
            "mean": 1,
            "std": 0,
            "max": 1,
            "min": 1,
            "duration": 1,
            "slope": 0,
            "change_score": 0.5,
            "candidate": True,
            "flank": "positive",
            "created_ts": now,
        }
    )
    seg2 = store.add_segment(
        {
            "start_ts": now + 2,
            "end_ts": now + 3,
            "mean": 1,
            "std": 0,
            "max": 1,
            "min": 1,
            "duration": 1,
            "slope": 0,
            "change_score": 0.1,
            "candidate": False,
            "flank": "flat",
            "label_appliance": "x",
            "label_phase": "start",
            "created_ts": now,
        }
    )
    all_segments = store.list_segments(limit=10)
    assert {s["id"] for s in all_segments} == {seg1, seg2}
    candidates = store.list_segments(candidate_only=True)
    assert {s["id"] for s in candidates} == {seg1}
    unlabeled = store.list_segments(unlabeled_only=True)
    assert {s["id"] for s in unlabeled} == {seg1}


def test_delete_unlabeled_before(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    now = int(time.time())
    old_unlabeled = store.add_segment(
        {
            "start_ts": now - 100,
            "end_ts": now - 90,
            "mean": 1,
            "std": 0,
            "max": 1,
            "min": 1,
            "duration": 10,
            "slope": 0,
            "change_score": 0.2,
            "candidate": True,
            "flank": "negative",
            "created_ts": now - 100,
        }
    )
    old_labeled = store.add_segment(
        {
            "start_ts": now - 100,
            "end_ts": now - 90,
            "mean": 1,
            "std": 0,
            "max": 1,
            "min": 1,
            "duration": 10,
            "slope": 0,
            "change_score": 0.2,
            "candidate": True,
            "label_appliance": "x",
            "label_phase": "start",
            "flank": "positive",
            "created_ts": now - 100,
        }
    )
    recent = store.add_segment(
        {
            "start_ts": now - 10,
            "end_ts": now,
            "mean": 1,
            "std": 0,
            "max": 1,
            "min": 1,
            "duration": 10,
            "slope": 0,
            "change_score": 0.2,
            "candidate": True,
            "flank": "positive",
            "created_ts": now - 10,
        }
    )
    deleted = store.delete_unlabeled_before(now - 50)
    assert deleted == 1
    remaining_ids = {s["id"] for s in store.list_segments(limit=10)}
    assert old_labeled in remaining_ids
    assert recent in remaining_ids
    assert old_unlabeled not in remaining_ids


def test_appliance_activity_sensors(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    store.add_appliance("x", "", "sensor.power", "binary_sensor.door, binary_sensor.window")
    appliance = store.get_appliance("x")
    assert "binary_sensor.door" in appliance["activity_sensors"]


def test_update_current_power(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    store.add_appliance("x", "", "sensor.power", "")
    store.update_appliance_current_power("x", 123.4)
    appliance = store.get_appliance("x")
    assert appliance["current_power"] == 123.4


def test_import_segments(tmp_path):
    db = tmp_path / "store.sqlite"
    store = DataStore(str(db))
    now = int(time.time())
    inserted = store.import_segments(
        [
            {
                "start_ts": now,
                "end_ts": now + 1,
                "mean": 1,
                "std": 0,
                "max": 1,
                "min": 1,
                "duration": 1,
                "slope": 0,
                "change_score": 0.2,
                "label_appliance": "a",
                "label_phase": "start",
                "flank": "positive",
            }
        ]
    )
    assert inserted == 1
    labeled = store.get_labeled_segments()
    assert labeled[0]["label_appliance"] == "a"
