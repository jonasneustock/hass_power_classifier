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
