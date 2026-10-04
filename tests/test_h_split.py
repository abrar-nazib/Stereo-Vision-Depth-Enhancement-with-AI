from experiments.H.run import split_rows_2000


def test_split_keeps_scene20_out_of_training_and_groups_variations():
    rows = [{"scene": scene, "frame": frame, "variation": variation}
            for scene, count in (("Scene01", 45), ("Scene02", 45),
                                 ("Scene06", 45), ("Scene18", 45), ("Scene20", 20))
            for frame in range(count) for variation in range(10)]
    train, val, test = split_rows_2000(rows, seed=42)
    assert (len(train), len(val), len(test)) == (1800, 100, 100)
    assert all(row["scene"] != "Scene20" for row in train)
    assert not ({row["frame"] for row in val} & {row["frame"] for row in test})
