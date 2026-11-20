from dogpick.io import tilt_image_name


def test_tilt_image_name_matches_jolt():
    assert tilt_image_name(0, 31) == "tilt_01"
    assert tilt_image_name(30, 31) == "tilt_31"
    assert tilt_image_name(0, 120) == "tilt_001"
