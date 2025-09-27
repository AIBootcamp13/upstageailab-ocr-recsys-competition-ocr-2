import numpy as np

from ocr.lightning_modules.callbacks.wandb_image_logging import WandbImageLoggingCallback


def test_normalise_polygons_converts_sequences_to_arrays():
    callback = WandbImageLoggingCallback()

    raw_polygons = [
        [
            [10.0, 20.0],
            [30.0, 20.0],
            [30.0, 40.0],
            [10.0, 40.0],
        ],
        np.array(
            [
                [50.0, 60.0],
                [70.0, 60.0],
                [70.0, 80.0],
                [50.0, 80.0],
            ],
            dtype=np.float32,
        ),
    ]

    normalised = callback._normalise_polygons(raw_polygons)

    assert len(normalised) == 2
    for polygon in normalised:
        assert isinstance(polygon, np.ndarray)
        assert polygon.shape == (1, 4, 2)


def test_postprocess_polygons_filters_degenerate_regions():
    callback = WandbImageLoggingCallback()

    raw_polygons = [
        [[[80.0, 82.5], [80.0, 96.0], [80.0, 96.0], [80.0, 82.5]]],
        [[[95.0, 100.0], [120.0, 100.0], [120.0, 130.0], [95.0, 130.0]]],
    ]

    normalised = callback._normalise_polygons(raw_polygons)
    processed = callback._postprocess_polygons(normalised, image_size=(200, 200))

    assert len(processed) == 1
    assert np.allclose(processed[0].reshape(-1, 2)[0], [95.0, 100.0])
