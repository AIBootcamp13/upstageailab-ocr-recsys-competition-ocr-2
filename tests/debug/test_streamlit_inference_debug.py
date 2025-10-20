"""Debug script to test Streamlit inference components."""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

# Test imports
print("=" * 80)
print("TESTING STREAMLIT INFERENCE COMPONENTS")
print("=" * 80)

print("\n1. Testing imports...")
try:
    from ui.apps.inference.models.batch_request import BatchPredictionRequest
    from ui.apps.inference.models.ui_events import InferenceRequest
    from ui.apps.inference.services.inference_runner import InferenceService

    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("\n2. Testing inference engine...")
try:
    from ui.utils.inference.engine import run_inference_on_image

    print(f"✅ Inference engine available: {run_inference_on_image is not None}")
except Exception as e:
    print(f"❌ Inference engine error: {e}")
    sys.exit(1)

print("\n3. Testing checkpoint discovery...")
try:
    checkpoint = Path("./outputs/transforms_test-dbnetpp-dbnetpp_decoder-resnet18/checkpoints/epoch-18_step-003895.ckpt")
    test_image = Path("data/datasets/LOW_PERFORMANCE_IMGS_canonical/drp.en_ko.in_house.selectstar_003949.jpg")

    if not checkpoint.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint}")
        # Find any available checkpoint
        checkpoints = list(Path(".").glob("outputs/**/checkpoints/*.ckpt"))
        if checkpoints:
            checkpoint = checkpoints[0]
            print(f"   Using alternative: {checkpoint}")
        else:
            print("❌ No checkpoints found")
            sys.exit(1)

    if not test_image.exists():
        print(f"⚠️  Test image not found: {test_image}")
        # Find any test image
        images = list(Path(".").glob("data/**/*.jpg"))
        if images:
            test_image = images[0]
            print(f"   Using alternative: {test_image}")
        else:
            print("❌ No test images found")
            sys.exit(1)

    print(f"✅ Using checkpoint: {checkpoint}")
    print(f"✅ Using test image: {test_image}")
except Exception as e:
    print(f"❌ File discovery error: {e}")
    sys.exit(1)

print("\n4. Testing direct inference...")
try:
    result = run_inference_on_image(
        str(test_image),
        str(checkpoint),
        None,
        0.3,  # binarization_thresh
        0.4,  # box_thresh
        300,  # max_candidates
        5,  # min_detection_size
    )
    if result:
        polygons = result.get("polygons", "")
        num_polygons = len(polygons.split("|")) if polygons else 0
        print(f"✅ Direct inference successful: {num_polygons} polygons detected")
    else:
        print("❌ Direct inference returned None")
except Exception as e:
    print(f"❌ Direct inference error: {e}")
    import traceback

    traceback.print_exc()

print("\n5. Testing InferenceService with mock uploaded file...")
try:
    # Create a mock UploadedFile-like object
    class MockUploadedFile:
        def __init__(self, path: Path):
            self.name = path.name
            self.path = path

        def getvalue(self):
            return self.path.read_bytes()

    mock_file = MockUploadedFile(test_image)

    # Create InferenceRequest
    request = InferenceRequest(
        files=[mock_file],
        model_path=str(checkpoint),
        config_path=None,
        use_preprocessing=False,
        preprocessing_config=None,
    )

    print("✅ InferenceRequest created:")
    print(f"   - Model: {request.model_path}")
    print(f"   - Files: {len(request.files)}")
    print(f"   - Preprocessing: {request.use_preprocessing}")

except Exception as e:
    print(f"❌ InferenceRequest creation error: {e}")
    import traceback

    traceback.print_exc()

print("\n6. Testing InferenceService.run()...")
try:
    # We can't fully test this without Streamlit session state,
    # but we can test the _perform_inference method directly
    service = InferenceService()

    hyperparams = {
        "binarization_thresh": 0.3,
        "box_thresh": 0.4,
        "max_candidates": 300.0,
        "min_detection_size": 5.0,
    }

    result = service._perform_inference(test_image, checkpoint, None, test_image.name, hyperparams, False, None)

    print("✅ InferenceService._perform_inference result:")
    print(f"   - Success: {result.success}")
    print(f"   - Filename: {result.filename}")
    if result.success and result.predictions:
        num_polygons = len(result.predictions.polygons.split("|")) if result.predictions.polygons else 0
        print(f"   - Polygons: {num_polygons}")
    elif not result.success:
        print(f"   - Error: {result.error}")

except Exception as e:
    print(f"❌ InferenceService error: {e}")
    import traceback

    traceback.print_exc()

print("\n7. Testing batch prediction...")
try:
    # Create a temp directory with test images
    temp_dir = Path(tempfile.mkdtemp())
    print(f"   Created temp dir: {temp_dir}")

    # Copy test image to temp dir
    shutil.copy(test_image, temp_dir / test_image.name)

    from ui.apps.inference.models.batch_request import (
        BatchHyperparameters,
        BatchOutputConfig,
    )

    batch_request = BatchPredictionRequest(
        input_dir=str(temp_dir),
        model_path=str(checkpoint),
        config_path=None,
        use_preprocessing=False,
        output_config=BatchOutputConfig(
            output_dir=str(temp_dir),
            filename_prefix="test_batch",
            save_json=True,
            save_csv=False,
            include_confidence=False,
        ),
        hyperparameters=BatchHyperparameters(
            binarization_thresh=0.3,
            box_thresh=0.4,
            max_candidates=300,
            min_detection_size=5,
        ),
    )

    print("✅ BatchPredictionRequest created:")
    print(f"   - Input dir: {batch_request.input_dir}")
    print(f"   - Model: {batch_request.model_path}")

    # Get image files
    image_files = batch_request.get_image_files()
    print(f"   - Found {len(image_files)} images")

    # Cleanup
    shutil.rmtree(temp_dir)

except Exception as e:
    print(f"❌ Batch prediction error: {e}")
    import traceback

    traceback.print_exc()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
