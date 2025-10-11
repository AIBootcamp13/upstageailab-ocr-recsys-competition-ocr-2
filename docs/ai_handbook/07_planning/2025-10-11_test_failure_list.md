pstageailab-ocr-recsys-competition-ocr-2 (08_refactor/ocr_pl) ❯ /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/bin/python -m pytest -m "not slow" --collect-only | wc -l
545
upstageailab-ocr-recsys-competition-ocr-2 (08_refactor/ocr_pl) ❯ /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/bin/python -m pytest -m "not slow" --tb=no -q
........E........................FF................................................................................................................... [ 38%]
...................s...................s.........F............FFFF.FFFF.FFF....F...FF................................................................. [ 77%]
...............................................................................x........                                                               [100%]
====================================================================== warnings summary ======================================================================
.venv/lib/python3.10/site-packages/defusedxml/__init__.py:30
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/defusedxml/__init__.py:30: DeprecationWarning: defusedxml.cElementTree is deprecated, import from defusedxml.ElementTree instead.
    from . import cElementTree

docs/ai_handbook/04_experiments/debugging/2025-10-08_performance_assessment/02_performance_test.py:19
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/04_experiments/debugging/2025-10-08_performance_assessment/02_performance_test.py:19: PytestCollectionWarning: cannot collect test class 'TestResult' because it has a __init__ constructor (from: docs/ai_handbook/04_experiments/debugging/2025-10-08_performance_assessment/02_performance_test.py)
    @dataclass

scripts/performance_benchmarking/performance_test.py:19
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/scripts/performance_benchmarking/performance_test.py:19: PytestCollectionWarning: cannot collect test class 'TestResult' because it has a __init__ constructor (from: scripts/performance_benchmarking/performance_test.py)
    @dataclass

ui/utils/command/models.py:39
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/command/models.py:39: PytestCollectionWarning: cannot collect test class 'TestCommandParams' because it has a __init__ constructor (from: tests/integration/test_command_builder_integration.py)
    @dataclass

ui/utils/command/models.py:39
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/command/models.py:39: PytestCollectionWarning: cannot collect test class 'TestCommandParams' because it has a __init__ constructor (from: tests/smoke/test_command_builder_smoke.py)
    @dataclass

ui/utils/command/models.py:39
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/command/models.py:39: PytestCollectionWarning: cannot collect test class 'TestCommandParams' because it has a __init__ constructor (from: tests/unit/test_command_modules.py)
    @dataclass

debug/bug_reproduction/test_bug_reproduction.py::test_pil_image_fails
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/bug_reproduction/test_bug_reproduction.py::test_pil_image_fails returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

debug/bug_reproduction/test_bug_reproduction.py::test_numpy_array_works
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/bug_reproduction/test_bug_reproduction.py::test_numpy_array_works returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

debug/bug_reproduction/test_bug_reproduction.py::test_albumentations_handles_pil
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/bug_reproduction/test_bug_reproduction.py::test_albumentations_handles_pil returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

debug/verification/test_bug_fix_verification.py::test_fix_handles_pil_image
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/verification/test_bug_fix_verification.py::test_fix_handles_pil_image returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

debug/verification/test_bug_fix_verification.py::test_numpy_uint8_still_works
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/verification/test_bug_fix_verification.py::test_numpy_uint8_still_works returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

debug/verification/test_bug_fix_verification.py::test_numpy_float32_still_works
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/verification/test_bug_fix_verification.py::test_numpy_float32_still_works returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

debug/verification/test_load_maps_disabled.py::test_load_maps_disabled
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but debug/verification/test_load_maps_disabled.py::test_load_maps_disabled returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

scripts/agent_tools/test_validation.py::test_validation
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but scripts/agent_tools/test_validation.py::test_validation returned <class 'bool'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

tests/integration/test_checkpoint_fixes.py::test_checkpoint_catalog
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but tests/integration/test_checkpoint_fixes.py::test_checkpoint_catalog returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

tests/integration/test_checkpoint_fixes.py::test_ui_compatibility
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/_pytest/python.py:161: PytestReturnNotNoneWarning: Test functions should return None, but tests/integration/test_checkpoint_fixes.py::test_ui_compatibility returned <class 'list'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

tests/integration/test_performance_profiler.py::test_profiler_callback_enabled
tests/integration/test_performance_profiler.py::test_profiler_callback_disabled
tests/integration/test_performance_profiler.py::test_profiler_metrics_logged_to_model
tests/integration/test_performance_profiler.py::test_profiler_batch_timing
tests/integration/test_performance_profiler.py::test_profiler_verbose_mode
tests/performance/test_regression.py::TestValidationPerformance::test_validation_time_within_threshold
tests/performance/test_regression.py::TestValidationPerformance::test_batch_time_variance
tests/performance/test_regression.py::TestMemoryUsage::test_cpu_memory_within_limit
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/lightning/pytorch/trainer/setup.py:177: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.

tests/integration/test_performance_profiler.py::test_profiler_callback_enabled
tests/integration/test_performance_profiler.py::test_profiler_callback_disabled
tests/integration/test_performance_profiler.py::test_profiler_metrics_logged_to_model
tests/integration/test_performance_profiler.py::test_profiler_verbose_mode
tests/performance/test_regression.py::TestValidationPerformance::test_validation_time_within_threshold
tests/performance/test_regression.py::TestValidationPerformance::test_batch_time_variance
tests/performance/test_regression.py::TestMemoryUsage::test_gpu_memory_within_limit
tests/performance/test_regression.py::TestMemoryUsage::test_cpu_memory_within_limit
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=47` in the `DataLoader` to improve performance.

tests/integration/test_performance_profiler.py::test_profiler_callback_enabled
tests/integration/test_performance_profiler.py::test_profiler_callback_disabled
tests/integration/test_performance_profiler.py::test_profiler_metrics_logged_to_model
tests/integration/test_performance_profiler.py::test_profiler_verbose_mode
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=47` in the `DataLoader` to improve performance.

tests/ocr/datasets/test_transform_pipeline_contracts.py::TestTransformPipelineContracts::test_end_to_end_with_real_dataset_sample
  /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/lib/python3.10/site-packages/albumentations/augmentations/blur/transforms.py:184: UserWarning: blur_limit and sigma_limit minimum value can not be both equal to 0. blur_limit minimum value changed to 3.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================================== short test summary info ===================================================================
FAILED tests/integration/test_inference_service.py::test_perform_inference_with_doc_tr_preprocessing - TypeError: 'InferenceResult' object is not subscriptable
FAILED tests/integration/test_inference_service.py::test_perform_inference_preprocessing_failure_falls_back - TypeError: 'InferenceResult' object is not subscriptable
FAILED tests/test_ocr_dataset_and_collate.py::test_ocr_dataset_getitem_map_loading - AssertionError: assert 'prob_map' in OrderedDict([('image', tensor([[[0.8823, 0.9150, 0.3829,  ..., 0.4701, 0.6202, 0.6401],\n         [0.0459, 0.3155, 0...
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_instantiate_dataset_and_collate_fn - ValueError: Expected 2 map files, found 1
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_output_directory_creation - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_successful_preprocessing - ValueError: Map validation failed with 2 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_samples_with_no_polygons_skipped - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_limit_samples_config - ValueError: Map validation failed with 3 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_sample_limit_larger_than_dataset - ValueError: Map validation failed with 2 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_numpy_image_conversion - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunction::test_grayscale_numpy_image_conversion - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunctionErrorHandling::test_error_processing_sample_continues - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunctionErrorHandling::test_map_generation_failure_continues - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessFunctionErrorHandling::test_sanity_check_on_generated_files - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessConfigValidation::test_valid_config_structure_passes - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessEdgeCases::test_polygons_with_invalid_ndim - ValueError: Map validation failed with 1 errors:
FAILED tests/test_preprocess_maps.py::TestPreprocessEdgeCases::test_polygons_with_insufficient_vertices - ValueError: Map validation failed with 1 errors:
ERROR logs/doctr_cropping_finetune/scripts/baseline_test.py::test_document_detection
17 failed, 367 passed, 2 skipped, 1 xfailed, 37 warnings, 1 error in 221.61s (0:03:41)
upstageailab-ocr-recsys-competition-ocr-2 (08_refactor/ocr_pl) ❯
