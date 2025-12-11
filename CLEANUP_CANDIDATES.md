# Cleanup Candidates - Files to Move to `old/` or Remove

## ✅ CLEANUP COMPLETED

All files have been moved or removed as planned.

## 🗑️ Files to Remove (No longer needed)

### BiGRU Model Files (Replaced by LightGBM)
- `utils/bigru_predictor.py` - Replaced by `utils/lgbm_predictor.py`
- `scripts/bigru_score_predictor.py` - No longer used
- `scripts/train_bigru_model.py` - Old training script
- `scripts/test_bigru_prediction.py` - Test for old model
- `scripts/old/create_dummy_model.py` - Already in old/, can be removed

### Obsolete Documentation
- `docs/BIGRU_INTEGRATION.md` - Documents old BiGRU integration
- `docs/BIGRU_README.md` - Old BiGRU documentation
- `utils/ACCELEROMETER_TRANSFORM.md` - Outdated (mentions BiGRU)

## 📦 Files to Move to `scripts/old/`

### Debug Scripts (Temporary debugging, not needed in production)
- `scripts/debug_app_speed.py`
- `scripts/debug_banked_turn.py`
- `scripts/debug_coordinate_system.py`
- `scripts/debug_curvature.py`
- `scripts/debug_lateral_axis.py`
- `scripts/debug_longitudinal_spikes.py`
- `scripts/debug_physics_models.py`
- `scripts/debug_speed_accumulation.py`
- `scripts/debug_speed_calculation.py`
- `scripts/debug_speed_jump.py`
- `scripts/debug_speed_sharp_edges.py`
- `scripts/debug_speed_smoothing.py`
- `scripts/debug_speed.py`
- `scripts/debug_summary.md`
- `scripts/debug_time_calculation.py`
- `scripts/debug_trackpoints_accel.py`

### Test Scripts (One-off tests, not part of test suite)
- `scripts/test_3d_blocks.py`
- `scripts/test_3d_velocity.py`
- `scripts/test_accelerometer_transform.py`
- `scripts/test_cc_scrape.py`
- `scripts/test_flat_lateral.py`
- `scripts/test_full_integration.py`
- `scripts/test_full_pipeline.py`
- `scripts/test_lateral_gs.py`
- `scripts/test_launch_acceleration.py`
- `scripts/test_launch.py`
- `scripts/test_model_input_ranges.py`
- `scripts/test_multi_loop.py`
- `scripts/test_physics_comparison.py`
- `scripts/test_smoothness.py`
- `scripts/test_speed.py`
- `scripts/test_speeds.py`
- `scripts/test_time_calculation.py`
- `scripts/test_unclipped_gs.py`

### Analysis/Comparison Scripts (One-off analysis)
- `scripts/analyze_vertical_gs.py`
- `scripts/check_curvature.py`
- `scripts/compare_accel_methods.py`
- `scripts/validate_physics_coherence.py`

### Data Preparation Scripts (One-time use, already run)
- `scripts/data_preparation.py` - Old data prep, replaced by notebook
- `scripts/create_coaster_url_mapping.py` - One-time mapping creation
- `scripts/create_name_mapping.py` - One-time mapping creation
- `scripts/create_rfdb_mapping.py` - One-time mapping creation (if already done)
- `scripts/merge_cc_into_mapping.py` - One-time merge operation
- `scripts/show_rfdb_mapping_stats.py` - Analysis script
- `scripts/run_cc_batch.py` - Batch processing script (if no longer needed)

## 📚 Files to Move to `notebooks/old/`

### Already in old/ (keep there)
- `notebooks/old/BiGRU Model for Coaster Score Prediction.ipynb`
- `notebooks/old/BigRU_code_new.ipynb`
- `notebooks/old/Copy of Dec3_BigRU_code_new.ipynb`

### Potentially move (if not actively used)
- `notebooks/cc_cluster_analysis.ipynb` - Analysis notebook, may not be needed
- `notebooks/model_feature_importance.ipynb` - Analysis notebook, may not be needed

## 📄 Documentation to Archive

### Already in archive/ (keep there)
- `ratings_data/archive/` - All files can stay

### Consider moving to `docs/archive/`
- `docs/COMPLETE.md` - Old completion notes
- `docs/INTEGRATION_SUMMARY.md` - Old integration summary
- `docs/APP_DRAWING_GUIDE.md` - If not actively used

## ✅ Files to KEEP (Active/Important)

### Active Scripts
- `scripts/process_rfdb_to_leaderboard.py` - **KEEP** (actively used)
- `scripts/rerun_rfdb_scores.py` - **KEEP** (actively used)
- `scripts/add_steel_vengeance.py` - **KEEP** (utility script)
- `scripts/pov_geometry_infer.py` - **KEEP** (video processing)
- `scripts/build_track_library.py` - **KEEP** (if used for library management)

### Active Utils
- `utils/lgbm_predictor.py` - **KEEP** (current model)
- `utils/accelerometer_transform.py` - **KEEP** (active)
- `utils/submission_manager.py` - **KEEP** (active)
- `utils/acceleration.py` - **KEEP** (active)
- `utils/track_blocks.py` - **KEEP** (active)
- `utils/track_library.py` - **KEEP** (active)
- `utils/cloud_data_loader.py` - **KEEP** (active)

### Active Notebooks
- `notebooks/Dec6_Feature_Only_Cleaned (1).ipynb` - **KEEP** (training notebook)

## 🐛 Files with Issues to Fix First

### Broken Imports (Need to fix before cleanup)
- `pages/03_Leaderboard.py` - Line 6: Imports `load_submissions_from_s3` and `load_submission_geometry_from_s3` which don't exist
  - Should use: `load_submissions` and `load_submission_geometry` instead

## 📊 Summary

- **Remove**: ~5 files (BiGRU-related, obsolete docs)
- **Move to old/**: ~40 files (debug, test, analysis scripts)
- **Archive docs**: ~3 files (old documentation)
- **Fix first**: 1 file (broken imports in leaderboard)

Total cleanup: ~48 files

