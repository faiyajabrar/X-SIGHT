@echo off
echo =============================================
echo  ADVANCED HYPERPARAMETER TUNING - RESUME
echo =============================================
echo.
echo 🔄 Resuming hyperparameter optimization from existing study
echo.
echo This will continue the existing study towards 100 total trials
echo 📊 Will automatically resume from where it left off
echo ⏹️  Use Ctrl+C to stop when satisfied with results
echo.
echo ⏱️  Each trial: 10 epochs (vs 60 in full training)
echo 📊 Results continuously saved to tune_runs/
echo 🎯 Target: 100 total trials (original + additional)
echo.

python training/tune_advanced.py ^
    --n_trials 100 ^
    --study_name attention_unet_advanced ^
    --n_jobs 1 ^
    --resume

echo.
echo ✅ RESUMED TUNING COMPLETE!
echo.
echo 📊 Study progress updated - check tune_runs/ for:
echo   - Latest best parameters (JSON)
echo   - All trial results with full history
echo   - Study database with complete optimization path
echo.
echo 🎯 Use the latest best parameters for optimal performance!
echo 💡 If more trials needed, run this script again!
echo.
pause 