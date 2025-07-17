@echo off
echo =============================================
echo  ADVANCED HYPERPARAMETER TUNING - RESUME
echo =============================================
echo.
echo 🔄 Resuming hyperparameter optimization from existing study
echo.
echo 📋 Resume Features:
echo   ✅ Continues from existing successful trials
echo   🔄 Automatically reruns failed/pruned trials  
echo   📊 Only counts successful trials toward 100 target
echo   💾 Preserves all trial history and best parameters
echo.
echo ⏹️  Use Ctrl+C to stop when satisfied with results
echo.
echo ⏱️  Each trial: 24 epochs (vs 120 in full training)
echo 📊 Progressive Resizing: start_size → end_size (full size at epoch 19)
echo 🛑 Early Stopping: 8 epochs patience (33% of trial duration)
echo 📊 Results continuously saved to tune_runs/
echo 🎯 Target: 100 SUCCESSFUL trials total
echo.
echo 🧹 Note: Failed trial checkpoints will be cleaned for rerun
echo.
echo 📚 Using virtual environment Python...
echo.

venv\Scripts\python.exe training/tune_advanced.py ^
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
echo 📝 Note: Failed trials have been rerun automatically
echo     Only successful trials count toward the 100 target
echo.
pause 