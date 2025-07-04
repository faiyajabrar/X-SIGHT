@echo off
echo =============================================
echo  ADVANCED HYPERPARAMETER TUNING - START
echo =============================================
echo.
echo 🎯 Starting comprehensive hyperparameter optimization
echo.
echo Features to optimize:
echo   📊 Learning rates, dropout, weight decay
echo   ⚡ Gradient clipping and scheduler parameters
echo   🎨 Progressive training strategies
echo   📱 Advanced features (MixUp, EMA, etc.)
echo   🔧 Loss function weights and parameters
echo.
echo ⏱️  Each trial: 10 epochs (vs 60 in full training)
echo 🎯 Target: 100 trials total
echo ⏰ Expected time: 2-4 hours (depending on GPU)
echo 📊 Results continuously saved to tune_runs/
echo ⏹️  Use Ctrl+C to stop early if satisfied
echo.
echo 📚 Using virtual environment Python...
echo.

venv\Scripts\python.exe training/tune_advanced.py ^
    --n_trials 100 ^
    --study_name attention_unet_advanced ^
    --n_jobs 1

echo.
echo ✅ TUNING COMPLETE!
echo.
echo 📊 Check results in tune_runs/:
echo   - attention_unet_advanced_best_params.json (optimal parameters)
echo   - attention_unet_advanced_all_results.json (all trial results)
echo   - attention_unet_advanced.db (study database)
echo.
echo 🎯 Use the best parameters in your training script!
echo 💡 To continue optimization, use tune_resume.bat
echo.
pause 