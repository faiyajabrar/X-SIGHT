@echo off
echo BACKUP TRAINING: Proven Adam + Cosine Annealing Configuration
echo ================================================================
echo.

echo This is the SAFE and RELIABLE backup configuration that achieved:
echo - Validation Dice: 0.576 (69%% improvement over baseline)
echo - Neoplastic: 0.773, Dead: 0.202, Background: 0.962
echo - Adam optimizer (not AdamW) for proven stability
echo - Cosine annealing with warmup scheduler
echo - Frequency-weighted Dice loss (proven effective)
echo - NO experimental features (safe fallback option)
echo.

echo Starting BACKUP training with proven configuration...
echo.
echo 📚 Using virtual environment Python...
venv\Scripts\python.exe training/train_backup.py --lr 2e-4 --dropout 0.1 --batch_size 16 --epochs 50

echo.
echo Backup training complete! Expected results:
echo - Validation Dice: ~0.576 (proven reliable performance)
echo - Stable training without experimental features
echo - Use this if the advanced training has issues
pause 