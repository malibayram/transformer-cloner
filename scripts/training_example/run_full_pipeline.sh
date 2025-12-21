#!/bin/bash
# Complete Training Pipeline for Gemma3-141M Turkish
# Run on A100 instance
#
# TWO-PHASE PIPELINE:
# Phase 1: Test Run (~30 min) - Validate all stages with 10 steps each
# Phase 2: Full Run (~24h) - Complete training after test passes
#
# Stages:
# 1. Pretraining Stage 0: Cosmos Corpus (general Turkish)
# 2. Pretraining Stage 1: Turkish Books (high-quality)
# 3. SFT Stages: Alpaca → Medical → Wikipedia → Instructions

set -e  # Exit immediately if a command exits with a non-zero status

cd /root/work || exit 1
source /root/work/venv/bin/activate

echo "=============================================="
echo "🚀 GEMMA3-141M TURKISH TRAINING PIPELINE"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Cleanup old test directories only (don't kill processes - they might be this script!)
echo "🧹 Removing old test checkpoint directories..."
rm -rf gemma3-141m-*-test
echo "✅ Cleanup complete."

# ========================================
# PHASE 1: TEST RUN (~30 min total)
# Validates entire pipeline with 10 steps
# ========================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           � PHASE 1: TEST RUN (Validation)                       ║"
echo "║           ~30 minutes - 10 steps per stage                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

echo "📍 TEST: Pretraining Stage 0 (Cosmos)"
python3 train_gemma3_141m.py --stage 0 --test-mode
echo "✅ Test Stage 0 passed at $(date)"

echo ""
echo "📍 TEST: Pretraining Stage 1 (Books)"
python3 train_gemma3_141m.py --stage 1 --test-mode
echo "✅ Test Stage 1 passed at $(date)"

echo ""
echo "📍 TEST: All SFT Stages"
python3 train_sft.py --all --test-mode
echo "✅ All SFT test stages passed at $(date)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           ✅ PHASE 1 COMPLETE - All tests passed!                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Cleanup test checkpoints to save space
echo "🧹 Removing test checkpoints..."
rm -rf gemma3-141m-*-test
echo ""

# ========================================
# PHASE 2: FULL TRAINING RUN
# ========================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           � PHASE 2: FULL TRAINING RUN                           ║"
echo "║           ~24 hours total                                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Remove old full training directories
echo "🧹 Removing old training checkpoint directories..."
rm -rf gemma3-141m-stage* gemma3-141m-sft*
echo ""

echo "📍 PRETRAINING STAGE 0: Cosmos Corpus (5000 steps)"
python3 train_gemma3_141m.py --stage 0
echo "✅ Pretraining Stage 0 complete at $(date)"
echo ""

echo "📍 PRETRAINING STAGE 1: Turkish Books (4000 steps)"
python3 train_gemma3_141m.py --stage 1
echo "✅ Pretraining Stage 1 complete at $(date)"
echo ""

echo "📍 SFT TRAINING: All 4 Stages"
python3 train_sft.py --all
echo "✅ All SFT stages complete at $(date)"
echo ""

echo "=============================================="
echo "🎉 FULL PIPELINE COMPLETE!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Models saved to HuggingFace Hub:"
echo "  Pretraining:"
echo "  - alibayram/gemma3-141m-stage1-cosmos"
echo "  - alibayram/gemma3-141m-stage2-books"
echo "  SFT:"
echo "  - alibayram/gemma3-141m-sft-alpaca"
echo "  - alibayram/gemma3-141m-sft-medical"
echo "  - alibayram/gemma3-141m-sft-wiki"
echo "  - alibayram/gemma3-141m-sft-instructions"
