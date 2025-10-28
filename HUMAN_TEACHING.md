# 🎓 Human Teaching Workflow

**Teach the AI how to solve ARC puzzles through demonstration!**

This guide explains how to use the Human-in-the-Loop system to create an intelligent ARC solver.

---

## 📋 Overview

Instead of relying on LLMs (which struggle with ARC-AGI-2 puzzles), **YOU** become the teacher. The workflow:

1. **You demonstrate** how to solve puzzles (using mouse/keyboard)
2. **System records** your actions automatically
3. **AI learns** from your demonstrations (behavioral cloning)
4. **AI solves** new puzzles autonomously!

---

## 🚀 Quick Start

### Step 1: Record Demonstrations

```bash
python demo_playground.py
```

**What you'll do:**
- Press **E** to enable Edit Mode
- Select colors from the palette (left side)
- Click on the CURRENT grid to paint cells
- Solve the puzzle step-by-step
- Your actions are recorded automatically!

**Controls:**
- `E` - Toggle Edit Mode (REQUIRED)
- `SPACE` - Pause/Resume
- `R` - Reset if you make a mistake
- `T` - Switch between Train/Test mode
- `N` - Next training sample
- `LEFT ARROW` - Previous sample
- `Q` - Quit and save

**Tips:**
- Start with **easy puzzles** to build the dataset
- Try to solve each puzzle **optimally** (fewest steps)
- **Diversity matters** - demonstrate different patterns
- If you make a mistake, press `R` to reset

---

### Step 2: Train the AI

After recording 10-50 demonstrations:

```bash
python train_imitation.py --epochs 100
```

**What happens:**
- Loads your demonstrations
- Trains a CNN-based policy network
- Saves checkpoints every 10 epochs
- Saves the best model automatically

**Training Options:**
```bash
# Train for more epochs (better learning)
python train_imitation.py --epochs 200

# Use larger batch size (faster training)
python train_imitation.py --batch-size 64

# Include failed attempts too (more data)
python train_imitation.py --all-demos

# Use GPU if available
python train_imitation.py --device cuda
```

---

### Step 3: Test Autonomous Solving

```bash
# Coming soon: playground.py with imitation agent
python playground.py --agent imitation --checkpoint checkpoints/imitation/imitation_best.pt
```

---

## 📊 Expected Results

| Demonstrations | Training Time | Expected Accuracy |
|----------------|---------------|-------------------|
| 10 demos | 1-2 min | ~30-40% |
| 50 demos | 5-10 min | ~50-60% |
| 100 demos | 10-20 min | ~60-70% |
| 500+ demos | 30-60 min | ~70-80%+ |

**Notes:**
- Quality > Quantity: 10 perfect demos beats 100 sloppy ones
- Diverse patterns: Cover different puzzle types
- Success rate: Only successful demos are used for training (configurable)

---

## 🎯 Best Practices

### 1. **Start Simple**
- Begin with easy puzzles (simple transformations)
- Build up to complex patterns gradually
- This mirrors curriculum learning

### 2. **Be Consistent**
- Solve similar puzzles the same way
- Use the same color palette conventions
- Helps the AI learn patterns faster

### 3. **Solve Optimally**
- Fewer steps = better demonstrations
- Don't paint unnecessary cells
- Think before each action

### 4. **Diversify**
- Cover multiple puzzle types:
  - Symmetry puzzles
  - Color mapping puzzles
  - Pattern repetition
  - Object detection
  - Flood fill operations

### 5. **Review Your Data**
```bash
python -c "from src.utils.demonstration_buffer import DemonstrationBuffer; \
           db = DemonstrationBuffer(); \
           db.load('demonstrations.pkl'); \
           db.print_summary()"
```

---

## 🔧 Advanced Usage

### Progressive Training

Train in stages as you gather more data:

```bash
# Stage 1: Initial 20 demos
python train_imitation.py --epochs 50

# Record 30 more demos...

# Stage 2: Retrain with 50 demos
python train_imitation.py --epochs 100

# Record 50 more demos...

# Stage 3: Final training with 100 demos
python train_imitation.py --epochs 200 --learning-rate 5e-5
```

### Fine-tuning

If your agent is already trained, fine-tune on new data:

```bash
# Load existing checkpoint and continue training
python train_imitation.py \
    --checkpoint checkpoints/imitation/imitation_best.pt \
    --epochs 50 \
    --learning-rate 1e-5  # Lower LR for fine-tuning
```

### Filtering Demonstrations

Only use high-quality demonstrations:

```bash
# Only use demos with 95%+ accuracy
python train_imitation.py --accuracy-threshold 95.0

# Include all attempts (even failed ones)
python train_imitation.py --all-demos
```

---

## 📈 Monitoring Training

Watch the training output:

```
Epoch   1/100 | Loss: 4.2341 | Accuracy: 12.3% | Time: 2.45s
Epoch   2/100 | Loss: 3.8912 | Accuracy: 18.7% | Time: 2.41s
...
Epoch  50/100 | Loss: 0.8234 | Accuracy: 67.4% | Time: 2.38s
   [BEST] New best accuracy: 67.4% -> Saved to checkpoints/imitation/imitation_best.pt
```

**What to look for:**
- **Loss decreasing** - Model is learning
- **Accuracy increasing** - Model predicts your actions better
- **Best model saved** - Automatically saved when accuracy improves

---

## 🐛 Troubleshooting

### "No demonstrations found"
- Run `demo_playground.py` first to record demos
- Check that `demonstrations/demonstrations.pkl` exists

### "No training data available"
- You might have no successful demos
- Try `--all-demos` to include failed attempts
- Or lower `--accuracy-threshold`

### "Training is slow"
- Use `--device cuda` if you have a GPU
- Increase `--batch-size` (default: 32)
- Reduce `--epochs` for quick experiments

### "Agent still performs poorly"
- Need more demonstrations (aim for 50+)
- Make sure demos are high quality
- Try training for more epochs (200-500)
- Check demo diversity (different puzzle types)

---

## 📁 File Structure

```
demonstrations/
├── demonstrations.pkl           # Your recorded demos
└── demonstrations_summary.json  # Human-readable summary

checkpoints/imitation/
├── imitation_best.pt           # Best model (use this!)
├── imitation_final.pt          # Last epoch model
└── imitation_epoch_10.pt       # Intermediate checkpoints
```

---

## 💡 Why This Works

**Behavioral Cloning** (Imitation Learning) is proven to work when:
1. ✅ **Expert demonstrations** are available (you!)
2. ✅ **State space** is observable (grids)
3. ✅ **Actions** are discrete (paint operations)
4. ✅ **Patterns** exist (ARC has learnable patterns)

**Advantages over LLMs:**
- ✅ No API costs
- ✅ Learns YOUR solving strategy
- ✅ Fast inference (milliseconds)
- ✅ Works offline
- ✅ Privacy (no data sent externally)

**Real-world examples:**
- AlphaGo: Started with human game demonstrations
- Self-driving cars: Learn from human drivers
- Robotic manipulation: Learn from teleoperation

---

## 🎓 Next Steps

1. ✅ Record 10-20 demonstrations
2. ✅ Train your first model
3. ⏳ Test autonomous solving (coming soon)
4. ⏳ Iterate: More demos → Better model
5. ⏳ Hybrid approach: Combine with RL fine-tuning

---

## 📞 Need Help?

**Common workflow:**
```bash
# Day 1: Record demos (30 mins - 1 hour)
python demo_playground.py

# Day 1: Train first model (5-10 mins)
python train_imitation.py --epochs 50

# Day 2: Record more demos (1 hour)
python demo_playground.py

# Day 2: Retrain with more data (10-20 mins)
python train_imitation.py --epochs 100

# Day 3: Evaluate and iterate
python playground.py --agent imitation
```

**Happy teaching! 🎉**
