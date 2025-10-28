# 🧠 Neuro-Symbolic Architecture for ARC-AGI

**Goal:** Build a complete reasoning system that combines neural networks with symbolic AI for solving ARC puzzles.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT GRID (30×30)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 ↓
╔═════════════════════════════════════════════════════════════════════╗
║                     MODULE 1: PERCEPTION                            ║
║                     (Neural Component)                              ║
╚═════════════════════════════════════════════════════════════════════╝
                                 ↓
          ┌──────────────────────┼──────────────────────┐
          ↓                      ↓                      ↓
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │ Object   │         │ Attention│         │ Feature  │
    │ Detector │         │ Map      │         │ Extractor│
    └────┬─────┘         └────┬─────┘         └────┬─────┘
         ↓                    ↓                     ↓
    Objects with         Important regions    Grid properties
    properties           & relationships      & patterns
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              ↓
╔═════════════════════════════════════════════════════════════════════╗
║                MODULE 2: SYMBOLIC REASONING                         ║
║                (Symbolic Component)                                 ║
╚═════════════════════════════════════════════════════════════════════╝
                              ↓
          ┌───────────────────┼───────────────────┐
          ↓                   ↓                   ↓
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Rule     │       │ Pattern  │       │ Program  │
    │ Inference│       │ Matcher  │       │ Synthesis│
    └────┬─────┘       └────┬─────┘       └────┬─────┘
         ↓                  ↓                   ↓
    Abstract rules     Known patterns      DSL Program
         │                  │                   │
         └──────────────────┼───────────────────┘
                            ↓
╔═════════════════════════════════════════════════════════════════════╗
║                MODULE 3: HIERARCHICAL POLICY                        ║
║                (Planning Component)                                 ║
╚═════════════════════════════════════════════════════════════════════╝
                            ↓
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
    ┌──────────┐      ┌──────────┐     ┌──────────┐
    │High-Level│      │Mid-Level │     │Low-Level │
    │ Policy   │      │ Policy   │     │ Executor │
    └────┬─────┘      └────┬─────┘     └────┬─────┘
         ↓                 ↓                 ↓
    "What to do?"    "Where to do it?"  "How to do it?"
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ↓
╔═════════════════════════════════════════════════════════════════════╗
║                MODULE 4: EXECUTION                                  ║
║                (Action Component)                                   ║
╚═════════════════════════════════════════════════════════════════════╝
                           ↓
                     GRID ACTIONS
                (paint, rotate, fill, etc.)
                           ↓
                    OUTPUT GRID
```

---

## Module Breakdown

### Module 1: Perception (Neural) 🧠

**Components:**

#### 1.1 Object Detector
- **Input:** Grid (30×30)
- **Output:** List of objects with properties
  ```python
  Object = {
      'id': int,
      'pixels': List[(x, y)],
      'bbox': (x1, y1, x2, y2),
      'color': int,
      'size': int,
      'shape': str  # 'rectangle', 'line', 'blob', etc.
  }
  ```
- **Method:** Connected components + CNN classifier

#### 1.2 Attention Mechanism
- **Input:** Grid + Task context
- **Output:** Attention map (which regions are important)
- **Method:** Transformer self-attention

#### 1.3 Feature Extractor
- **Input:** Grid
- **Output:** High-level features
  ```python
  Features = {
      'has_symmetry': bool,
      'symmetry_axis': str,  # 'horizontal', 'vertical', 'both'
      'num_colors': int,
      'grid_size': (h, w),
      'has_pattern': bool,
      'pattern_type': str
  }
  ```
- **Method:** CNN + hand-crafted feature extractors

---

### Module 2: Symbolic Reasoning 🔣

**Components:**

#### 2.1 DSL (Domain Specific Language)
```python
# ARC Primitive Operations
class ARCOperation:
    # Object operations
    rotate(obj, angle)
    mirror(obj, axis)
    translate(obj, dx, dy)
    scale(obj, factor)

    # Color operations
    recolor(obj, new_color)
    color_map(grid, mapping)

    # Grid operations
    fill_region(grid, color, region)
    crop(grid, bbox)
    extend(grid, direction, amount)

    # Pattern operations
    repeat(pattern, times)
    tile(pattern, grid_size)

    # Logic operations
    if_then_else(condition, op1, op2)
    for_each(objects, operation)

    # Relational operations
    find_largest(objects)
    find_by_color(objects, color)
    count(objects)
```

#### 2.2 Rule Inference
- **Input:** Training examples (input-output pairs)
- **Output:** Inferred transformation rules
- **Method:** Program synthesis search

#### 2.3 Pattern Matcher
- **Input:** Grid + Known patterns
- **Output:** Matched pattern templates
- **Method:** Template matching + neural classifier

---

### Module 3: Hierarchical Policy 🎯

**Components:**

#### 3.1 High-Level Policy
- **Question:** "What operation should I perform?"
- **Action Space:**
  ```python
  HighLevelActions = [
      'rotate_objects',
      'mirror_grid',
      'fill_pattern',
      'color_transform',
      'repeat_pattern',
      'crop_and_extend',
      'object_manipulation',
      'conditional_transform'
  ]
  ```
- **Method:** Transformer classifier

#### 3.2 Mid-Level Policy
- **Question:** "Which object/region to operate on?"
- **Action Space:**
  ```python
  MidLevelActions = [
      'select_object(id)',
      'select_region(bbox)',
      'select_all_objects()',
      'select_by_property(color/size/shape)'
  ]
  ```
- **Method:** Attention-based selector

#### 3.3 Low-Level Executor
- **Question:** "How to execute the operation?"
- **Action Space:** DSL program execution
- **Method:** DSL interpreter + Neural executor for complex ops

---

### Module 4: Execution ⚙️

**Components:**

#### 4.1 DSL Interpreter
- Executes DSL programs on grids
- Handles errors and edge cases

#### 4.2 Neural Executor (Fallback)
- When DSL can't handle operation
- Use CNN to perform pixel-level manipulation

---

## Training Strategy

### Phase 1: Supervised Learning (Human Demos)
```
Human solves puzzle
    ↓
Record: state + DSL program
    ↓
Train all modules:
  - Object detector: Detect objects in grids
  - Program synthesizer: Generate DSL from examples
  - Hierarchical policy: Choose correct operations
```

### Phase 2: Reinforcement Learning (Self-Play)
```
Agent attempts puzzles
    ↓
Use environment reward
    ↓
Fine-tune:
  - High-level policy (operation selection)
  - Mid-level policy (object selection)
  - Program synthesis (better programs)
```

### Phase 3: Meta-Learning (Fast Adaptation)
```
Train on many puzzles
    ↓
Learn to learn quickly
    ↓
Few-shot adaptation:
  - Given 2-3 examples
  - Infer rule immediately
  - Apply to test puzzle
```

---

## Data Flow Example

**Puzzle:** "Rotate each blue object 90° clockwise"

### Step 1: Perception
```
Input Grid:
■ □ □ ● □
■ □ □ ● □
□ □ □ □ □

Object Detector Output:
- Object 1: pixels={(0,0), (1,0)}, color=red, shape=vertical_line
- Object 2: pixels={(0,3), (1,3)}, color=blue, shape=vertical_line

Attention Map:
- High attention on blue objects
- Low attention on empty space

Features:
- num_objects: 2
- has_pattern: yes
- grid_size: (3, 5)
```

### Step 2: Symbolic Reasoning
```
Rule Inference:
"Transform blue objects differently from red objects"

Pattern Matching:
"Selective object transformation pattern"

Program Synthesis:
program = """
objects = detect_objects(grid)
for obj in objects:
    if obj.color == BLUE:
        rotate(obj, 90)
return compose(objects)
"""
```

### Step 3: Hierarchical Policy

```
High-Level Policy:
Action: "object_manipulation"

Mid-Level Policy:
Action: "select_by_property(color=BLUE)"

Low-Level Executor:
Execute DSL program
```

### Step 4: Execution
```
DSL Interpreter runs:
1. detect_objects(grid) → [obj1, obj2]
2. for obj2 (blue):
     rotate(obj2, 90)
3. compose([obj1, obj2_rotated])

Output Grid:
■ □ □ □ □
■ □ □ ● ●
□ □ □ □ □
```

---

## Implementation Plan

### Week 1: Perception Module
- [x] Design architecture
- [ ] Object detection with connected components
- [ ] CNN-based object classifier
- [ ] Attention mechanism (Transformer)
- [ ] Feature extractors

### Week 2: DSL & Symbolic Module
- [ ] Define ARC DSL primitives
- [ ] DSL interpreter
- [ ] Rule inference engine
- [ ] Pattern matcher

### Week 3: Hierarchical Policy
- [ ] High-level policy network
- [ ] Mid-level object selector
- [ ] Low-level executor
- [ ] Integration with DSL

### Week 4: Training Pipeline
- [ ] Supervised learning from demos
- [ ] RL fine-tuning
- [ ] Meta-learning wrapper
- [ ] Evaluation framework

### Week 5: Integration & Testing
- [ ] End-to-end system
- [ ] Debugging and optimization
- [ ] Documentation
- [ ] Benchmarking

---

## Expected Performance

### Current CNN Agent: ~30-40%
- Simple pixel transformations only
- No reasoning

### With Object Detection: ~45-50%
- Can reason about objects
- Still no complex logic

### With DSL + Program Synthesis: ~55-60%
- Can handle rules and conditions
- Limited compositional reasoning

### With Hierarchical Policy: ~65-70%
- Multi-step planning
- Better generalization

### With Attention + Full System: ~70-75%+
- Focus on important regions
- Near-human performance on many puzzles

---

## Key Innovations

1. **Neuro-Symbolic Integration**
   - Neural: Perception, attention, learning
   - Symbolic: Reasoning, rules, programs
   - Best of both worlds!

2. **Hierarchical Decomposition**
   - What → Where → How
   - Easier to learn and generalize

3. **DSL as Interface**
   - Interpretable programs
   - Debuggable and editable
   - Human-understandable

4. **Multi-Stage Training**
   - Human demos → RL → Meta-learning
   - Progressive capability building

---

## File Structure

```
src/
├── perception/
│   ├── object_detector.py
│   ├── attention_module.py
│   └── feature_extractor.py
├── symbolic/
│   ├── dsl.py
│   ├── dsl_interpreter.py
│   ├── rule_inference.py
│   └── pattern_matcher.py
├── policy/
│   ├── hierarchical_policy.py
│   ├── high_level_policy.py
│   ├── mid_level_policy.py
│   └── low_level_executor.py
├── agents/
│   ├── neuro_symbolic_agent.py
│   ├── program_synthesis_agent.py
│   └── hierarchical_agent.py
└── training/
    ├── supervised_trainer.py
    ├── rl_trainer.py
    └── meta_learner.py
```

---

**Ready to implement? Let's start with Object Detection!** 🚀
