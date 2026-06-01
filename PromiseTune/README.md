
# PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning

## Source

**Paper Preprint**: https://arxiv.org/abs/2507.05995

## Code Structure
   - **Code/**: Core implementation of PromiseTune
     - `PromiseTune.py`: Main algorithm implementation
     - **Data/**: System performance datasets (12 systems)
     - **util/**: Utility modules for tuning operations
       - `ei.py`: Expected Improvement calculation
       - `get_objective.py`: Objective function handling
       - `helper.py`: Helper functions
       - `matrix.py`: Matrix operations
       - `read_file.py`: Data loading utilities
   - **Data/**: Additional dataset copies
   - **results/**: Experimental results for all baseline tuners
   - **parameter_results/**: Sensitivity analysis results
   - **RQs/**: Research question specific data and results
   - `requirements.txt`: Python package dependencies
   - `rq1_effectiveness.py`, `rq2_ablation.py`, `rq3_sensitivity.py`: Scripts for reproducing research questions

## Setup

**Note**: The experiments can be completed on a standard desktop or laptop computer. Runtime varies by system complexity (from minutes to hours per system).

### Software Requirements

**Operating System**: 
- Linux (Ubuntu 18.04+ recommended)
- Windows 10/11 (with Python support)

**Dependencies**:
- Python 3.8 or higher
- R studio (for scott_test, if not need, remove related code)
- Required Python packages (automatically installed via requirements.txt):
  - numpy >= 1.19.0
  - pandas >= 1.1.0
  - scikit-learn >= 0.23.0
  - scipy >= 1.5.0
  - matplotlib >= 3.3.0
  - seaborn >= 0.11.0
  - Additional dependencies listed in `requirements.txt`


**Standard Installation**:

1. **Download the repository**:
```bash
cd PromiseTune
```

2. **Set up Python environment** (recommended: use virtual environment):
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Update (2026-03)

- Added configurable causal refresh interval support for PromiseTune.
- You can now control how frequently the causal model is refreshed via `causal_refresh_interval`.
- Larger interval values (e.g., 10) reduce causal-model rebuild frequency and often reduce runtime.


### Basic Usage Example

After installation, verify that PromiseTune is working correctly with this basic test:

```bash
cd Code
python3 PromiseTune.py
```

**Expected Output**:
The script will run PromiseTune on the default dataset (e.g., 7z) and display:
- Progress messages showing iteration numbers
- Performance values found at each iteration
- Final best configuration discovered
- Output files saved in the `results/` directory

**Example Expected Output**:
```
=>Loop: 1, =>Reward: 5650.60, =>Config: [1, 1, 1, 1, 1, 0, 1, 0, 4, 128, 32, 128]
...
Results saved to: ../results/Promisetune/7z_seed.csv
```

If you see similar output without errors, the installation is successful!

### Custom Dataset Usage

To use PromiseTune with your own dataset:

1. **Prepare your CSV file**:
   - Place the CSV file in `Code/Data/`
   - The target column must start with "$<" (e.g., "$<runtime")
   - Default: minimize the target (for maximization, reverse the values)

2. **Modify the code**:
   ```python
   # In PromiseTune.py, add your dataset to the list
   datasets = ['7z', 'your_dataset_name']
   ```

3. **Run PromiseTune**:
   ```bash
   cd Code
   python3 PromiseTune.py
   ```

### Reproducing Paper Results

This section provides detailed instructions to reproduce all four research questions (RQ1-RQ4) from the paper.

#### RQ1: Effectiveness Comparison

**Goal**: Compare PromiseTune with 11 state-of-the-art tuners across 12 systems.

**Command**:
```bash
python3 rq1_effectiveness.py
```
**Note**:
- We do not use skott-knott in this version because it need R package which is not easy to install. If you want to use it, please install R package first and run rq1_effectiveness_test.py

**What it does**:
- Compares results with baseline methods (stored in `results/` directory)
- Generates performance comparison tables and statistical tests
- Creates visualizations showing rank comparisons

**Expected Output**:
- `scott_knott_results.md`: Statistical analysis results
- `normalized_performance.md`: Performance comparison table
- Charts showing PromiseTune's superior rank

---

#### RQ2: Ablation Study

**Goal**: Evaluate the impact of the causal rule component.

**Command**:
```bash
python3 rq2_ablation.py
```

**What it does**:
- Compares performance to demonstrate the effectiveness of causal purification

**Alternative Manual Setup**:
```python
# In PromiseTune.py, modify:
rule = False  # Disable causal rules
# Then run: python3 PromiseTune.py
```

**Expected Output**:
- `res_ablation.md`: Comparison results

---

#### RQ3: Sensitivity Analysis

**Goal**: Analyze the sensitivity of the key parameter `l`.

**Command**:
```bash
python3 rq3_sensitivity.py
```

**What it does**:
- Analyzes performance stability across different parameter settings

**Expected Output**:
- `sensitivity_results_all_budgets.md`: Comparison results

---

#### RQ4: Explainability Case Study

**Goal**: Demonstrate the explainability of PromiseTune using x264 as a case study.

**Location**: Results are already provided in `RQs/RQ4/`

**To explore**:
```bash
# View the extracted rules for x264
cat RQs/RQ4/rules_in_x264.txt

# Or analyze other systems
cd Code
# Modify PromiseTune.py to enable rule extraction
python3 PromiseTune.py
# Rules will be saved with performance results
```

**What to examine**:
- Rules extracted by PromiseTune
- How rules define promising configuration regions

---

#### Complete Reproduction

To reproduce all results at once:

```bash
# Run all RQ scripts sequentially
python3 rq1_effectiveness.py
python3 rq2_ablation.py
python3 rq3_sensitivity.py

# Results will be in respective output files
ls *.md
```

**Note on Variance**: Due to the stochastic nature of optimization algorithms, exact numerical values may vary slightly from the paper (±1%), but statistical trends and rankings should be consistent.

---

## Data

**Context**: This artifact includes performance measurement data from 12 real-world configurable software systems across different domains (databases, compilers, video encoders, etc.). Each dataset contains configuration-performance pairs measured from actual system executions.

**Data Sources**: 
- The datasets are collected from diverse systems including 7z, DConvert, ExaStencils, BDB-C, DeepArch, PostgreSQL, JavaGC, Storm, x264, Redis, HSQLDB, and LLVM
- Performance measurements were obtained by executing each system with different configuration settings and recording the resulting performance metrics (runtime, throughput, latency, etc.)
- All measurements are real-world data from actual system executions, not synthetic data

**Ethical and Legal Considerations**: 
- All systems used are open-source or publicly available software
- No personally identifiable information (PII) or sensitive data is included
- The data consists solely of system performance measurements
- All datasets comply with their respective software licenses


### General Tuners
 - Random: a commonly used random search strategy, which is simple to implement and performs well in some cases.
 - [SMAC](https://github.com/automl/SMAC3): a sequential model-based optimization, which deals with categorical parameters by constructing a random forest model to select promising configurations in the algorithm configuration space.
 - [GA](https://github.com/jMetal/jMetalPy): a genetic algorithm for optimal configurations using natural selection and cross-variance heuristics.
 - [MBO](https://github.com/PKU-DAIR/open-box): a bayesian model-based approach which constructs the mixed kernel gausian process model to predict the objective function and uses the model to guide the search.


### Configuration Tuners
 - [FLASH](https://github.com/FlashRepo/Flash-SingleConfig): a sequential model-based approach that efficiently solves the single-objective configuration optimization problem for software systems and requires fewer measurements in the search for better configurations by using a priori knowledge of the configuration space to select the next promising configuration.
 - [Unicorn](https://github.com/softsys4ai/unicorn): an approach to analyze the performance of configurable systems through causal reasoning, which recovers the causal structure from performance data to help identify the root causes of performance failures, estimate parameter causal effects, and give recommendations for optimal configurations.

### Compiler Tuners
 - [BOCA](https://github.com/BOCA313/BOCA): an automatic compiler tuning method based on bayesian optimization, which designs novel search strategies by approximating the objective function using a tree-based model.
 - [CFSCA](https://github.com/zhumxxx/CFSCA): a compiler auto-tuning technique based on key flags selection, which determines potentially relevant flags by analyzing the program structure and compiler documentation, and then identifies the key flags by statistical analysis to narrow down the search space, so as to select an optimized sequence for the target program to improve performance.

### Database Tuners
 - [LlamaTune](https://github.com/uw-mad-dash/llamatune): a tool for configuration tuning of database management systems that utilizes techniques such as stochastic low-dimensional projection, special value bias sampling, and knob-value bucketing to reduce the search space.

 - [OtterTune](https://github.com/cmu-db/ottertune): an automated database management system tuning tool that combines supervised and unsupervised machine learning methods to optimize database configurations by reusing previous tuning data to select important configuration knobs, map workloads, and recommend settings.

---

## Additional Information

## Systems

| System     | Version | Benchmark    | Domain            | Language | Performance         | $B/N$ | $S_{space}$    |
| ---------- | ------- | -------------------------------------------------- | ----------------- | -------- | ------------------- | ------------------------------ | ------------------------- |
| 7z         | 9.20    | Compressing a 3 GB directory                       | File Compressor   | C++      | Runtime (ms)        | 11/3                           | $1.68 \times 10^8$         |
| DConvert   | 1.0.0   | Transform resources at different scales            | Image Scaling     | Java     | Runtime (s)         | 17/1                           | $1.05 \times 10^7$         |
| ExaStencils| 1.2     | Default benchmarks                                 | Code Generator    | Scala    | Runtime (ms)        | 7/5                            | $1.61 \times 10^9$         |
| BDB-C      | 18.0    | Benchmark provided by vendor                       | Database          | C        | Latency (s)         | 16/0                           | $6.55 \times 10^4$         |
| DeepArch   | 2.2.4   | UCR Archive time series dataset                    | Deep Learning Tool| Python   | Runtime (min)       | 12/0                           | $4.10 \times 10^3$         |
| PostgreSQL | 22.0    | PolePosition 0.6.0                                 | Database          | C        | Runtime (ms)        | 6/3                            | $1.42 \times 10^9$         |
| JavaGC     | 7.0     | DaCapo benchmark suite                             | Java Runtime      | Java     | Runtime (ms)        | 12/23                          | $2.67 \times 10^{41}$      |
| Storm      | 0.9.5   | Randomly generated benchmark                       | Data Analytics    | Clojure  | Messages per Second | 12/0                           | $4.10 \times 10^{3}$       |
| x264       | 0.157   | Video files of various sizes                       | Video Encoder     | C        | Peak signal-to-noise ratio | 4/13                   | $6.43 \times 10^{26}$      |
| Redis      | 6.0     | Sysbench                                           | Database          | C        | Requests per second | 1/8                            | $5.78 \times 10^{16}$      |
| HSQLDB     | 19.0    | PolePosition 0.6.0                                 | Database          | Java     | Runtime (ms)        | 18/0                           | $2.62 \times 10^5$         |
| LLVM       | 3.0     | LLVM’s test suite                                  | Compiler          | C++      | Runtime (ms)        | 10/0                           | $1.02 \times 10^3$         |


### Extending to New Systems

To apply PromiseTune to a new system:

1. **Collect Performance Data**:
   - Measure system performance with different configurations
   - Format as CSV with columns: config_param1, config_param2, ..., $<performance

2. **Add Dataset**:
   - Place CSV in `Code/Data/your_system.csv`
   - Ensure target column starts with "$<"

3. **Configure PromiseTune**:
   ```python
   # In PromiseTune.py
   datasets = ['7z', 'x264', 'your_system']
   ```

4. **Run Tuning**:
   ```bash
   cd Code
   python3 PromiseTune.py
   ```

### License

This project is licensed under the terms specified in the LICENSE file.

---

## Acknowledgments

We thank the authors of the baseline tuners for making their implementations available and the maintainers of the benchmark systems used in our evaluation.
