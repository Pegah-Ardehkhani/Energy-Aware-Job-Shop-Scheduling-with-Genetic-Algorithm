# Energy-Aware Job Shop Scheduling with NSGA-II 🏭

A multi-objective optimization system for job shop scheduling that simultaneously minimizes makespan, energy costs, and tardiness using the NSGA-II (Non-dominated Sorting Genetic Algorithm II) evolutionary algorithm.

## 🚀 Features

- **Multi-objective optimization** with three objectives:
  - **Makespan**: Total completion time
  - **Energy Cost**: Time-of-use electricity costs
  - **Tardiness**: Penalty for late job completion
- **Energy-aware scheduling** with configurable tariff periods
- **Machine-specific energy profiles** (idle vs. working power consumption)
- **Flexible job shop configuration** with precedence constraints
- **Pareto front visualization** in 2D and 3D
- **Gantt chart generation** for schedule visualization

## 📋 Requirements

```
numpy
matplotlib
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-aware-job-shop-scheduling.git
cd energy-aware-job-shop-scheduling
```

2. Install required dependencies:
```bash
pip install numpy matplotlib
```

## 🏃‍♂️ Quick Start

```python
from nsga2_scheduler import *

# Define machines with energy profiles
machines = [
    Machine("M1", idle_power=0.1, working_power=1.0),
    Machine("M2", idle_power=0.15, working_power=1.5),
    Machine("M3", idle_power=0.08, working_power=0.8),
]

# Define operations and jobs
op1_1 = Operation("Op1.1", "J1", "M1", 5, 1)
op1_2 = Operation("Op1.2", "J1", "M2", 3, 2)
job1 = Job("J1", [op1_1, op1_2], due_date=10)

# Define energy cost model
energy_tariffs = {
    "off_peak": (0, 7, 0.10),
    "peak": (7, 17, 0.25),
    "off_peak_evening": (17, 24, 0.15),
}
energy_model = EnergyCostModel(energy_tariffs)

# Create and run scheduler
scheduler = NSGA2Scheduler(
    jobs=[job1], 
    machines=machines, 
    energy_model=energy_model,
    population_size=100,
    generations=300
)

results = scheduler.run()

# Visualize results
plot_pareto_front(results["pareto_front"])
```

## 📊 Core Components

### 1. Operation
Represents a single manufacturing operation with:
- Processing time and machine assignment
- Energy consumption calculation
- Precedence constraints within jobs

### 2. Job
Collection of operations with:
- Sequential execution requirements
- Due date constraints
- Release time specifications

### 3. Machine
Manufacturing resource with:
- Configurable energy profiles (idle/working power)
- Availability tracking
- Schedule management

### 4. Energy Cost Model
Time-of-use electricity pricing with:
- Multiple tariff periods
- Hourly cost calculation
- Dynamic pricing support

### 5. NSGA-II Scheduler
Multi-objective evolutionary algorithm featuring:
- **Population initialization** with diverse strategies
- **Fast non-dominated sorting** for Pareto ranking
- **Crowding distance** for diversity preservation
- **Tournament selection** with elitism
- **Order crossover (OX1)** for permutation chromosomes
- **Multi-strategy mutation** (swap, insertion, inversion)

## 🎯 Algorithm Overview

The NSGA-II algorithm works by:

1. **Encoding**: Jobs represented as permutation chromosomes
2. **Decoding**: Converting chromosomes to feasible schedules
3. **Evaluation**: Calculating makespan, energy cost, and tardiness
4. **Selection**: Multi-objective tournament selection
5. **Reproduction**: Order crossover and mutation operators
6. **Replacement**: Elitist selection maintaining population diversity

## 📈 Visualization

### Pareto Front Plotting
- 3D scatter plot showing all three objectives
- 2D projections for pairwise objective analysis
- Color-coded solutions for easy interpretation

### Gantt Charts
- Machine-based timeline visualization
- Job-specific color coding
- Operation details and timing

## ⚙️ Configuration

### Population Parameters
```python
NSGA2Scheduler(
    population_size=150,    # Population size
    generations=500,        # Number of generations
    crossover_rate=0.9,     # Crossover probability
    mutation_rate=0.2       # Mutation probability
)
```

### Energy Tariffs
```python
energy_tariffs = {
    "night": (0, 6, 0.08),        # (start_hour, end_hour, cost_per_unit)
    "morning": (6, 12, 0.20),
    "afternoon": (12, 18, 0.15),
    "evening": (18, 24, 0.25),
}
```

## 📋 Example Output

```
--- NSGA-II Optimization Complete ---
Pareto Front Size: 12

--- Pareto Front Solutions ---
Solution 1:
  Makespan: 18
  Energy Cost: 4.25
  Tardiness: 2

Solution 2:
  Makespan: 20
  Energy Cost: 3.80
  Tardiness: 0

Best individual objectives:
  Best Makespan: 18
  Best Energy Cost: 3.80
  Best Tardiness: 0
```

## 🔬 Research Applications

This implementation is suitable for:
- **Manufacturing scheduling** with energy constraints
- **Sustainable production** planning
- **Multi-objective optimization** research
- **Smart factory** implementations
- **Energy cost reduction** studies

## 🛠️ Customization

### Adding New Objectives
Extend the `_decode_chromosome` method to calculate additional objectives:
```python
def _decode_chromosome(self, chromosome):
    # ... existing code ...
    new_objective = calculate_new_objective(scheduled_operations)
    return scheduled_operations, makespan, energy_cost, tardiness, new_objective
```

### Custom Machine Types
Create specialized machine classes:
```python
class AdvancedMachine(Machine):
    def __init__(self, machine_id, setup_time=0, maintenance_cost=0):
        super().__init__(machine_id)
        self.setup_time = setup_time
        self.maintenance_cost = maintenance_cost
```

## 📚 References

- Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- Pinedo, M. (2016). "Scheduling: Theory, Algorithms, and Systems"
- Zhang, R., & Chiong, R. (2016). "Solving the energy-efficient job shop scheduling problem"

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Keywords**: Job Shop Scheduling, Multi-objective Optimization, NSGA-II, Energy Efficiency, Manufacturing, Evolutionary Algorithm, Pareto Front, Production Planning
