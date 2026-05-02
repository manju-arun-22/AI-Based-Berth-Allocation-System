"""
STEP 4 — Genetic Algorithm Berth Allocation
=============================================
What this does:
  - Takes 30 vessels from the test set
  - Shows what a naive First-Come First-Served (FCFS) schedule looks like
  - Runs the Genetic Algorithm for 200 generations
  - Shows how the schedule improves generation by generation
  - Compares GA result vs FCFS baseline
  - Saves the final optimised schedule to CSV

Run AFTER step3_train_models.py
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from src.optimization.genetic_algorithm import GeneticAlgorithmBAP, Vessel, Berth

print("=" * 55)
print("  STEP 4: Genetic Algorithm Berth Allocation")
print("=" * 55)

# ── Load test predictions from Step 3 ────────────────────
print("\nLoading test vessels and berths...")
test_df  = joblib.load("data/processed/test_df_with_predictions.pkl")
berths_df = pd.read_csv("data/synthetic/berths.csv")

# ── Take 30 vessels for the scheduling window ─────────────
sample = test_df.head(30).copy()
sample["pred_ata_h"] = pd.to_datetime(sample["ata"]).apply(
    lambda x: x.timestamp() / 3600
)

print("\n" + "-" * 55)
print("THE 30 VESSELS TO SCHEDULE:")
print("-" * 55)
print(f"  {'ID':<12} {'Type':<12} {'LOA':>7} {'Beam':>6} {'Draft':>7} {'Service':>9} {'Priority':<10}")
print(f"  {'-'*12} {'-'*12} {'-'*7} {'-'*6} {'-'*7} {'-'*9} {'-'*10}")
for _, row in sample.head(10).iterrows():
    print(f"  {row['vessel_id']:<12} {row['vessel_type']:<12} "
          f"{row['length_m']:>6.0f}m {row['beam_m']:>5.0f}m {row['draft_m']:>6.1f}m "
          f"{row['service_hours']:>8.1f}h {row['priority']:<10}")
print(f"  ... (showing 10 of 30)")

print("\n" + "-" * 55)
print("THE 5 BERTHS:")
print("-" * 55)
print(f"  {'ID':<6} {'Max LOA':>9} {'Max Beam':>10} {'Depth':>8}  Allowed Types")
print(f"  {'-'*6} {'-'*9} {'-'*10} {'-'*8}  {'-'*30}")
for _, row in berths_df.iterrows():
    print(f"  {row['id']:<6} {row['length']:>7.0f}m {row['beam']:>8.0f}m {row['depth']:>6.0f}m  {row['allowed_types']}")

# ── Build domain objects ──────────────────────────────────
vessels_for_ga = [
    Vessel(
        id=row["vessel_id"],
        length=row["length_m"],
        beam=row["beam_m"],
        draft=row["draft_m"],
        predicted_ata=float(row["pred_ata_h"]),
        service_hours=float(row["service_hours"]),
        priority=row["priority"],
        vessel_type=row["vessel_type"],
    )
    for _, row in sample.iterrows()
]

berths_for_ga = [
    Berth(
        id=row["id"],
        length=row["length"],
        depth=row["depth"],
        beam=row["beam"],
        allowed_types=str(row["allowed_types"]).split(","),
    )
    for _, row in berths_df.iterrows()
]

# ── BASELINE: First-Come First-Served ─────────────────────
print("\n" + "-" * 55)
print("BASELINE: First-Come First-Served (FCFS)")
print("-" * 55)
print("  Rule: assign each vessel to the first available")
print("  compatible berth in order of arrival.")

def fcfs_cost(vessels, berths):
    """Simple FCFS scheduler respecting LOA, beam, draft, and cargo type."""
    berth_free = {b.id: 0.0 for b in berths}
    total_wait = 0.0
    violations = 0
    for v in vessels:
        assigned = False
        for b in berths:
            compatible = (
                v.length <= b.length
                and v.beam   <= b.beam
                and v.draft  <= b.depth
                and v.vessel_type in b.allowed_types
            )
            if compatible:
                start = max(v.predicted_ata, berth_free[b.id])
                wait  = max(0, start - v.predicted_ata)
                total_wait += wait
                berth_free[b.id] = start + v.service_hours + 1.0
                assigned = True
                break
        if not assigned:
            violations += 1000
    return total_wait + violations

fcfs_cost_val = fcfs_cost(vessels_for_ga, berths_for_ga)
print(f"  FCFS total cost: {fcfs_cost_val:,.1f}")

# ── GENETIC ALGORITHM ─────────────────────────────────────
print("\n" + "-" * 55)
print("GENETIC ALGORITHM — HOW IT WORKS:")
print("-" * 55)
print("  Population  : 100 candidate schedules (chromosomes)")
print("  Generations : 200 evolution cycles")
print("  Selection   : Tournament (keep the best)")
print("  Crossover   : Swap gene segments between parents")
print("  Mutation    : Randomly reassign berth/time (10%)")
print("  Elitism     : Top 10% survive unchanged each gen")
print("\n  Fitness = -(waiting_time + 0.3*idle_time + penalties)")
print("  LOA violation:          1000 penalty  (vessel too long)")
print("  Beam violation:         1000 penalty  (vessel too wide)")
print("  Draft violation:        1000 penalty  (vessel too deep)")
print("  Cargo type mismatch:    1000 penalty  (wrong terminal)")
print("  Tidal window (draft>12m): 300 penalty (scheduled at low tide)")
print("  Excessive wait (>8h):     50 penalty")
print("  High priority vessel:      3x wait penalty")

print("\nRunning GA...")
print(f"  {'Generation':>12}  {'Best Cost':>12}  {'Avg Cost':>12}")
print(f"  {'-'*12}  {'-'*12}  {'-'*12}")

ga = GeneticAlgorithmBAP(
    vessels=vessels_for_ga,
    berths=berths_for_ga,
    pop_size=100,
    generations=200,
)
best_schedule, best_fitness = ga.optimize(verbose=True)

# ── RESULTS ───────────────────────────────────────────────
final_cost = -best_fitness
print("\n" + "-" * 55)
print("RESULTS COMPARISON:")
print("-" * 55)
improvement = (fcfs_cost_val - final_cost) / fcfs_cost_val * 100
print(f"  FCFS baseline cost : {fcfs_cost_val:>10,.1f}")
print(f"  GA optimised cost  : {final_cost:>10,.1f}")
print(f"  Improvement        : {improvement:>10.1f}%")

# ── SHOW FINAL SCHEDULE ───────────────────────────────────
print("\n" + "-" * 55)
print("FINAL OPTIMISED SCHEDULE (first 10 vessels):")
print("-" * 55)
schedule = ga.decode_schedule(best_schedule)
print(f"  {'Vessel':<12} {'Berth':<8} {'Start offset':>14} {'Service':>10} {'Priority':<10}")
print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*10} {'-'*10}")
for row in schedule[:10]:
    print(f"  {row['vessel_id']:<12} {row['berth_id']:<8} "
          f"{row['start_offset']:>13.1f}h {row['service_hours']:>9.1f}h "
          f"{row['priority']:<10}")
print(f"  ... (showing 10 of 30)")

# ── SAVE SCHEDULE ─────────────────────────────────────────
os.makedirs("results/reports", exist_ok=True)
schedule_df = pd.DataFrame(schedule)
schedule_df.to_csv("results/reports/ga_schedule.csv", index=False)

# ── GA CONVERGENCE SUMMARY ───────────────────────────────
history = ga.best_fitness_history
print("\n" + "-" * 55)
print("GA CONVERGENCE (cost at key generations):")
print("-" * 55)
for gen in [0, 25, 50, 100, 150, 199]:
    if gen < len(history):
        print(f"  Generation {gen:>3}: cost = {history[gen]:,.1f}")

print("\n  Schedule saved to results/reports/ga_schedule.csv")

print("\n" + "=" * 55)
print("  Step 4 complete.")
print("  Run step5_view_results.py to see all outputs.")
print("=" * 55)
