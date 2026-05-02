"""
Genetic Algorithm for Berth Allocation Optimization
Encodes berth schedules as chromosomes and evolves towards minimum waiting time.
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Vessel:
    id: str
    length: float
    draft: float
    beam: float            # breadth of vessel (m)
    predicted_ata: float   # hours from epoch
    service_hours: float
    priority: str          # "high", "normal", "low"
    vessel_type: str


@dataclass
class Berth:
    id: str
    length: float
    depth: float
    beam: float            # max beam the berth can accommodate (m)
    allowed_types: list    # vessel types this berth serves


class GeneticAlgorithmBAP:
    """
    Genetic Algorithm for Berth Allocation Problem.
    Chromosome: list of (berth_id, start_time) for each vessel.
    Objective: minimize weighted sum of waiting time + idle time + violations.
    """

    def __init__(
        self,
        vessels: List[Vessel],
        berths:  List[Berth],
        pop_size:    int   = 150,
        generations: int   = 300,
        crossover_rate: float = 0.8,
        mutation_rate:  float = 0.1,
        elite_pct:      float = 0.10,
        w_wait:  float = 1.0,
        w_idle:  float = 0.3,
        w_pref:  float = 0.5,
    ):
        self.vessels     = vessels
        self.berths      = berths
        self.pop_size    = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.elite_pct      = elite_pct
        self.w_wait  = w_wait
        self.w_idle  = w_idle
        self.w_pref  = w_pref
        self.n_vessels = len(vessels)
        self.n_berths  = len(berths)
        self.best_fitness_history = []

    # ── CHROMOSOME ──────────────────────────────────────────────────────────

    def _random_chromosome(self):
        """Chromosome: [(berth_idx, start_time_offset), ...] per vessel."""
        return [
            (random.randint(0, self.n_berths - 1),
             random.uniform(0, 24))
            for _ in range(self.n_vessels)
        ]

    # ── FITNESS ─────────────────────────────────────────────────────────────

    def _fitness(self, chromosome) -> float:
        total_wait      = 0.0
        total_idle      = 0.0
        total_violations = 0.0

        # Track berth occupancy: {berth_id: [(start, end), ...]}
        berth_schedule: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(self.n_berths)}

        for v_idx, (b_idx, start_offset) in enumerate(chromosome):
            vessel = self.vessels[v_idx]
            berth  = self.berths[b_idx]

            # Find earliest valid start time for this berth
            ata = vessel.predicted_ata
            earliest_start = ata + start_offset

            # Ensure no overlap on this berth
            for (occ_start, occ_end) in berth_schedule[b_idx]:
                if earliest_start < occ_end + 1.0:  # 1-hour safety buffer
                    earliest_start = occ_end + 1.0

            end_time = earliest_start + vessel.service_hours
            berth_schedule[b_idx].append((earliest_start, end_time))
            berth_schedule[b_idx].sort()

            # Waiting time (time between ATA and actual berth start)
            waiting = max(0, earliest_start - ata)
            total_wait += waiting

            # Idle time on berth before this vessel
            if len(berth_schedule[b_idx]) > 1:
                prev_end = berth_schedule[b_idx][-2][1]
                idle = max(0, earliest_start - prev_end)
                total_idle += idle

            # Constraint violations
            if vessel.length > berth.length:
                total_violations += 1000   # LOA violation

            if vessel.beam > berth.beam:
                total_violations += 1000   # beam (width) violation

            if vessel.draft > berth.depth:
                total_violations += 1000   # draft violation

            if vessel.vessel_type not in berth.allowed_types:
                total_violations += 1000   # cargo type incompatibility

            # Tidal window: deep-draft vessels (>12m) need tide height >6.5m
            # tide_height(t) = 5.0 + 3.0 * sin(2π * t / 12.4)  where t in hours
            if vessel.draft > 12.0:
                tide_at_start = 5.0 + 3.0 * np.sin(
                    2 * np.pi * earliest_start / 12.4
                )
                if tide_at_start < 6.5:
                    total_violations += 300   # soft penalty — prefer high tide

            if waiting > 8:
                total_violations += 50     # excessive waiting

            # Priority penalty: high priority vessels penalized more for waiting
            priority_mult = {"high": 3.0, "normal": 1.0, "low": 0.5}
            total_wait += waiting * (priority_mult.get(vessel.priority, 1.0) - 1.0)

        cost = (self.w_wait * total_wait +
                self.w_idle * total_idle +
                total_violations)

        return -cost   # GA maximizes, so negate

    # ── SELECTION ───────────────────────────────────────────────────────────

    def _tournament_select(self, population, fitnesses, k=5):
        contestants = random.sample(range(len(population)), k)
        best = max(contestants, key=lambda i: fitnesses[i])
        return population[best]

    # ── CROSSOVER ───────────────────────────────────────────────────────────

    def _two_point_crossover(self, p1, p2):
        if random.random() > self.crossover_rate or self.n_vessels < 2:
            return p1[:], p2[:]
        a, b = sorted(random.sample(range(self.n_vessels), 2))
        c1 = p1[:a] + p2[a:b] + p1[b:]
        c2 = p2[:a] + p1[a:b] + p2[b:]
        return c1, c2

    # ── MUTATION ────────────────────────────────────────────────────────────

    def _mutate(self, chromosome):
        for i in range(self.n_vessels):
            if random.random() < self.mutation_rate:
                chromosome[i] = (
                    random.randint(0, self.n_berths - 1),
                    random.uniform(0, 24),
                )
        return chromosome

    # ── EVOLVE ──────────────────────────────────────────────────────────────

    def optimize(self, verbose=True) -> Tuple[list, float]:
        """Run genetic algorithm and return best schedule + fitness."""

        population = [self._random_chromosome() for _ in range(self.pop_size)]
        n_elite    = max(1, int(self.pop_size * self.elite_pct))

        best_chromosome = None
        best_fitness    = float("-inf")

        for gen in range(self.generations):
            fitnesses = [self._fitness(c) for c in population]

            # Track best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness    = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx][:]

            self.best_fitness_history.append(-best_fitness)  # store as cost

            if verbose and gen % 50 == 0:
                avg_cost = -np.mean(fitnesses)
                print(f"  Gen {gen:4d} | Best cost: {-best_fitness:,.1f} | Avg: {avg_cost:,.1f}")

            # Elitism: carry top N directly
            sorted_idx = np.argsort(fitnesses)[::-1]
            new_pop = [population[i] for i in sorted_idx[:n_elite]]

            # Fill rest via selection + crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._two_point_crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            population = new_pop

        if verbose:
            print(f"\nGA complete. Best cost: {-best_fitness:,.1f}")

        return best_chromosome, best_fitness

    def decode_schedule(self, chromosome) -> list:
        """Convert chromosome to human-readable schedule with conflict-resolved start times."""
        berth_free: Dict[int, float] = {i: 0.0 for i in range(self.n_berths)}
        schedule = []
        for v_idx, (b_idx, start_offset) in enumerate(chromosome):
            vessel = self.vessels[v_idx]
            berth  = self.berths[b_idx]

            # Resolve actual start: honour ATA + offset, then push past any occupancy
            earliest = vessel.predicted_ata + start_offset
            if berth_free[b_idx] > earliest + 1.0:
                earliest = berth_free[b_idx]
            berth_free[b_idx] = earliest + vessel.service_hours + 1.0

            schedule.append({
                "vessel_id":    vessel.id,
                "berth_id":     berth.id,
                "start_offset": round(earliest - vessel.predicted_ata, 2),
                "service_hours": vessel.service_hours,
                "priority":     vessel.priority,
            })
        return schedule
