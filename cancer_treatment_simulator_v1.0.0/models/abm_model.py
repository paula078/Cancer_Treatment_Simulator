"""
Agent-Based Model for Cell-Level Cancer Dynamics (3D Enhanced)

This module implements a 3D agent-based model for simulating individual
cell behaviors, interactions, and spatial dynamics in cancer treatment.

Includes tumor cells, immune cells, and microenvironmental factors with
full 3D spatial representation, advanced cell behaviors, and realistic
tissue architecture.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
import random
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter


class CellType(Enum):
    """Cell type enumeration."""

    TUMOR = "tumor"
    IMMUNE = "immune"
    DEAD = "dead"
    EMPTY = "empty"


@dataclass
class Cell:
    """Individual cell agent with 3D positioning and advanced properties."""

    x: int
    y: int
    z: int  # Added 3D coordinate
    cell_type: CellType
    age: float = 0.0
    health: float = 1.0
    division_timer: float = 0.0
    drug_resistance: float = 0.0
    radiation_damage: float = 0.0
    oxygen_level: float = 1.0

    # Enhanced properties
    energy: float = 1.0
    migration_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    phenotype: str = "normal"
    metabolic_state: str = "aerobic"  # aerobic, anaerobic, quiescent
    death_timer: float = 0.0

    # Immune cell specific
    activation_level: float = 0.0
    cytokine_production: float = 0.0

    # Tumor cell specific
    invasiveness: float = 0.0
    stem_like: bool = False
    mutation_count: int = 0

    def __post_init__(self):
        """Initialize cell-specific properties based on type."""
        if self.cell_type == CellType.TUMOR:
            self.division_time = random.uniform(18, 36)  # hours
            self.drug_resistance = random.uniform(0, 0.3)
            self.invasiveness = random.uniform(0.1, 0.8)
            self.stem_like = random.random() < 0.05  # 5% cancer stem cells
            if self.stem_like:
                self.drug_resistance *= 2.0
                self.division_time *= 1.5

        elif self.cell_type == CellType.IMMUNE:
            self.division_time = random.uniform(40, 80)  # hours
            self.cytotoxicity = random.uniform(0.3, 1.0)
            self.activation_level = random.uniform(0.2, 0.6)
            self.cytokine_production = random.uniform(0.1, 0.5)

    def get_position(self) -> Tuple[int, int, int]:
        """Get 3D position as tuple."""
        return (self.x, self.y, self.z)

    def distance_to(self, other: "Cell") -> float:
        """Calculate 3D Euclidean distance to another cell."""
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def is_hypoxic(self) -> bool:
        """Check if cell is in hypoxic conditions."""
        return self.oxygen_level < 0.3

    def update_metabolic_state(self):
        """Update metabolic state based on oxygen and energy."""
        if self.oxygen_level > 0.5 and self.energy > 0.7:
            self.metabolic_state = "aerobic"
        elif self.oxygen_level > 0.2:
            self.metabolic_state = "anaerobic"
        else:
            self.metabolic_state = "quiescent"


class TumorImmuneABM3D:
    """
    3D Agent-based model for tumor-immune cell interactions.

    Simulates individual cell behaviors on a 3D grid including:
    - Tumor cell proliferation, death, and invasion
    - Immune cell migration and cytotoxicity
    - Drug and radiation effects with 3D diffusion
    - Microenvironmental factors (oxygen, nutrients, growth factors)
    - Vascular network simulation
    - Tissue architecture effects
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (80, 80, 40),
        parameters: Optional[Dict] = None,
    ):
        """
        Initialize 3D ABM.

        Args:
            grid_size: (width, height, depth) of simulation grid
            parameters: Model parameters
        """
        self.width, self.height, self.depth = grid_size
        self.params = self._get_default_parameters()
        if parameters:
            self.params.update(parameters)

        # Initialize 3D grid and fields
        self.grid = np.full((self.width, self.height, self.depth), None, dtype=object)
        self.oxygen_field = np.ones((self.width, self.height, self.depth))
        self.drug_field = np.zeros((self.width, self.height, self.depth))
        self.radiation_field = np.zeros((self.width, self.height, self.depth))
        self.nutrient_field = np.ones((self.width, self.height, self.depth))
        self.growth_factor_field = np.zeros((self.width, self.height, self.depth))
        self.ecm_density = (
            np.ones((self.width, self.height, self.depth)) * 0.5
        )  # Extracellular matrix

        # Vascular network (simplified as blood vessel locations)
        self.vascular_network = self._initialize_vasculature()

        # Cell tracking
        self.cells = []
        self.cell_counts = {CellType.TUMOR: 0, CellType.IMMUNE: 0, CellType.DEAD: 0}
        self.time = 0.0

        # Enhanced history tracking
        self.history = {
            "time": [],
            "tumor_count": [],
            "immune_count": [],
            "dead_count": [],
            "avg_oxygen": [],
            "total_drug": [],
            "tumor_volume": [],
            "invasion_depth": [],
            "immune_activation": [],
            "metabolic_states": [],
        }

    def _get_default_parameters(self) -> Dict:
        """Enhanced default ABM parameters for 3D model."""
        return {
            # Tumor cell parameters
            "tumor_division_rate": 0.015,  # Reduced for 3D
            "tumor_death_rate": 0.001,
            "tumor_migration_rate": 0.08,  # Reduced for 3D constraints
            "tumor_max_age": 2000,
            "tumor_invasion_rate": 0.05,  # Invasive migration
            "tumor_stem_fraction": 0.05,  # Cancer stem cell fraction
            # Immune cell parameters
            "immune_migration_rate": 0.25,  # Higher than tumor
            "immune_cytotoxicity": 0.12,
            "immune_recruitment_rate": 0.008,
            "immune_death_rate": 0.002,
            "immune_max_age": 1000,
            "immune_activation_threshold": 0.3,
            # Drug parameters
            "drug_diffusion_rate": 0.15,  # 3D diffusion
            "drug_decay_rate": 0.05,
            "drug_cytotoxicity": 0.06,
            "drug_resistance_evolution": 0.0008,
            "drug_vascular_delivery": 0.8,  # Delivery efficiency from vessels
            # Radiation parameters
            "radiation_kill_prob": 0.75,  # Base kill probability
            "radiation_repair_rate": 0.12,
            "radiation_bystander": 0.025,
            "radiation_oer_factor": 2.5,  # Oxygen enhancement ratio
            # Microenvironment parameters
            "oxygen_consumption": 0.012,
            "oxygen_diffusion": 0.25,  # 3D diffusion
            "oxygen_vascular_supply": 1.0,
            "hypoxia_threshold": 0.25,
            "hypoxia_resistance": 2.2,
            # Nutrient parameters
            "nutrient_consumption": 0.008,
            "nutrient_diffusion": 0.18,
            "nutrient_threshold": 0.3,
            # Growth factors
            "growth_factor_production": 0.02,
            "growth_factor_decay": 0.08,
            "growth_factor_diffusion": 0.12,
            # ECM and mechanical
            "ecm_degradation_rate": 0.01,
            "mechanical_resistance": 0.3,
            "carrying_capacity_local": 0.85,
            # Vascular parameters
            "vessel_density": 0.15,  # Fraction of space with vessels
            "angiogenesis_rate": 0.005,
            "vessel_regression_rate": 0.002,
            # 3D specific
            "z_migration_penalty": 0.7,  # Reduced z-direction migration
            "layer_interaction_range": 2,  # Interaction range between layers
        }

    def _initialize_vasculature(self) -> np.ndarray:
        """Initialize simplified vascular network."""
        vessels = np.zeros((self.width, self.height, self.depth), dtype=bool)

        # Create main vessels along z-axis (blood supply from tissue surface)
        for i in range(0, self.width, 8):
            for j in range(0, self.height, 8):
                if random.random() < self.params["vessel_density"]:
                    vessels[i, j, :] = True

        # Add some branching vessels
        for z in range(self.depth):
            for _ in range(int(self.width * self.height * 0.01)):
                x, y = (
                    random.randint(0, self.width - 1),
                    random.randint(0, self.height - 1),
                )
                vessels[x, y, z] = True

        return vessels

    def initialize_tumor(self, center: Tuple[int, int, int], radius: int = 8):
        """
        Initialize 3D tumor cluster.

        Args:
            center: (x, y, z) center position
            radius: Initial tumor radius
        """
        center_x, center_y, center_z = center

        for x in range(max(0, center_x - radius), min(self.width, center_x + radius)):
            for y in range(
                max(0, center_y - radius), min(self.height, center_y + radius)
            ):
                for z in range(
                    max(0, center_z - radius), min(self.depth, center_z + radius)
                ):
                    distance = np.sqrt(
                        (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
                    )
                    if distance <= radius:
                        # Higher density in center, lower at edges
                        prob = max(0.4, 0.9 - distance / radius)
                        if random.random() < prob:
                            self.add_cell(x, y, z, CellType.TUMOR)

    def initialize_immune_cells(self, count: int = 100):
        """
        Initialize immune cells randomly in 3D space.

        Args:
            count: Number of immune cells to add
        """
        for _ in range(count):
            position = self.get_random_empty_position()
            if position[0] is not None:
                self.add_cell(position[0], position[1], position[2], CellType.IMMUNE)

    def add_cell(self, x: int, y: int, z: int, cell_type: CellType) -> Optional[Cell]:
        """
        Add cell to 3D grid.

        Args:
            x, y, z: Grid coordinates
            cell_type: Type of cell to add

        Returns:
            Created cell or None if position occupied
        """
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            if self.grid[x, y, z] is not None:
                return None

            cell = Cell(x, y, z, cell_type)
            self.grid[x, y, z] = cell
            self.cells.append(cell)
            self.cell_counts[cell_type] += 1

            return cell
        return None

    def remove_cell(self, cell: Cell):
        """Remove cell from 3D grid and tracking."""
        if cell in self.cells:
            self.grid[cell.x, cell.y, cell.z] = None
            self.cells.remove(cell)
            self.cell_counts[cell.cell_type] -= 1

    def get_random_empty_position(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Find random empty position on 3D grid."""
        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            z = random.randint(0, self.depth - 1)
            if self.grid[x, y, z] is None:
                return x, y, z
            attempts += 1

        return None, None, None

    def get_neighbors(self, x: int, y: int, z: int, radius: int = 1) -> List[Cell]:
        """
        Get neighboring cells within radius in 3D.

        Args:
            x, y, z: Center coordinates
            radius: Search radius

        Returns:
            List of neighboring cells
        """
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and 0 <= nz < self.depth
                        and self.grid[nx, ny, nz] is not None
                    ):
                        neighbors.append(self.grid[nx, ny, nz])
        return neighbors

    def get_empty_neighbors(self, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """Get empty neighboring positions in 3D."""
        empty_positions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and 0 <= nz < self.depth
                        and self.grid[nx, ny, nz] is None
                    ):
                        empty_positions.append((nx, ny, nz))
        return empty_positions

    def update_oxygen_field(self):
        """Update 3D oxygen diffusion and consumption."""
        new_oxygen = self.oxygen_field.copy()

        # Reset to baseline from vascular supply
        vessel_supply = (
            self.vascular_network.astype(float) * self.params["oxygen_vascular_supply"]
        )
        new_oxygen = np.maximum(new_oxygen, vessel_supply)

        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    cell = self.grid[x, y, z]

                    # Oxygen consumption by cells
                    if cell and cell.cell_type == CellType.TUMOR:
                        consumption = self.params["oxygen_consumption"]
                        if (
                            hasattr(cell, "metabolic_state")
                            and cell.metabolic_state == "aerobic"
                        ):
                            consumption *= (
                                1.5  # Higher consumption in aerobic metabolism
                            )
                        new_oxygen[x, y, z] -= consumption
                    elif cell and cell.cell_type == CellType.IMMUNE:
                        new_oxygen[x, y, z] -= self.params["oxygen_consumption"] * 0.3

                    # 3D Diffusion using simple finite differences
                    diffusion_rate = self.params["oxygen_diffusion"]
                    diffusion = 0
                    neighbors = 0

                    for dx, dy, dz in [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (
                            0 <= nx < self.width
                            and 0 <= ny < self.height
                            and 0 <= nz < self.depth
                        ):
                            diffusion += self.oxygen_field[nx, ny, nz]
                            neighbors += 1

                    if neighbors > 0:
                        new_oxygen[x, y, z] += diffusion_rate * (
                            diffusion / neighbors - self.oxygen_field[x, y, z]
                        )

        # Clamp values
        self.oxygen_field = np.clip(new_oxygen, 0, 1.5)

    def update_drug_field(self, drug_input: float = 0.0):
        """Update 3D drug diffusion and decay."""
        new_drug = self.drug_field.copy()

        # Drug delivery from vessels
        vessel_delivery = (
            self.vascular_network.astype(float)
            * drug_input
            * self.params["drug_vascular_delivery"]
        )
        new_drug += vessel_delivery

        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    # Drug decay
                    new_drug[x, y, z] *= 1 - self.params["drug_decay_rate"]

                    # 3D Diffusion
                    diffusion_rate = self.params["drug_diffusion_rate"]
                    diffusion = 0
                    neighbors = 0

                    for dx, dy, dz in [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (
                            0 <= nx < self.width
                            and 0 <= ny < self.height
                            and 0 <= nz < self.depth
                        ):
                            diffusion += self.drug_field[nx, ny, nz]
                            neighbors += 1

                    if neighbors > 0:
                        new_drug[x, y, z] += diffusion_rate * (
                            diffusion / neighbors - self.drug_field[x, y, z]
                        )

        self.drug_field = np.clip(new_drug, 0, 10.0)

    def update_nutrient_field(self):
        """Update 3D nutrient distribution."""
        new_nutrients = self.nutrient_field.copy()

        # Nutrient supply from vessels
        vessel_supply = self.vascular_network.astype(float) * 0.8
        new_nutrients = np.maximum(new_nutrients, vessel_supply)

        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    cell = self.grid[x, y, z]

                    # Nutrient consumption
                    if cell:
                        consumption = self.params["nutrient_consumption"]
                        if cell.cell_type == CellType.TUMOR:
                            consumption *= 1.2  # Tumor cells consume more
                        new_nutrients[x, y, z] -= consumption

                    # Diffusion
                    diffusion_rate = self.params["nutrient_diffusion"]
                    diffusion = 0
                    neighbors = 0

                    for dx, dy, dz in [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (
                            0 <= nx < self.width
                            and 0 <= ny < self.height
                            and 0 <= nz < self.depth
                        ):
                            diffusion += self.nutrient_field[nx, ny, nz]
                            neighbors += 1

                    if neighbors > 0:
                        new_nutrients[x, y, z] += diffusion_rate * (
                            diffusion / neighbors - self.nutrient_field[x, y, z]
                        )

        self.nutrient_field = np.clip(new_nutrients, 0, 2.0)

    def apply_radiation(
        self, center: Tuple[int, int, int], radius: int = 20, dose: float = 5.0
    ):
        """
        Apply radiation beam to 3D area.

        Args:
            center: (x, y, z) beam center
            radius: Beam radius
            dose: Radiation dose
        """
        center_x, center_y, center_z = center

        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    distance = np.sqrt(
                        (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
                    )
                    if distance <= radius:
                        # Gaussian beam profile
                        beam_intensity = dose * np.exp(
                            -((distance / (radius / 3)) ** 2)
                        )
                        self.radiation_field[x, y, z] += beam_intensity

                        # Apply radiation damage to cells
                        cell = self.grid[x, y, z]
                        if cell and cell.cell_type == CellType.TUMOR:
                            # Linear-quadratic model
                            kill_prob = (
                                self.params["radiation_kill_prob"]
                                * beam_intensity
                                * (1 + 0.1 * beam_intensity)
                            )

                            # Oxygen enhancement
                            oer_factor = (3 * cell.oxygen_level + 1) / 4
                            kill_prob *= oer_factor

                            if random.random() < kill_prob:
                                cell.cell_type = CellType.DEAD
                                cell.health = 0

    def step_tumor_cell(self, cell: Cell):
        """Update tumor cell behavior with enhanced biological realism."""
        # Age the cell
        cell.age += 1
        cell.division_timer += 1

        # Update microenvironment awareness (3D)
        cell.oxygen_level = self.oxygen_field[cell.x, cell.y, cell.z]
        drug_level = self.drug_field[cell.x, cell.y, cell.z]

        # Update cell health based on microenvironment
        oxygen_stress = max(0, self.params["hypoxia_threshold"] - cell.oxygen_level)
        drug_damage = drug_level * (1 - cell.drug_resistance)

        # Health deterioration
        cell.health -= 0.005 * (oxygen_stress + drug_damage + cell.radiation_damage)
        cell.health = max(0, min(1, cell.health))  # Clamp between 0 and 1

        # Radiation damage repair
        cell.radiation_damage *= 1 - self.params["radiation_repair_rate"]

        # Enhanced death conditions
        death_prob = self.params["tumor_death_rate"]

        # Health-based death
        if cell.health <= 0:
            death_prob = 1.0
        elif cell.health < 0.3:
            death_prob += 0.05  # Increased death probability for damaged cells

        # Drug cytotoxicity with resistance
        drug_kill_prob = (
            self.params["drug_cytotoxicity"] * drug_level * (1 - cell.drug_resistance)
        )
        death_prob += drug_kill_prob

        # Severe hypoxia-induced death
        if cell.oxygen_level < 0.1:
            death_prob += 0.02
        elif cell.oxygen_level < self.params["hypoxia_threshold"]:
            death_prob += 0.005

        # Age-related death with senescence
        if cell.age > self.params["tumor_max_age"]:
            death_prob += 0.2
        elif cell.age > 0.8 * self.params["tumor_max_age"]:
            death_prob += 0.01  # Gradual senescence

        if random.random() < death_prob:
            cell.cell_type = CellType.DEAD
            cell.health = 0
            return

        # Enhanced cell division with growth factors
        can_divide = (
            cell.division_timer >= cell.division_time
            and cell.oxygen_level > self.params["hypoxia_threshold"]
            and cell.health > 0.6  # Need good health to divide
        )

        if can_divide:
            # Check local microenvironment
            neighbors = self.get_neighbors(cell.x, cell.y, cell.z)
            tumor_neighbors = sum(1 for n in neighbors if n.cell_type == CellType.TUMOR)

            # Enhanced local density control
            local_density = tumor_neighbors / 8.0  # Normalize by max neighbors

            # Growth factor calculation
            growth_stimulus = cell.oxygen_level * (1 - local_density)

            if (
                local_density < self.params["carrying_capacity_local"]
                and growth_stimulus > 0.3
            ):
                empty_positions = self.get_empty_neighbors(cell.x, cell.y, cell.z)
                if empty_positions:
                    # Division probability depends on conditions
                    division_prob = self.params["tumor_division_rate"] * growth_stimulus

                    if random.random() < division_prob:
                        new_x, new_y = random.choice(empty_positions)
                        new_cell = self.add_cell(new_x, new_y, CellType.TUMOR)
                        if new_cell:
                            # Inherit properties with evolutionary pressure
                            mutation_strength = 0.02 if drug_level > 0.1 else 0.005
                            new_cell.drug_resistance = np.clip(
                                cell.drug_resistance
                                + random.gauss(0, mutation_strength),
                                0,
                                1,
                            )
                            new_cell.health = (
                                0.7 + random.random() * 0.3
                            )  # New cells start healthy

                            # Division cost for parent
                            cell.health *= 0.9
                            cell.division_timer = 0

        # Enhanced migration with chemotaxis
        if random.random() < self.params["tumor_migration_rate"]:
            empty_positions = self.get_empty_neighbors(cell.x, cell.y, cell.z)
            if empty_positions:
                # Intelligent migration towards better conditions
                best_positions = []
                best_score = -999

                for new_x, new_y, new_z in empty_positions:
                    # Score based on oxygen availability and drug avoidance
                    oxygen_score = self.oxygen_field[new_x, new_y, new_z]
                    drug_penalty = self.drug_field[new_x, new_y, new_z] * (
                        1 - cell.drug_resistance
                    )

                    # Distance from tumor core (slight preference for expansion)
                    center_x, center_y, center_z = (
                        self.width // 2,
                        self.height // 2,
                        self.depth // 2,
                    )
                    distance_from_center = np.sqrt(
                        (new_x - center_x) ** 2
                        + (new_y - center_y) ** 2
                        + (new_z - center_z) ** 2
                    )
                    expansion_bonus = (
                        0.1
                        * distance_from_center
                        / max(self.width, self.height, self.depth)
                    )

                    total_score = oxygen_score - drug_penalty + expansion_bonus

                    if total_score > best_score:
                        best_score = total_score
                        best_positions = [(new_x, new_y, new_z)]
                    elif abs(total_score - best_score) < 0.05:  # Similar scores
                        best_positions.append((new_x, new_y, new_z))

                if best_positions:
                    new_x, new_y, new_z = random.choice(best_positions)
                    # Move cell
                    self.grid[cell.x, cell.y, cell.z] = None
                    cell.x, cell.y, cell.z = new_x, new_y, new_z
                    self.grid[new_x, new_y, new_z] = cell

    def step_immune_cell(self, cell: Cell):
        """Update immune cell behavior with enhanced immunological realism."""
        cell.age += 1

        # Enhanced death conditions
        death_prob = self.params["immune_death_rate"]

        # Age-related death with immune senescence
        if cell.age > self.params["immune_max_age"]:
            death_prob += 0.15
        elif cell.age > 0.7 * self.params["immune_max_age"]:
            death_prob += 0.005

        # Exhaustion from prolonged activation
        tumor_neighbors = [
            n
            for n in self.get_neighbors(cell.x, cell.y, cell.z, radius=2)
            if n.cell_type == CellType.TUMOR
        ]

        if len(tumor_neighbors) > 3:  # High tumor burden causes exhaustion
            death_prob += 0.01

        if random.random() < death_prob:
            self.remove_cell(cell)
            return

        # Enhanced cytotoxic activity
        immediate_tumor_neighbors = [
            n
            for n in self.get_neighbors(cell.x, cell.y, cell.z, radius=1)
            if n.cell_type == CellType.TUMOR
        ]

        if immediate_tumor_neighbors:
            # Attack tumor cells with variable efficiency
            for target in immediate_tumor_neighbors:
                # Cytotoxicity depends on target health and oxygen
                base_kill_prob = self.params["immune_cytotoxicity"]

                # More effective against damaged tumor cells
                damage_bonus = (1 - target.health) * 0.5

                # Less effective in hypoxic conditions
                oxygen_penalty = (
                    max(0, self.params["hypoxia_threshold"] - target.oxygen_level) * 0.5
                )

                effective_kill_prob = base_kill_prob + damage_bonus - oxygen_penalty
                effective_kill_prob = max(0, min(1, effective_kill_prob))

                if random.random() < effective_kill_prob:
                    target.cell_type = CellType.DEAD
                    target.health = 0

                    # Immune cell activation and memory formation
                    cell.health = min(
                        1.0, cell.health + 0.1
                    )  # Successful kill boosts immune cell

                    # Small probability of recruiting new immune cells
                    if random.random() < 0.05:
                        empty_nearby = self.get_empty_neighbors(cell.x, cell.y, cell.z)
                        if empty_nearby:
                            recruit_x, recruit_y, recruit_z = random.choice(
                                empty_nearby
                            )
                            self.add_cell(
                                recruit_x, recruit_y, recruit_z, CellType.IMMUNE
                            )

        # Enhanced migration with improved chemotaxis
        if random.random() < self.params["immune_migration_rate"]:
            empty_positions = self.get_empty_neighbors(cell.x, cell.y, cell.z)
            if empty_positions:
                # Advanced chemotaxis towards tumor cells
                best_positions = []
                max_tumor_signal = -1

                for new_x, new_y, new_z in empty_positions:
                    # Calculate tumor signal in surrounding area
                    tumor_signal = 0
                    search_radius = 3

                    for dx in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            for dz in range(-search_radius, search_radius + 1):
                                check_x, check_y, check_z = (
                                    new_x + dx,
                                    new_y + dy,
                                    new_z + dz,
                                )
                                if (
                                    0 <= check_x < self.width
                                    and 0 <= check_y < self.height
                                    and 0 <= check_z < self.depth
                                ):
                                    distance = max(
                                        1, np.sqrt(dx * dx + dy * dy + dz * dz)
                                    )
                                    cell_at_pos = self.grid[check_x, check_y, check_z]
                                    if (
                                        cell_at_pos
                                        and cell_at_pos.cell_type == CellType.TUMOR
                                    ):
                                        # Weight by distance and tumor health (more attracted to healthy tumors)
                                        tumor_signal += cell_at_pos.health / distance

                    if tumor_signal > max_tumor_signal:
                        max_tumor_signal = tumor_signal
                        best_positions = [(new_x, new_y, new_z)]
                    elif abs(tumor_signal - max_tumor_signal) < 0.1:
                        best_positions.append((new_x, new_y, new_z))

                if best_positions and max_tumor_signal > 0:
                    # Move towards tumor
                    new_x, new_y, new_z = random.choice(best_positions)
                    self.grid[cell.x, cell.y, cell.z] = None
                    cell.x, cell.y, cell.z = new_x, new_y, new_z
                    self.grid[new_x, new_y, new_z] = cell
                elif empty_positions:
                    # Random walk if no tumor signal
                    new_x, new_y, new_z = random.choice(empty_positions)
                    self.grid[cell.x, cell.y, cell.z] = None
                    cell.x, cell.y, cell.z = new_x, new_y, new_z
                    self.grid[new_x, new_y, new_z] = cell

                    for pos in empty_positions:
                        # This code block is now handled by enhanced immune migration above
                        pass

    def cleanup_dead_cells(self):
        """Remove dead cells from grid."""
        dead_cells = [cell for cell in self.cells if cell.cell_type == CellType.DEAD]
        for cell in dead_cells:
            self.remove_cell(cell)
            self.cell_counts[CellType.DEAD] += 1

    def step(self, dt: float = 1.0, therapy_input: Optional[Dict] = None):
        """
        Advance 3D simulation by one time step with enhanced dynamics.

        Args:
            dt: Time step (hours)
            therapy_input: Therapy inputs for this step
        """
        if therapy_input is None:
            therapy_input = {}

        self.time += dt

        # Update 3D fields
        self.update_oxygen_field()
        self.update_drug_field(therapy_input.get("drug_dose", 0.0))
        self.update_nutrient_field()

        # Apply 3D radiation if specified
        if "radiation" in therapy_input:
            rad_config = therapy_input["radiation"]
            center = rad_config.get(
                "center", (self.width // 2, self.height // 2, self.depth // 2)
            )
            self.apply_radiation_3d(
                center,
                rad_config.get("radius", 15),
                rad_config.get("dose", 5.0),
            )

        # Update cells (randomize order to avoid bias)
        cells_to_update = self.cells.copy()
        random.shuffle(cells_to_update)

        for cell in cells_to_update:
            if cell.cell_type == CellType.TUMOR:
                self.step_tumor_cell_3d(cell)
            elif cell.cell_type == CellType.IMMUNE:
                self.step_immune_cell_3d(cell)

        # Clean up dead cells with delay (allow for phagocytosis)
        self.cleanup_dead_cells_3d()

        # Enhanced immune recruitment based on tumor burden
        tumor_burden = self.cell_counts[CellType.TUMOR]
        recruitment_rate = self.params["immune_recruitment_rate"] * (
            1 + tumor_burden / 1000
        )

        if random.random() < recruitment_rate:
            position = self.get_random_empty_position()
            if position[0] is not None:
                self.add_cell(position[0], position[1], position[2], CellType.IMMUNE)

        # Update counts and enhanced metrics
        self.cell_counts = {CellType.TUMOR: 0, CellType.IMMUNE: 0, CellType.DEAD: 0}
        for cell in self.cells:
            self.cell_counts[cell.cell_type] += 1

        # Calculate enhanced metrics
        tumor_volume = (
            self.cell_counts[CellType.TUMOR] * 1e-9
        )  # Assuming each cell is ~1000 μm³
        invasion_depth = self.calculate_invasion_depth()
        avg_immune_activation = self.calculate_avg_immune_activation()
        metabolic_distribution = self.calculate_metabolic_distribution()

        # Record enhanced history
        self.history["time"].append(self.time)
        self.history["tumor_count"].append(self.cell_counts[CellType.TUMOR])
        self.history["immune_count"].append(self.cell_counts[CellType.IMMUNE])
        self.history["dead_count"].append(self.cell_counts[CellType.DEAD])
        self.history["avg_oxygen"].append(np.mean(self.oxygen_field))
        self.history["total_drug"].append(np.sum(self.drug_field))
        self.history["tumor_volume"].append(tumor_volume)
        self.history["invasion_depth"].append(invasion_depth)
        self.history["immune_activation"].append(avg_immune_activation)
        self.history["metabolic_states"].append(metabolic_distribution)

    def cleanup_dead_cells_3d(self):
        """Remove dead cells from 3D grid with phagocytosis delay."""
        cells_to_remove = []

        for cell in self.cells:
            if cell.cell_type == CellType.DEAD:
                cell.death_timer += 1
                # Remove after some time (phagocytosis/clearance)
                if cell.death_timer > 10:  # 10 time steps
                    cells_to_remove.append(cell)

        for cell in cells_to_remove:
            self.remove_cell(cell)

    def calculate_invasion_depth(self) -> float:
        """Calculate tumor invasion depth from initial center."""
        if not hasattr(self, "initial_center"):
            self.initial_center = (self.width // 2, self.height // 2, self.depth // 2)

        max_distance = 0
        center_x, center_y, center_z = self.initial_center

        for cell in self.cells:
            if cell.cell_type == CellType.TUMOR:
                distance = np.sqrt(
                    (cell.x - center_x) ** 2
                    + (cell.y - center_y) ** 2
                    + (cell.z - center_z) ** 2
                )
                max_distance = max(max_distance, distance)

        return max_distance

    def calculate_avg_immune_activation(self) -> float:
        """Calculate average immune cell activation level."""
        immune_cells = [c for c in self.cells if c.cell_type == CellType.IMMUNE]
        if not immune_cells:
            return 0.0

        total_activation = sum(c.activation_level for c in immune_cells)
        return total_activation / len(immune_cells)

    def calculate_metabolic_distribution(self) -> Dict[str, int]:
        """Calculate distribution of metabolic states."""
        states = {"aerobic": 0, "anaerobic": 0, "quiescent": 0}

        for cell in self.cells:
            if cell.cell_type == CellType.TUMOR:
                if hasattr(cell, "metabolic_state"):
                    states[cell.metabolic_state] = (
                        states.get(cell.metabolic_state, 0) + 1
                    )

        return states

    def get_grid_visualization(self, slice_z: int = None) -> np.ndarray:
        """
        Get colored grid for visualization.

        Args:
            slice_z: Z-slice to visualize (default: aggregated view of all slices)

        Returns:
            2D array with cell type codes
        """
        vis_grid = np.zeros((self.width, self.height), dtype=int)

        if slice_z is not None:
            # Show specific slice
            for x in range(self.width):
                for y in range(self.height):
                    cell = self.grid[x, y, slice_z]
                    if cell is None:
                        vis_grid[x, y] = 0  # Empty
                    elif cell.cell_type == CellType.TUMOR:
                        vis_grid[x, y] = 1  # Tumor
                    elif cell.cell_type == CellType.IMMUNE:
                        vis_grid[x, y] = 2  # Immune
                    elif cell.cell_type == CellType.DEAD:
                        vis_grid[x, y] = 3  # Dead
        else:
            # Aggregate view: project all cells onto 2D plane
            # Priority: Dead > Tumor > Immune > Empty
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        cell = self.grid[x, y, z]
                        if cell is not None:
                            current_value = vis_grid[x, y]
                            if cell.cell_type == CellType.DEAD:
                                vis_grid[x, y] = 3  # Dead has highest priority
                            elif cell.cell_type == CellType.TUMOR and current_value != 3:
                                vis_grid[x, y] = 1  # Tumor (if not dead)
                            elif cell.cell_type == CellType.IMMUNE and current_value == 0:
                                vis_grid[x, y] = 2  # Immune (if empty)

        return vis_grid

    def plot_state(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot current simulation state.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Cell grid
        colors = ["white", "red", "green", "gray"]  # Empty, Tumor, Immune, Dead
        cmap = mcolors.ListedColormap(colors)
        vis_grid = self.get_grid_visualization()

        axes[0, 0].imshow(vis_grid.T, cmap=cmap, vmin=0, vmax=3)
        axes[0, 0].set_title("Cell Distribution")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")

        # Oxygen field
        im2 = axes[0, 1].imshow(self.oxygen_field.T, cmap="Blues", vmin=0, vmax=1)
        axes[0, 1].set_title("Oxygen Level")
        plt.colorbar(im2, ax=axes[0, 1])

        # Drug field
        im3 = axes[0, 2].imshow(self.drug_field.T, cmap="Purples")
        axes[0, 2].set_title("Drug Concentration")
        plt.colorbar(im3, ax=axes[0, 2])

        # Population dynamics
        if len(self.history["time"]) > 1:
            axes[1, 0].plot(
                self.history["time"], self.history["tumor_count"], "r-", label="Tumor"
            )
            axes[1, 0].plot(
                self.history["time"], self.history["immune_count"], "g-", label="Immune"
            )
            axes[1, 0].set_xlabel("Time (hours)")
            axes[1, 0].set_ylabel("Cell Count")
            axes[1, 0].set_title("Population Dynamics")
            axes[1, 0].legend()

        # Oxygen dynamics
        if len(self.history["time"]) > 1:
            axes[1, 1].plot(self.history["time"], self.history["avg_oxygen"], "b-")
            axes[1, 1].set_xlabel("Time (hours)")
            axes[1, 1].set_ylabel("Average Oxygen")
            axes[1, 1].set_title("Oxygen Dynamics")

        # Drug dynamics
        if len(self.history["time"]) > 1:
            axes[1, 2].plot(self.history["time"], self.history["total_drug"], "m-")
            axes[1, 2].set_xlabel("Time (hours)")
            axes[1, 2].set_ylabel("Total Drug")
            axes[1, 2].set_title("Drug Dynamics")

        plt.tight_layout()
        return fig

    # === 3D Enhanced Methods ===

    def apply_radiation_3d(
        self, center: Tuple[int, int, int], radius: int = 15, dose: float = 5.0
    ):
        """Apply 3D radiation beam with realistic dose distribution."""
        center_x, center_y, center_z = center

        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    distance = np.sqrt(
                        (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
                    )
                    if distance <= radius:
                        # 3D Gaussian beam profile with depth attenuation
                        depth_attenuation = np.exp(
                            -abs(z - center_z) * 0.1
                        )  # Tissue attenuation
                        beam_intensity = (
                            dose
                            * np.exp(-((distance / (radius / 3)) ** 2))
                            * depth_attenuation
                        )
                        self.radiation_field[x, y, z] += beam_intensity

                        # Apply radiation damage to cells
                        cell = self.grid[x, y, z]
                        if cell:
                            # Enhanced linear-quadratic model
                            alpha = (
                                self.params["radiation_alpha"]
                                if hasattr(self.params, "radiation_alpha")
                                else 0.3
                            )
                            beta = 0.03

                            # Oxygen enhancement ratio
                            oer = (
                                self.params["radiation_oer_factor"]
                                if cell.oxygen_level > 0.1
                                else 1.0
                            )
                            oer_factor = (oer * cell.oxygen_level + 1) / (oer + 1)

                            kill_prob = 1 - np.exp(
                                -(alpha * beam_intensity + beta * beam_intensity**2)
                                * oer_factor
                            )

                            if random.random() < kill_prob:
                                cell.health = 0
                                cell.cell_type = CellType.DEAD
                                cell.death_timer = 0

    def step_tumor_cell_3d(self, cell: Cell):
        """Enhanced 3D tumor cell behavior with advanced mechanics."""
        cell.age += 1
        cell.division_timer += 1

        # Update microenvironment
        cell.oxygen_level = self.oxygen_field[cell.x, cell.y, cell.z]
        drug_level = self.drug_field[cell.x, cell.y, cell.z]
        nutrient_level = self.nutrient_field[cell.x, cell.y, cell.z]

        # Update metabolic state
        cell.update_metabolic_state()

        # Health calculation with multiple factors
        oxygen_stress = max(0, self.params["hypoxia_threshold"] - cell.oxygen_level)
        drug_damage = (
            drug_level * (1 - cell.drug_resistance) * self.params["drug_cytotoxicity"]
        )
        nutrient_stress = max(0, self.params["nutrient_threshold"] - nutrient_level)

        # Health deterioration
        health_loss = 0.002 * (
            oxygen_stress + drug_damage + nutrient_stress + cell.radiation_damage
        )
        cell.health -= health_loss
        cell.health = max(0, min(1, cell.health))

        # Death check
        if cell.health <= 0 or cell.age > self.params["tumor_max_age"]:
            cell.cell_type = CellType.DEAD
            cell.death_timer = 0
            return

        # Division logic with enhanced conditions
        if (
            cell.division_timer >= cell.division_time
            and cell.health > 0.7
            and cell.oxygen_level > 0.3
            and nutrient_level > 0.4
        ):
            empty_neighbors = self.get_empty_neighbors(cell.x, cell.y, cell.z)
            if empty_neighbors and random.random() < self.params["tumor_division_rate"]:
                # Choose division location considering space and gradients
                best_position = None
                best_score = -1

                for pos in empty_neighbors:
                    # Score based on oxygen and nutrient availability
                    score = (
                        self.oxygen_field[pos[0], pos[1], pos[2]]
                        + self.nutrient_field[pos[0], pos[1], pos[2]]
                    ) / 2
                    if score > best_score:
                        best_score = score
                        best_position = pos

                if best_position:
                    daughter_cell = self.add_cell(
                        best_position[0],
                        best_position[1],
                        best_position[2],
                        CellType.TUMOR,
                    )
                    if daughter_cell:
                        # Inherit properties with mutations
                        daughter_cell.drug_resistance = min(
                            1.0, cell.drug_resistance + random.uniform(-0.1, 0.1)
                        )
                        daughter_cell.invasiveness = min(
                            1.0, cell.invasiveness + random.uniform(-0.1, 0.1)
                        )
                        daughter_cell.stem_like = (
                            cell.stem_like and random.random() < 0.8
                        )  # Stem-like inheritance

                        # Reset division timer
                        cell.division_timer = 0
                        cell.health *= 0.8  # Division cost

        # Migration with enhanced 3D logic
        if random.random() < self.params["tumor_migration_rate"]:
            self.migrate_cell_3d(cell)

    def step_immune_cell_3d(self, cell: Cell):
        """Enhanced 3D immune cell behavior."""
        cell.age += 1

        # Update microenvironment
        cell.oxygen_level = self.oxygen_field[cell.x, cell.y, cell.z]

        # Find nearby tumor cells for activation
        tumor_neighbors = [
            c
            for c in self.get_neighbors(cell.x, cell.y, cell.z, radius=2)
            if c.cell_type == CellType.TUMOR
        ]

        # Activation based on tumor density
        tumor_density = len(tumor_neighbors) / 26  # Max neighbors in 3D
        cell.activation_level = min(1.0, cell.activation_level + tumor_density * 0.1)

        # Cytotoxicity against neighboring tumor cells
        for tumor_cell in tumor_neighbors:
            if tumor_cell.distance_to(cell) <= 1.5:  # Close contact
                kill_prob = (
                    cell.cytotoxicity
                    * cell.activation_level
                    * self.params["immune_cytotoxicity"]
                )
                if random.random() < kill_prob:
                    tumor_cell.health -= 0.3
                    if tumor_cell.health <= 0:
                        tumor_cell.cell_type = CellType.DEAD

        # Migration toward tumor cells (chemotaxis)
        if random.random() < self.params["immune_migration_rate"]:
            self.migrate_immune_cell_3d(cell)

        # Natural death
        if (
            cell.age > self.params["immune_max_age"]
            or random.random() < self.params["immune_death_rate"]
        ):
            cell.cell_type = CellType.DEAD

    def migrate_cell_3d(self, cell: Cell):
        """3D cell migration with bias and constraints."""
        empty_neighbors = self.get_empty_neighbors(cell.x, cell.y, cell.z)
        if not empty_neighbors:
            return

        # Tumor cells: migrate toward better conditions or invasion
        if cell.cell_type == CellType.TUMOR:
            best_position = None
            best_score = -1

            for pos in empty_neighbors:
                # Score based on oxygen, nutrients, and ECM density
                oxygen_score = self.oxygen_field[pos[0], pos[1], pos[2]]
                nutrient_score = self.nutrient_field[pos[0], pos[1], pos[2]]
                ecm_resistance = self.ecm_density[pos[0], pos[1], pos[2]]

                # Invasive cells can better penetrate ECM
                invasion_bonus = cell.invasiveness * (1 - ecm_resistance)

                score = (oxygen_score + nutrient_score) / 2 + invasion_bonus

                # Z-direction penalty (harder to migrate vertically)
                if abs(pos[2] - cell.z) > 0:
                    score *= self.params["z_migration_penalty"]

                if score > best_score:
                    best_score = score
                    best_position = pos

            if (
                best_position and random.random() < 0.7
            ):  # 70% chance to migrate to best position
                self.move_cell(cell, best_position)

        else:  # Random migration for other cell types
            new_position = random.choice(empty_neighbors)
            if random.random() < 0.5:
                self.move_cell(cell, new_position)

    def migrate_immune_cell_3d(self, cell: Cell):
        """3D immune cell migration with chemotaxis."""
        empty_neighbors = self.get_empty_neighbors(cell.x, cell.y, cell.z)
        if not empty_neighbors:
            return

        # Find direction toward highest tumor density
        best_position = None
        best_tumor_score = -1

        for pos in empty_neighbors:
            # Count tumor cells in neighborhood of potential position
            tumor_count = 0
            for nx in range(max(0, pos[0] - 2), min(self.width, pos[0] + 3)):
                for ny in range(max(0, pos[1] - 2), min(self.height, pos[1] + 3)):
                    for nz in range(max(0, pos[2] - 2), min(self.depth, pos[2] + 3)):
                        neighbor_cell = self.grid[nx, ny, nz]
                        if neighbor_cell and neighbor_cell.cell_type == CellType.TUMOR:
                            tumor_count += 1

            if tumor_count > best_tumor_score:
                best_tumor_score = tumor_count
                best_position = pos

        if best_position:
            self.move_cell(cell, best_position)

    def move_cell(self, cell: Cell, new_position: Tuple[int, int, int]):
        """Move cell to new 3D position."""
        old_x, old_y, old_z = cell.x, cell.y, cell.z
        new_x, new_y, new_z = new_position

        # Update grid
        self.grid[old_x, old_y, old_z] = None
        self.grid[new_x, new_y, new_z] = cell

        # Update cell position
        cell.x, cell.y, cell.z = new_x, new_y, new_z

    def plot_3d_state(
        self, figsize: Tuple[int, int] = (16, 12), slice_z: int = None
    ) -> plt.Figure:
        """Create comprehensive 3D visualization."""
        if slice_z is None:
            slice_z = self.depth // 2  # Middle slice

        fig = plt.figure(figsize=figsize)

        # Create 3D scatter plot
        ax1 = fig.add_subplot(221, projection="3d")

        tumor_positions = []
        immune_positions = []
        dead_positions = []

        for cell in self.cells:
            if cell.cell_type == CellType.TUMOR:
                tumor_positions.append([cell.x, cell.y, cell.z])
            elif cell.cell_type == CellType.IMMUNE:
                immune_positions.append([cell.x, cell.y, cell.z])
            elif cell.cell_type == CellType.DEAD:
                dead_positions.append([cell.x, cell.y, cell.z])

        if tumor_positions:
            tumor_pos = np.array(tumor_positions)
            ax1.scatter(
                tumor_pos[:, 0],
                tumor_pos[:, 1],
                tumor_pos[:, 2],
                c="red",
                s=20,
                alpha=0.7,
                label="Tumor",
            )

        if immune_positions:
            immune_pos = np.array(immune_positions)
            ax1.scatter(
                immune_pos[:, 0],
                immune_pos[:, 1],
                immune_pos[:, 2],
                c="green",
                s=15,
                alpha=0.7,
                label="Immune",
            )

        if dead_positions:
            dead_pos = np.array(dead_positions)
            ax1.scatter(
                dead_pos[:, 0],
                dead_pos[:, 1],
                dead_pos[:, 2],
                c="black",
                s=10,
                alpha=0.3,
                label="Dead",
            )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("3D Cell Distribution")
        ax1.legend()

        # 2D slices
        ax2 = fig.add_subplot(222)
        oxygen_slice = self.oxygen_field[:, :, slice_z]
        im2 = ax2.imshow(oxygen_slice.T, origin="lower", cmap="viridis")
        ax2.set_title(f"Oxygen Field (z={slice_z})")
        plt.colorbar(im2, ax=ax2)

        ax3 = fig.add_subplot(223)
        drug_slice = self.drug_field[:, :, slice_z]
        im3 = ax3.imshow(drug_slice.T, origin="lower", cmap="plasma")
        ax3.set_title(f"Drug Field (z={slice_z})")
        plt.colorbar(im3, ax=ax3)

        # Population dynamics
        ax4 = fig.add_subplot(224)
        if len(self.history["time"]) > 1:
            ax4.plot(
                self.history["time"], self.history["tumor_count"], "r-", label="Tumor"
            )
            ax4.plot(
                self.history["time"], self.history["immune_count"], "g-", label="Immune"
            )
            ax4.plot(
                self.history["time"], self.history["dead_count"], "k-", label="Dead"
            )
            ax4.set_xlabel("Time (hours)")
            ax4.set_ylabel("Cell Count")
            ax4.set_title("Population Dynamics")
            ax4.legend()

        plt.tight_layout()
        return fig

    def create_interactive_3d_plot(self):
        """Create interactive 3D plot using plotly."""
        tumor_positions = []
        immune_positions = []
        dead_positions = []

        for cell in self.cells:
            if cell.cell_type == CellType.TUMOR:
                tumor_positions.append([cell.x, cell.y, cell.z])
            elif cell.cell_type == CellType.IMMUNE:
                immune_positions.append([cell.x, cell.y, cell.z])
            elif cell.cell_type == CellType.DEAD:
                dead_positions.append([cell.x, cell.y, cell.z])

        fig = go.Figure()

        if tumor_positions:
            tumor_pos = np.array(tumor_positions)
            fig.add_trace(
                go.Scatter3d(
                    x=tumor_pos[:, 0],
                    y=tumor_pos[:, 1],
                    z=tumor_pos[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5, opacity=0.8),
                    name="Tumor Cells",
                )
            )

        if immune_positions:
            immune_pos = np.array(immune_positions)
            fig.add_trace(
                go.Scatter3d(
                    x=immune_pos[:, 0],
                    y=immune_pos[:, 1],
                    z=immune_pos[:, 2],
                    mode="markers",
                    marker=dict(color="green", size=4, opacity=0.8),
                    name="Immune Cells",
                )
            )

        if dead_positions:
            dead_pos = np.array(dead_positions)
            fig.add_trace(
                go.Scatter3d(
                    x=dead_pos[:, 0],
                    y=dead_pos[:, 1],
                    z=dead_pos[:, 2],
                    mode="markers",
                    marker=dict(color="black", size=3, opacity=0.4),
                    name="Dead Cells",
                )
            )

        fig.update_layout(
            title="3D Cancer Treatment Simulation",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
            ),
            width=800,
            height=600,
        )

        return fig


# For backward compatibility, keep the original class name as an alias
TumorImmuneABM = TumorImmuneABM3D
