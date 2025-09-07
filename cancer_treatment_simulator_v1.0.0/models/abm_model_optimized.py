"""
Optimized Agent-Based Model for Cell-Level Cancer Dynamics (3D Enhanced)

Performance optimizations:
- Vectorized operation        # Cell counts
        self.cell_counts = {CellType.TUMOR.value: 0, CellType.IMMUNE.value: 0, CellType.DEAD.value: 0}
        
        # Enhanced history tracking - compatible with original model
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
        
        # Performance trackingr field updates
- Spatial indexing for neighbor searches
- Batch processing for cell updates
- Reduced memory allocations
- Optimized data structures

These optimizations can significantly reduce simulation time while maintaining 
biological accuracy.
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
from scipy.spatial import cKDTree
import numba
from numba import jit, njit


class CellType(Enum):
    """Cell type enumeration."""
    TUMOR = "tumor"
    IMMUNE = "immune"
    DEAD = "dead"
    EMPTY = "empty"


@dataclass
class CellOptimized:
    """Optimized cell agent with vectorized properties."""
    x: int
    y: int
    z: int
    cell_type: CellType
    age: float = 0.0
    health: float = 1.0
    division_timer: float = 0.0
    drug_resistance: float = 0.0
    radiation_damage: float = 0.0
    oxygen_level: float = 1.0
    death_timer: float = 0.0
    division_time: float = 20.0
    last_division: float = 0.0


@njit
def update_cell_health_vectorized(health, oxygen, drug, nutrient, radiation, resistance, params):
    """Vectorized health update for multiple cells."""
    oxygen_stress = np.maximum(0, params[0] - oxygen)  # hypoxia_threshold
    drug_damage = drug * (1 - resistance) * params[1]  # drug_cytotoxicity
    nutrient_stress = np.maximum(0, params[2] - nutrient)  # nutrient_threshold
    
    health_loss = 0.002 * (oxygen_stress + drug_damage + nutrient_stress + radiation)
    new_health = np.maximum(0, np.minimum(1, health - health_loss))
    return new_health


@njit
def check_division_conditions_vectorized(health, oxygen, nutrient, division_timer, division_time):
    """Vectorized division condition checking."""
    can_divide = (
        (division_timer >= division_time) &
        (health > 0.7) &
        (oxygen > 0.3) &
        (nutrient > 0.4)
    )
    return can_divide


class TumorImmuneABM3DOptimized:
    """Optimized 3D Agent-Based Model for tumor-immune dynamics."""

    def __init__(self, grid_size=(100, 100, 50)):
        """Initialize optimized 3D ABM with performance enhancements."""
        self.width, self.height, self.depth = grid_size
        self.time = 0.0
        
        # Optimized data structures
        self.cells = []
        self.cell_positions = {}  # For fast spatial lookups
        self.spatial_tree = None  # For neighbor searches
        
        # Vectorized fields - pre-allocate
        self.oxygen_field = np.ones(grid_size, dtype=np.float32)
        self.drug_field = np.zeros(grid_size, dtype=np.float32)
        self.nutrient_field = np.ones(grid_size, dtype=np.float32)
        
        # Pre-computed arrays for vectorized operations
        self.cell_arrays = {
            'x': np.array([], dtype=np.int32),
            'y': np.array([], dtype=np.int32),
            'z': np.array([], dtype=np.int32),
            'health': np.array([], dtype=np.float32),
            'age': np.array([], dtype=np.float32),
            'oxygen': np.array([], dtype=np.float32),
            'division_timer': np.array([], dtype=np.float32),
            'drug_resistance': np.array([], dtype=np.float32),
            'radiation_damage': np.array([], dtype=np.float32)
        }
        
        # Performance parameters
        self.params = {
            "tumor_growth_rate": 0.05,
            "immune_kill_rate": 0.001,  # Reduced to very mild default - GUI will override this
            "drug_cytotoxicity": 0.8,
            "radiation_sensitivity": 0.3,
            "hypoxia_threshold": 0.2,
            "nutrient_threshold": 0.3,
            "immune_recruitment_rate": 0.01,
            "tumor_max_age": 100.0,
            "immune_max_age": 50.0,
            "mutation_rate": 0.001,
            "angiogenesis_threshold": 0.15,
            "vessel_formation_rate": 0.02,
            "necrosis_threshold": 0.05,
            "immune_migration_speed": 2.0,
        }
        
        # Cell counts
        self.cell_counts = {CellType.TUMOR.value: 0, CellType.IMMUNE.value: 0, CellType.DEAD.value: 0}
        
        # Performance tracking
        self.update_spatial_index_interval = 10  # Update every N steps
        self.step_count = 0
        
        # Batch sizes for processing
        self.batch_size = 1000
        
        # Initialize history for tracking simulation progress
        self.history = {
            'time': [],
            'tumor_count': [],
            'immune_count': [],
            'dead_count': [],
            'avg_oxygen': [],
            'total_drug': [],
            'tumor_volume': [],
            'invasion_depth': [],
            'immune_activation': [],
            'metabolic_states': [],
            'drug_concentration': [],
            'oxygen_levels': [],
            'treatment_responses': [],
            'spatial_distributions': [],
            'immune_effectiveness': []
        }

    def rebuild_cell_arrays(self):
        """Rebuild vectorized arrays from cell list for batch operations."""
        if not self.cells:
            return
            
        n_cells = len(self.cells)
        self.cell_arrays['x'] = np.array([c.x for c in self.cells], dtype=np.int32)
        self.cell_arrays['y'] = np.array([c.y for c in self.cells], dtype=np.int32)
        self.cell_arrays['z'] = np.array([c.z for c in self.cells], dtype=np.int32)
        self.cell_arrays['health'] = np.array([c.health for c in self.cells], dtype=np.float32)
        self.cell_arrays['age'] = np.array([c.age for c in self.cells], dtype=np.float32)
        self.cell_arrays['oxygen'] = np.array([c.oxygen_level for c in self.cells], dtype=np.float32)
        self.cell_arrays['division_timer'] = np.array([c.division_timer for c in self.cells], dtype=np.float32)
        self.cell_arrays['drug_resistance'] = np.array([c.drug_resistance for c in self.cells], dtype=np.float32)
        self.cell_arrays['radiation_damage'] = np.array([c.radiation_damage for c in self.cells], dtype=np.float32)

    def update_spatial_index(self):
        """Update spatial index for fast neighbor searches."""
        if not self.cells:
            return
            
        positions = np.array([[c.x, c.y, c.z] for c in self.cells])
        self.spatial_tree = cKDTree(positions)
        
        # Rebuild position dictionary to ensure consistency
        self.cell_positions.clear()
        for i, cell in enumerate(self.cells):
            self.cell_positions[(cell.x, cell.y, cell.z)] = i

    def update_cell_counts(self):
        """Update cell counts dictionary based on current cells."""
        # Reset counts
        self.cell_counts = {CellType.TUMOR.value: 0, CellType.IMMUNE.value: 0, CellType.DEAD.value: 0}
        
        # Count current cells
        for cell in self.cells:
            self.cell_counts[cell.cell_type.value] += 1

    def initialize_tumor_optimized(self, center=(50, 50, 25), radius=10):
        """Optimized tumor initialization with vectorized operations."""
        cx, cy, cz = center
        
        # Pre-calculate all positions within sphere
        positions = []
        for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for z in range(max(0, cz - radius), min(self.depth, cz + radius + 1)):
                    distance = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
                    if distance <= radius:
                        positions.append((x, y, z))
        
        # Batch create cells
        for x, y, z in positions:
            cell = CellOptimized(
                x=x, y=y, z=z,
                cell_type=CellType.TUMOR,
                division_time=np.random.exponential(20.0),
                drug_resistance=np.random.uniform(0.0, 0.1)
            )
            self.cells.append(cell)
        
        self.rebuild_cell_arrays()
        self.update_spatial_index()
        self.update_cell_counts()  # Update cell counts after tumor initialization

    def initialize_immune_cells_optimized(self, count=50):
        """Optimized immune cell initialization."""
        positions = []
        attempts = 0
        max_attempts = count * 10
        
        while len(positions) < count and attempts < max_attempts:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            z = np.random.randint(0, self.depth)
            
            if (x, y, z) not in self.cell_positions:
                positions.append((x, y, z))
            attempts += 1
        
        # Batch create immune cells
        for x, y, z in positions:
            cell = CellOptimized(
                x=x, y=y, z=z,
                cell_type=CellType.IMMUNE,
                health=np.random.uniform(0.8, 1.0)
            )
            self.cells.append(cell)
        
        self.rebuild_cell_arrays()
        self.update_spatial_index()
        self.update_cell_counts()  # Update cell counts after immune cell initialization

    def update_fields_vectorized(self, drug_dose=0.0):
        """Vectorized field updates for better performance."""
        # Oxygen field - vectorized diffusion with replenishment
        # First, add slight replenishment everywhere
        self.oxygen_field = np.minimum(1.0, self.oxygen_field + 0.01)
        
        if len(self.cells) > 0:
            tumor_positions = [(c.x, c.y, c.z) for c in self.cells if c.cell_type == CellType.TUMOR]
            if tumor_positions:
                tumor_array = np.array(tumor_positions)
                # Create consumption mask
                consumption_mask = np.ones_like(self.oxygen_field)
                for x, y, z in tumor_positions:
                    # Vectorized oxygen consumption
                    xx, yy, zz = np.mgrid[max(0, x-2):min(self.width, x+3),
                                         max(0, y-2):min(self.height, y+3),
                                         max(0, z-2):min(self.depth, z+3)]
                    dist = np.sqrt((xx-x)**2 + (yy-y)**2 + (zz-z)**2)
                    mask = dist <= 2
                    consumption_mask[xx[mask], yy[mask], zz[mask]] *= 0.98  # Less aggressive consumption
                
                self.oxygen_field *= consumption_mask
        
        # Drug field - faster diffusion
        if drug_dose > 0:
            self.drug_field = gaussian_filter(
                self.drug_field + drug_dose * 0.1, 
                sigma=1.0, 
                mode='constant'
            )
        else:
            self.drug_field *= 0.95  # Decay
        
        # Nutrient field - simplified update
        self.nutrient_field = np.clip(
            gaussian_filter(self.nutrient_field, sigma=0.5) + 0.01,
            0, 1
        )

    def step_optimized(self, dt: float = 1.0, therapy_input: Optional[Dict] = None):
        """Optimized simulation step with vectorized operations."""
        if therapy_input is None:
            therapy_input = {}
            
        self.time += dt
        self.step_count += 1
        
        # Update fields (less frequently for performance)
        if self.step_count % 5 == 0:  # Every 5 steps
            self.update_fields_vectorized(therapy_input.get("drug_dose", 0.0))
        
        # Update spatial index less frequently
        if self.step_count % self.update_spatial_index_interval == 0:
            self.update_spatial_index()
        
        # Apply radiation if specified
        if "radiation" in therapy_input:
            rad_config = therapy_input["radiation"]
            center = rad_config.get("center", (self.width // 2, self.height // 2, self.depth // 2))
            self.apply_radiation_optimized(
                center,
                rad_config.get("radius", 15),
                rad_config.get("dose", 5.0)
            )
        
        # Apply immunotherapy boost if specified
        if "immune_boost" in therapy_input:
            boost_factor = therapy_input["immune_boost"]
            # Increase immune recruitment and activity
            recruitment_rate = self.params["immune_recruitment_rate"] * boost_factor
            if np.random.random() < recruitment_rate:
                position = self.get_random_empty_position()
                if position[0] is not None:
                    new_cell = CellOptimized(
                        x=position[0], y=position[1], z=position[2],
                        cell_type=CellType.IMMUNE,
                        health=np.random.uniform(0.8, 1.0)
                    )
                    self.cells.append(new_cell)
        
        # Batch process cells
        self.process_cells_batch()
        
        # Update counts - use string keys for compatibility
        self.cell_counts = {CellType.TUMOR.value: 0, CellType.IMMUNE.value: 0, CellType.DEAD.value: 0}
        for cell in self.cells:
            self.cell_counts[cell.cell_type.value] += 1
        
        # Update history for compatibility with original model
        self.history["time"].append(self.time)
        self.history["tumor_count"].append(self.cell_counts.get('tumor', 0))
        self.history["immune_count"].append(self.cell_counts.get('immune', 0))
        self.history["dead_count"].append(self.cell_counts.get('dead', 0))
        self.history["avg_oxygen"].append(np.mean(self.oxygen_field))
        self.history["total_drug"].append(np.sum(self.drug_field))
        self.history["tumor_volume"].append(self.cell_counts.get('tumor', 0))  # Simplified
        self.history["invasion_depth"].append(0)  # Simplified
        self.history["immune_activation"].append(0)  # Simplified
        self.history["metabolic_states"].append([])

    def apply_radiation_optimized(self, center, radius, dose):
        """Optimized radiation application using vectorized operations."""
        cx, cy, cz = center
        
        if not self.cells:
            return
        
        # Vectorized distance calculation
        positions = np.array([[c.x, c.y, c.z] for c in self.cells])
        distances = np.linalg.norm(positions - np.array(center), axis=1)
        
        # Apply radiation to cells within radius
        affected_indices = np.where(distances <= radius)[0]
        
        for idx in affected_indices:
            cell = self.cells[idx]
            if cell.cell_type == CellType.TUMOR:
                cell.radiation_damage += dose * self.params["radiation_sensitivity"] * (1 - cell.drug_resistance)

    def step(self, dt: float = 1.0, therapy_input: Optional[Dict] = None):
        """Compatibility method that calls step_optimized."""
        return self.step_optimized(dt, therapy_input)

    def process_cells_batch(self):
        """Process cells in batches for better performance."""
        if not self.cells:
            return
        
        # Separate cells by type for batch processing
        tumor_cells = [i for i, c in enumerate(self.cells) if c.cell_type == CellType.TUMOR]
        immune_cells = [i for i, c in enumerate(self.cells) if c.cell_type == CellType.IMMUNE]
        
        # Process tumor cells in batches
        for i in range(0, len(tumor_cells), self.batch_size):
            batch_indices = tumor_cells[i:i + self.batch_size]
            self.process_tumor_batch(batch_indices)
        
        # Process immune cells in batches
        for i in range(0, len(immune_cells), self.batch_size):
            batch_indices = immune_cells[i:i + self.batch_size]
            self.process_immune_batch(batch_indices)
        
        # Remove dead cells efficiently
        self.cleanup_dead_cells_optimized()

    def process_tumor_batch(self, indices):
        """Process a batch of tumor cells."""
        for idx in indices:
            cell = self.cells[idx]
            cell.age += 1
            cell.division_timer += 1
            
            # Update microenvironment (batch this if possible)
            cell.oxygen_level = self.oxygen_field[cell.x, cell.y, cell.z]
            drug_level = self.drug_field[cell.x, cell.y, cell.z]
            nutrient_level = self.nutrient_field[cell.x, cell.y, cell.z]
            
            # Health calculation
            oxygen_stress = max(0, self.params["hypoxia_threshold"] - cell.oxygen_level)
            drug_damage = drug_level * (1 - cell.drug_resistance) * self.params["drug_cytotoxicity"]
            nutrient_stress = max(0, self.params["nutrient_threshold"] - nutrient_level)
            
            health_loss = 0.001 * (oxygen_stress + drug_damage + nutrient_stress + cell.radiation_damage)  # Reduced from 0.002
            cell.health = max(0, min(1, cell.health - health_loss))
            
            # Death check
            if cell.health <= 0 or cell.age > self.params["tumor_max_age"]:
                cell.cell_type = CellType.DEAD
                cell.death_timer = 0
                continue
            
            # Division check (simplified)
            if (cell.division_timer >= cell.division_time and 
                cell.health > 0.7 and 
                cell.oxygen_level > 0.3 and 
                nutrient_level > 0.4):
                
                # Find nearby empty position (optimized)
                new_pos = self.find_empty_neighbor_fast(cell.x, cell.y, cell.z)
                if new_pos[0] is not None:
                    new_cell = CellOptimized(
                        x=new_pos[0], y=new_pos[1], z=new_pos[2],
                        cell_type=CellType.TUMOR,
                        division_time=np.random.exponential(20.0),
                        drug_resistance=min(1.0, cell.drug_resistance + np.random.normal(0, 0.01))
                    )
                    self.cells.append(new_cell)
                    cell.division_timer = 0

    def process_immune_batch(self, indices):
        """Process a batch of immune cells."""
        for idx in indices:
            cell = self.cells[idx]
            cell.age += 1
            
            # Simplified immune behavior for performance
            cell.oxygen_level = self.oxygen_field[cell.x, cell.y, cell.z]
            
            # Death check
            if cell.age > self.params["immune_max_age"] or cell.health <= 0:
                cell.cell_type = CellType.DEAD
                continue
            
            # Immune-tumor interaction: attack nearby tumor cells
            # Simple proximity-based interaction with configurable rate
            if np.random.random() < self.params["immune_kill_rate"]:
                # Find nearby tumor cells within interaction radius
                interaction_radius = 2  # cells can interact within 2 grid units
                
                for other_idx, other_cell in enumerate(self.cells):
                    if (other_cell.cell_type == CellType.TUMOR and 
                        abs(other_cell.x - cell.x) <= interaction_radius and
                        abs(other_cell.y - cell.y) <= interaction_radius and
                        abs(other_cell.z - cell.z) <= interaction_radius):
                        
                        # Calculate actual distance
                        distance = np.sqrt((other_cell.x - cell.x)**2 + 
                                         (other_cell.y - cell.y)**2 + 
                                         (other_cell.z - cell.z)**2)
                        
                        if distance <= interaction_radius:
                            # Damage the tumor cell based on proximity and immune effectiveness
                            damage = 0.1 * (1 - distance / interaction_radius)  # Closer = more damage
                            other_cell.health -= damage
                            
                            # Break after attacking one cell (realistic immune behavior)
                            break

    def find_empty_neighbor_fast(self, x, y, z):
        """Fast empty neighbor finding using pre-computed positions."""
        # Check immediate neighbors first
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth and
                        (nx, ny, nz) not in self.cell_positions):
                        return (nx, ny, nz)
        
        return (None, None, None)

    def immune_migration_optimized(self, immune_cell):
        """Optimized immune cell migration using spatial index."""
        if not self.spatial_tree:
            return
        
        # Find nearest tumor cells
        immune_pos = np.array([immune_cell.x, immune_cell.y, immune_cell.z])
        distances, indices = self.spatial_tree.query(immune_pos, k=min(5, len(self.cells)))
        
        # Move towards nearest tumor
        for idx in indices:
            if idx < len(self.cells) and self.cells[idx].cell_type == CellType.TUMOR:
                target = self.cells[idx]
                # Simple movement (can be optimized further)
                dx = np.sign(target.x - immune_cell.x)
                dy = np.sign(target.y - immune_cell.y)
                dz = np.sign(target.z - immune_cell.z)
                
                new_x = np.clip(immune_cell.x + dx, 0, self.width - 1)
                new_y = np.clip(immune_cell.y + dy, 0, self.height - 1)
                new_z = np.clip(immune_cell.z + dz, 0, self.depth - 1)
                
                if (new_x, new_y, new_z) not in self.cell_positions:
                    # Safely remove old position
                    old_pos = (immune_cell.x, immune_cell.y, immune_cell.z)
                    if old_pos in self.cell_positions:
                        del self.cell_positions[old_pos]
                    
                    # Update cell position
                    immune_cell.x, immune_cell.y, immune_cell.z = new_x, new_y, new_z
                    
                    # Add new position (find correct cell index)
                    for i, c in enumerate(self.cells):
                        if c is immune_cell:
                            self.cell_positions[(new_x, new_y, new_z)] = i
                            break
                break

    def cleanup_dead_cells_optimized(self):
        """Optimized dead cell cleanup."""
        # Remove dead cells that have been dead long enough
        alive_cells = []
        for cell in self.cells:
            if cell.cell_type == CellType.DEAD:
                cell.death_timer += 1
                if cell.death_timer < 5:  # Keep for a few steps
                    alive_cells.append(cell)
                else:
                    # Remove from position tracking
                    pos = (cell.x, cell.y, cell.z)
                    if pos in self.cell_positions:
                        del self.cell_positions[pos]
            else:
                alive_cells.append(cell)
        
        self.cells = alive_cells

    def get_random_empty_position(self):
        """Find a random empty position in the grid."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            z = np.random.randint(0, self.depth)
            
            if (x, y, z) not in self.cell_positions:
                return (x, y, z)
        
        # If no empty position found, return None
        return (None, None, None)

    def get_state(self):
        """Get current simulation state - compatible with original interface."""
        state = {
            'time': self.time,
            'cell_counts': self.cell_counts.copy(),
            'total_cells': len(self.cells),
            'tumor_volume': self.cell_counts.get('tumor', 0),
            'immune_count': self.cell_counts.get('immune', 0),
            'dead_count': self.cell_counts.get('dead', 0)
        }
        
        if self.cells:
            # Spatial distribution
            positions = np.array([[c.x, c.y, c.z] for c in self.cells])
            state['spatial_data'] = {
                'positions': positions,
                'cell_types': [c.cell_type.value for c in self.cells],
                'health_values': [c.health for c in self.cells]
            }
        
        return state

    def get_grid_visualization(self, slice_z: int = None) -> np.ndarray:
        """
        Get colored grid for visualization - returns 3D grid for compatibility.

        Args:
            slice_z: Z-slice to visualize (default: full 3D grid)

        Returns:
            3D array with cell type codes (width, height, depth)
        """
        if slice_z is not None:
            # 🔧 FIX: Add bounds checking for slice_z
            if slice_z < 0 or slice_z >= self.depth:
                print(f"WARNING: slice_z={slice_z} is out of bounds [0, {self.depth-1}]. Using middle slice.")
                slice_z = self.depth // 2
            
            # Return 2D slice as 3D array with depth=1
            vis_grid = np.zeros((self.width, self.height, 1), dtype=int)
            for cell in self.cells:
                if cell.z == slice_z:  # Only cells in the specified slice
                    x, y = cell.x, cell.y
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if cell.cell_type == CellType.TUMOR:
                            vis_grid[x, y, 0] = 1  # Tumor
                        elif cell.cell_type == CellType.IMMUNE:
                            vis_grid[x, y, 0] = 2  # Immune
                        elif cell.cell_type == CellType.DEAD:
                            vis_grid[x, y, 0] = 3  # Dead
            return vis_grid
        else:
            # Return full 3D grid
            vis_grid = np.zeros((self.width, self.height, self.depth), dtype=int)
            for cell in self.cells:
                x, y, z = cell.x, cell.y, cell.z
                if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
                    if cell.cell_type == CellType.TUMOR:
                        vis_grid[x, y, z] = 1  # Tumor
                    elif cell.cell_type == CellType.IMMUNE:
                        vis_grid[x, y, z] = 2  # Immune
                    elif cell.cell_type == CellType.DEAD:
                        vis_grid[x, y, z] = 3  # Dead
            return vis_grid

    def run_simulation_optimized(self, time_points, therapy_schedule=None):
        """Run optimized simulation for multiple time points."""
        results = []
        
        for t in time_points:
            while self.time < t:
                therapy_input = {}
                if therapy_schedule:
                    # Handle both dictionary and list formats for therapy_schedule
                    if isinstance(therapy_schedule, dict):
                        # Convert dict format to therapy inputs
                        for therapy_type, therapy_config in therapy_schedule.items():
                            if (therapy_config.get('start_time', 0) <= self.time <= 
                                therapy_config.get('end_time', float('inf'))):
                                
                                if therapy_type == 'radiotherapy':
                                    therapy_input['radiation'] = {
                                        'center': (self.width // 2, self.height // 2, self.depth // 2),
                                        'radius': 15,
                                        'dose': therapy_config.get('dose', 2.0)
                                    }
                                elif therapy_type == 'chemotherapy':
                                    therapy_input['drug_dose'] = therapy_config.get('dose', 1.0)
                                elif therapy_type == 'immunotherapy':
                                    # Immunotherapy affects immune cell recruitment/activity
                                    therapy_input['immune_boost'] = therapy_config.get('dose', 1.0)
                    
                    else:
                        # Handle list format (legacy)
                        for therapy in therapy_schedule:
                            if therapy['start_time'] <= self.time <= therapy['end_time']:
                                therapy_input.update(therapy['params'])
                
                self.step_optimized(dt=1.0, therapy_input=therapy_input)
            
            results.append(self.get_state())
        
        return results


# Factory function for easy switching between optimized and original
def create_abm_model(grid_size=(100, 100, 50), optimized=True):
    """Create ABM model with option for optimized version."""
    if optimized:
        return TumorImmuneABM3DOptimized(grid_size)
    else:
        from .abm_model import TumorImmuneABM3D
        return TumorImmuneABM3D(grid_size)
