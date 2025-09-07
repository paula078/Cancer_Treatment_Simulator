"""
PDE Model for Spatial Tumor-Drug-Oxygen Dynamics

This module implements partial differential equations for modeling
spatial diffusion and transport of drugs, oxygen, and radiation effects
in tumor tissue.

Based on reaction-diffusion equations used in cancer modeling.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt


class SpatialTumorPDE:
    """
    PDE model for spatial tumor dynamics with drug and oxygen diffusion.

    Variables:
    - u: Tumor cell density
    - c: Drug concentration
    - o: Oxygen concentration
    - r: Radiation dose distribution
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        spatial_step: float = 0.1,
        parameters: Optional[Dict] = None,
    ):
        """
        Initialize spatial PDE model.

        Args:
            grid_size: (nx, ny) grid dimensions
            spatial_step: Spatial discretization step (mm)
            parameters: Model parameters
        """
        self.nx, self.ny = grid_size
        self.dx = spatial_step
        self.dy = spatial_step

        # Create coordinate grids
        self.x = np.linspace(0, self.nx * self.dx, self.nx)
        self.y = np.linspace(0, self.ny * self.dy, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Model parameters
        self.params = self._get_default_parameters()
        if parameters:
            self.params.update(parameters)

        # Initialize fields
        self.tumor_density = np.zeros((self.nx, self.ny))
        self.drug_concentration = np.zeros((self.nx, self.ny))
        self.oxygen_level = np.zeros((self.nx, self.ny))
        self.radiation_dose = np.zeros((self.nx, self.ny))

        # Boundary conditions
        self.boundary_conditions = {
            "tumor": "neumann",  # No-flux boundary
            "drug": "dirichlet",  # Fixed concentration at boundary
            "oxygen": "dirichlet",  # Fixed oxygen at boundary
            "radiation": "neumann",  # No-flux for radiation
        }

    def _get_default_parameters(self) -> Dict:
        """Default parameters for spatial model."""
        return {
            # Diffusion coefficients (mm²/day)
            "D_tumor": 0.01,  # Tumor cell motility
            "D_drug": 0.1,  # Drug diffusion
            "D_oxygen": 0.2,  # Oxygen diffusion
            "D_radiation": 0.05,  # Radiation scatter
            # Reaction parameters
            "r_tumor": 0.1,  # Tumor growth rate (1/day)
            "K_tumor": 1.0,  # Local carrying capacity
            "drug_decay": 0.1,  # Drug decay rate (1/day)
            "oxygen_consumption": 0.05,  # Oxygen consumption (1/day)
            "radiation_decay": 0.2,  # Radiation attenuation (1/day)
            # Interaction parameters
            "drug_kill_rate": 0.1,  # Drug cytotoxicity
            "oxygen_threshold": 0.2,  # Hypoxia threshold
            "radiation_alpha": 0.3,  # Radiation sensitivity
            # Boundary values
            "drug_boundary": 1.0,  # Drug concentration at boundary
            "oxygen_boundary": 1.0,  # Oxygen level at boundary
        }

    def initialize_conditions(
        self, tumor_center: Tuple[float, float] = None, tumor_radius: float = 5.0
    ):
        """
        Initialize spatial conditions.

        Args:
            tumor_center: (x, y) center of initial tumor
            tumor_radius: Initial tumor radius (mm)
        """
        if tumor_center is None:
            tumor_center = (self.nx * self.dx / 2, self.ny * self.dy / 2)

        # Initialize tumor as Gaussian blob
        center_x, center_y = tumor_center
        distance = np.sqrt((self.X - center_x) ** 2 + (self.Y - center_y) ** 2)
        self.tumor_density = np.exp(-((distance / tumor_radius) ** 2))

        # Initialize uniform oxygen
        self.oxygen_level = np.ones((self.nx, self.ny)) * self.params["oxygen_boundary"]

        # Initialize zero drug and radiation
        self.drug_concentration = np.zeros((self.nx, self.ny))
        self.radiation_dose = np.zeros((self.nx, self.ny))

    def apply_boundary_conditions(
        self, field: np.ndarray, bc_type: str, boundary_value: float = 0.0
    ) -> np.ndarray:
        """
        Apply boundary conditions to a field.

        Args:
            field: 2D array to apply BC to
            bc_type: 'dirichlet' or 'neumann'
            boundary_value: Value for Dirichlet BC

        Returns:
            Field with boundary conditions applied
        """
        if bc_type == "dirichlet":
            field[0, :] = boundary_value  # Top
            field[-1, :] = boundary_value  # Bottom
            field[:, 0] = boundary_value  # Left
            field[:, -1] = boundary_value  # Right

        elif bc_type == "neumann":
            # Zero gradient (no-flux) boundaries
            field[0, :] = field[1, :]  # Top
            field[-1, :] = field[-2, :]  # Bottom
            field[:, 0] = field[:, 1]  # Left
            field[:, -1] = field[:, -2]  # Right

        return field

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute 2D Laplacian using improved finite differences with boundary handling.

        Args:
            field: 2D array

        Returns:
            Laplacian of the field
        """
        # Use centered finite differences for interior points
        laplacian = np.zeros_like(field)

        # Interior points using 5-point stencil
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]
        ) / self.dx**2 + (
            field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]
        ) / self.dy**2

        # Handle boundaries with one-sided differences
        # Top and bottom boundaries
        laplacian[0, 1:-1] = (field[1, 1:-1] - field[0, 1:-1]) / self.dx**2
        laplacian[-1, 1:-1] = (field[-2, 1:-1] - field[-1, 1:-1]) / self.dx**2

        # Left and right boundaries
        laplacian[1:-1, 0] = (field[1:-1, 1] - field[1:-1, 0]) / self.dy**2
        laplacian[1:-1, -1] = (field[1:-1, -2] - field[1:-1, -1]) / self.dy**2

        return laplacian

    def step(self, dt: float, therapy_input: Optional[Dict] = None) -> None:
        """
        Advance PDE system by one time step using improved explicit scheme with stability control.

        Args:
            dt: Time step (days)
            therapy_input: Dictionary with therapy inputs
        """
        if therapy_input is None:
            therapy_input = {}

        # Check stability conditions and adapt time step if needed
        max_diffusion = max(
            self.params["D_tumor"],
            self.params["D_drug"],
            self.params["D_oxygen"],
            self.params["D_radiation"],
        )
        stability_dt = 0.25 * min(self.dx**2, self.dy**2) / max_diffusion

        if dt > stability_dt:
            # Use multiple sub-steps for stability
            n_substeps = int(np.ceil(dt / stability_dt))
            sub_dt = dt / n_substeps
            for _ in range(n_substeps):
                self._substep(sub_dt, therapy_input)
        else:
            self._substep(dt, therapy_input)

    def _substep(self, dt: float, therapy_input: Dict) -> None:
        """Perform a single sub-timestep."""
        # Get current fields
        u = self.tumor_density.copy()
        c = self.drug_concentration.copy()
        o = self.oxygen_level.copy()
        r = self.radiation_dose.copy()

        # Compute Laplacians
        lap_u = self.compute_laplacian(u)
        lap_c = self.compute_laplacian(c)
        lap_o = self.compute_laplacian(o)
        lap_r = self.compute_laplacian(r)

        # Enhanced tumor equation with angiogenesis and necrosis
        tumor_growth = self.params["r_tumor"] * u * (1 - u / self.params["K_tumor"])

        # Angiogenesis factor (promotes growth in areas with moderate tumor density)
        angio_factor = u * (1 - u) * 2  # Peak at u=0.5
        tumor_growth += 0.05 * angio_factor

        # Necrosis in low oxygen regions
        necrosis = 0.1 * u * np.maximum(0, self.params["oxygen_threshold"] - o)

        drug_kill = self.params["drug_kill_rate"] * c * u
        radiation_kill = self.params["radiation_alpha"] * r * u

        # Enhanced hypoxia effect with threshold
        hypoxia_factor = np.where(
            o > self.params["oxygen_threshold"],
            o / (o + self.params["oxygen_threshold"]),
            0.1,
        )  # Severe growth reduction in hypoxia
        tumor_growth *= hypoxia_factor

        du_dt = (
            self.params["D_tumor"] * lap_u
            + tumor_growth
            - drug_kill
            - radiation_kill
            - necrosis
        )

        # Enhanced drug equation with active transport and binding
        drug_decay = self.params["drug_decay"] * c
        drug_uptake = 0.02 * c * u  # Increased uptake rate

        # Active transport towards tumor regions
        grad_u_x = np.gradient(u, axis=0) / self.dx
        grad_u_y = np.gradient(u, axis=1) / self.dy
        grad_c_x = np.gradient(c, axis=0) / self.dx
        grad_c_y = np.gradient(c, axis=1) / self.dy

        # Advection term for active transport
        active_transport = 0.01 * (grad_u_x * grad_c_x + grad_u_y * grad_c_y)

        drug_source = therapy_input.get("drug_input", 0.0)

        dc_dt = (
            self.params["D_drug"] * lap_c
            - drug_decay
            - drug_uptake
            + drug_source
            + active_transport
        )

        # Enhanced oxygen equation with vascular supply and consumption kinetics
        # Michaelis-Menten consumption kinetics
        max_consumption = self.params["oxygen_consumption"]
        km_oxygen = 0.1  # Half-saturation constant
        oxygen_consumption = max_consumption * u * o / (km_oxygen + o)

        # Distance-dependent vascular supply (higher at boundaries)
        distance_from_boundary = np.minimum(
            np.minimum(
                np.arange(self.nx)[:, None], (self.nx - 1 - np.arange(self.nx))[:, None]
            ),
            np.minimum(
                np.arange(self.ny)[None, :], (self.ny - 1 - np.arange(self.ny))[None, :]
            ),
        )
        vascular_supply = 0.02 * (1 + 0.1 / (distance_from_boundary + 1))

        oxygen_supply = vascular_supply * (self.params["oxygen_boundary"] - o)

        do_dt = self.params["D_oxygen"] * lap_o - oxygen_consumption + oxygen_supply

        # Enhanced radiation equation with scatter and attenuation
        radiation_decay = self.params["radiation_decay"] * r
        radiation_beam = therapy_input.get("radiation_beam", np.zeros_like(r))

        # Tissue attenuation (reduces with depth)
        attenuation = np.exp(-0.01 * distance_from_boundary)
        radiation_beam *= attenuation

        dr_dt = self.params["D_radiation"] * lap_r - radiation_decay + radiation_beam

        # Update fields with explicit Euler
        self.tumor_density += dt * du_dt
        self.drug_concentration += dt * dc_dt
        self.oxygen_level += dt * do_dt
        self.radiation_dose += dt * dr_dt

        # Enhanced stability constraints
        self.tumor_density = np.clip(self.tumor_density, 0, 2 * self.params["K_tumor"])
        self.drug_concentration = np.maximum(self.drug_concentration, 0)
        self.oxygen_level = np.clip(
            self.oxygen_level, 0, 5 * self.params["oxygen_boundary"]
        )
        self.radiation_dose = np.maximum(self.radiation_dose, 0)

        # Apply boundary conditions
        self.tumor_density = self.apply_boundary_conditions(
            self.tumor_density, self.boundary_conditions["tumor"]
        )

        self.drug_concentration = self.apply_boundary_conditions(
            self.drug_concentration,
            self.boundary_conditions["drug"],
            self.params["drug_boundary"],
        )

        self.oxygen_level = self.apply_boundary_conditions(
            self.oxygen_level,
            self.boundary_conditions["oxygen"],
            self.params["oxygen_boundary"],
        )

        self.radiation_dose = self.apply_boundary_conditions(
            self.radiation_dose, self.boundary_conditions["radiation"]
        )

    def add_radiation_beam(
        self, center: Tuple[float, float], width: float = 5.0, dose: float = 2.0
    ) -> np.ndarray:
        """
        Create a radiation beam pattern.

        Args:
            center: (x, y) beam center
            width: Beam width (mm)
            dose: Radiation dose (Gy)

        Returns:
            2D radiation dose distribution
        """
        center_x, center_y = center
        distance = np.sqrt((self.X - center_x) ** 2 + (self.Y - center_y) ** 2)
        beam = dose * np.exp(-((distance / width) ** 2))
        return beam

    def add_drug_injection(
        self,
        center: Tuple[float, float],
        radius: float = 3.0,
        concentration: float = 1.0,
    ) -> np.ndarray:
        """
        Create a localized drug injection.

        Args:
            center: (x, y) injection site
            radius: Injection radius (mm)
            concentration: Drug concentration

        Returns:
            2D drug source distribution
        """
        center_x, center_y = center
        distance = np.sqrt((self.X - center_x) ** 2 + (self.Y - center_y) ** 2)
        injection = np.where(distance <= radius, concentration, 0.0)
        return injection

    def simulate(
        self,
        total_time: float,
        dt: float = 0.01,
        therapy_schedule: Optional[Dict] = None,
        save_frequency: int = 10,
    ) -> Dict:
        """
        Run spatial simulation over time with complete spatiotemporal data.

        Args:
            total_time: Total simulation time (days)
            dt: Time step (days)
            therapy_schedule: Therapy schedule with spatial inputs
            save_frequency: Save spatial fields every N time steps

        Returns:
            Dictionary with time series and complete spatiotemporal evolution
        """
        if therapy_schedule is None:
            therapy_schedule = {}

        n_steps = int(total_time / dt)
        time_points = np.linspace(0, total_time, n_steps)

        # Storage for time series
        tumor_volume = []
        total_drug = []
        avg_oxygen = []
        total_radiation = []

        # Storage for complete spatiotemporal evolution
        saved_times = []
        tumor_evolution = []
        drug_evolution = []
        oxygen_evolution = []
        radiation_evolution = []

        for i, t in enumerate(time_points):
            # Determine therapy inputs at current time
            therapy_input = {}

            # Radiation beam
            if "radiation" in therapy_schedule:
                rt_config = therapy_schedule["radiation"]
                if (
                    rt_config.get("start_time", 0)
                    <= t
                    <= rt_config.get("end_time", np.inf)
                ):
                    beam_center = rt_config.get(
                        "center", (self.nx * self.dx / 2, self.ny * self.dy / 2)
                    )
                    beam_width = rt_config.get("width", 5.0)
                    beam_dose = rt_config.get("dose", 2.0)
                    therapy_input["radiation_beam"] = self.add_radiation_beam(
                        beam_center, beam_width, beam_dose
                    )
                else:
                    therapy_input["radiation_beam"] = np.zeros_like(self.radiation_dose)

            # Drug injection
            if "drug" in therapy_schedule:
                drug_config = therapy_schedule["drug"]
                if (
                    drug_config.get("start_time", 0)
                    <= t
                    <= drug_config.get("end_time", np.inf)
                ):
                    injection_center = drug_config.get(
                        "center", (self.nx * self.dx / 2, self.ny * self.dy / 2)
                    )
                    injection_radius = drug_config.get("radius", 3.0)
                    injection_dose = drug_config.get("dose", 1.0)
                    therapy_input["drug_input"] = self.add_drug_injection(
                        injection_center, injection_radius, injection_dose
                    )
                else:
                    therapy_input["drug_input"] = 0.0

            # Step forward
            self.step(dt, therapy_input)

            # Record metrics
            tumor_volume.append(np.sum(self.tumor_density) * self.dx * self.dy)
            total_drug.append(np.sum(self.drug_concentration))
            avg_oxygen.append(np.mean(self.oxygen_level))
            total_radiation.append(np.sum(self.radiation_dose))

            # Save complete spatial fields at specified frequency
            if i % save_frequency == 0:
                saved_times.append(t)
                tumor_evolution.append(self.tumor_density.copy())
                drug_evolution.append(self.drug_concentration.copy())
                oxygen_evolution.append(self.oxygen_level.copy())
                radiation_evolution.append(self.radiation_dose.copy())

        # Convert evolution lists to 3D arrays (time, x, y)
        tumor_evolution = np.array(tumor_evolution)
        drug_evolution = np.array(drug_evolution)
        oxygen_evolution = np.array(oxygen_evolution)
        radiation_evolution = np.array(radiation_evolution)

        return {
            "time": time_points,
            "tumor_volume": np.array(tumor_volume),
            "total_drug": np.array(total_drug),
            "avg_oxygen": np.array(avg_oxygen),
            "total_radiation": np.array(total_radiation),
            # Complete spatiotemporal evolution
            "saved_times": np.array(saved_times),
            "tumor_evolution": tumor_evolution,
            "drug_evolution": drug_evolution,
            "oxygen_evolution": oxygen_evolution,
            "radiation_evolution": radiation_evolution,
            # Final spatial fields (for backward compatibility)
            "final_tumor": self.tumor_density.copy(),
            "final_drug": self.drug_concentration.copy(),
            "final_oxygen": self.oxygen_level.copy(),
            "final_radiation": self.radiation_dose.copy(),
            "x_grid": self.X,
            "y_grid": self.Y,
        }

    def plot_spatial_fields(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot all spatial fields.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Tumor density
        im1 = axes[0, 0].imshow(
            self.tumor_density.T,
            extent=[0, self.nx * self.dx, 0, self.ny * self.dy],
            origin="lower",
            cmap="Reds",
        )
        axes[0, 0].set_title("Tumor Density")
        axes[0, 0].set_xlabel("x (mm)")
        axes[0, 0].set_ylabel("y (mm)")
        plt.colorbar(im1, ax=axes[0, 0])

        # Drug concentration
        im2 = axes[0, 1].imshow(
            self.drug_concentration.T,
            extent=[0, self.nx * self.dx, 0, self.ny * self.dy],
            origin="lower",
            cmap="Blues",
        )
        axes[0, 1].set_title("Drug Concentration")
        axes[0, 1].set_xlabel("x (mm)")
        axes[0, 1].set_ylabel("y (mm)")
        plt.colorbar(im2, ax=axes[0, 1])

        # Oxygen level
        im3 = axes[1, 0].imshow(
            self.oxygen_level.T,
            extent=[0, self.nx * self.dx, 0, self.ny * self.dy],
            origin="lower",
            cmap="Greens",
        )
        axes[1, 0].set_title("Oxygen Level")
        axes[1, 0].set_xlabel("x (mm)")
        axes[1, 0].set_ylabel("y (mm)")
        plt.colorbar(im3, ax=axes[1, 0])

        # Radiation dose
        im4 = axes[1, 1].imshow(
            self.radiation_dose.T,
            extent=[0, self.nx * self.dx, 0, self.ny * self.dy],
            origin="lower",
            cmap="Purples",
        )
        axes[1, 1].set_title("Radiation Dose")
        axes[1, 1].set_xlabel("x (mm)")
        axes[1, 1].set_ylabel("y (mm)")
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        return fig

    def plot_spatiotemporal_evolution(
        self, results: Dict, figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Plot the complete spatiotemporal evolution of all fields.

        Args:
            results: Results dictionary from simulate() containing evolution data
            figsize: Figure size

        Returns:
            Matplotlib figure showing evolution over time
        """
        if "tumor_evolution" not in results:
            raise ValueError(
                "Results must contain spatiotemporal evolution data. Run simulate() with save_frequency parameter."
            )

        saved_times = results["saved_times"]
        n_timepoints = len(saved_times)

        # Create subplots for each field
        fig, axes = plt.subplots(4, min(n_timepoints, 5), figsize=figsize)
        if n_timepoints == 1:
            axes = axes.reshape(-1, 1)

        # Select time points to display (evenly spaced)
        time_indices = np.linspace(0, n_timepoints - 1, min(n_timepoints, 5), dtype=int)

        field_names = [
            "Tumor Density",
            "Drug Concentration",
            "Oxygen Level",
            "Radiation Dose",
        ]
        field_data = [
            results["tumor_evolution"],
            results["drug_evolution"],
            results["oxygen_evolution"],
            results["radiation_evolution"],
        ]
        cmaps = ["Reds", "Blues", "Greens", "Purples"]

        for col, time_idx in enumerate(time_indices):
            time_val = saved_times[time_idx]

            for row, (field_name, field_array, cmap) in enumerate(
                zip(field_names, field_data, cmaps)
            ):
                ax = axes[row, col] if axes.ndim == 2 else axes[row]

                im = ax.imshow(
                    field_array[time_idx].T,
                    extent=[0, self.nx * self.dx, 0, self.ny * self.dy],
                    origin="lower",
                    cmap=cmap,
                    aspect="equal",
                )

                if col == 0:
                    ax.set_ylabel(f"{field_name}\ny (mm)")
                if row == 3:
                    ax.set_xlabel("x (mm)")

                ax.set_title(f"t = {time_val:.1f} days")

                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle("Spatiotemporal Evolution of PDE Fields", fontsize=16)
        plt.tight_layout()
        return fig

    def plot_temporal_cross_sections(
        self,
        results: Dict,
        positions: List[Tuple[int, int]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot temporal evolution at specific spatial positions.

        Args:
            results: Results dictionary from simulate()
            positions: List of (x, y) grid positions to analyze
            figsize: Figure size

        Returns:
            Matplotlib figure showing temporal evolution at specific points
        """
        if "tumor_evolution" not in results:
            raise ValueError("Results must contain spatiotemporal evolution data.")

        if positions is None:
            # Default positions: center and off-center points
            cx, cy = self.nx // 2, self.ny // 2
            positions = [(cx, cy), (cx // 2, cy // 2), (3 * cx // 2, 3 * cy // 2)]

        saved_times = results["saved_times"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        field_names = [
            "Tumor Density",
            "Drug Concentration",
            "Oxygen Level",
            "Radiation Dose",
        ]
        field_data = [
            results["tumor_evolution"],
            results["drug_evolution"],
            results["oxygen_evolution"],
            results["radiation_evolution"],
        ]
        colors = ["red", "blue", "green", "purple"]

        for i, (field_name, field_array, color) in enumerate(
            zip(field_names, field_data, colors)
        ):
            ax = axes[i]

            for j, (x, y) in enumerate(positions):
                # Extract temporal evolution at this position
                temporal_data = field_array[:, x, y]
                ax.plot(
                    saved_times,
                    temporal_data,
                    color=color,
                    alpha=0.7,
                    linewidth=2,
                    label=f"Position ({x},{y})" if j == 0 else "",
                    linestyle=["-", "--", ":"][j % 3],
                )

            ax.set_xlabel("Time (days)")
            ax.set_ylabel(field_name)
            ax.set_title(f"{field_name} vs Time")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()

        plt.suptitle("Temporal Evolution at Different Spatial Positions", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_spatial_profiles(
        self,
        results: Dict,
        time_indices: List[int] = None,
        direction: str = "x",
        position: int = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot 1D spatial profiles along x or y direction at different times.

        Args:
            results: Results dictionary from simulate()
            time_indices: List of time indices to plot
            direction: "x" or "y" direction for the profile
            position: Fixed position in the other direction (default: center)
            figsize: Figure size

        Returns:
            Matplotlib figure showing spatial profiles
        """
        if "tumor_evolution" not in results:
            raise ValueError("Results must contain spatiotemporal evolution data.")

        saved_times = results["saved_times"]

        if time_indices is None:
            # Default: beginning, middle, and end
            n_times = len(saved_times)
            time_indices = [0, n_times // 2, n_times - 1]

        if position is None:
            position = (self.ny // 2) if direction == "x" else (self.nx // 2)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        field_names = [
            "Tumor Density",
            "Drug Concentration",
            "Oxygen Level",
            "Radiation Dose",
        ]
        field_data = [
            results["tumor_evolution"],
            results["drug_evolution"],
            results["oxygen_evolution"],
            results["radiation_evolution"],
        ]
        colors = ["red", "blue", "green", "purple"]

        # Set up spatial coordinate
        if direction == "x":
            spatial_coord = self.x
            coord_label = "x (mm)"
        else:
            spatial_coord = self.y
            coord_label = "y (mm)"

        for i, (field_name, field_array, color) in enumerate(
            zip(field_names, field_data, colors)
        ):
            ax = axes[i]

            for j, time_idx in enumerate(time_indices):
                time_val = saved_times[time_idx]

                # Extract 1D profile
                if direction == "x":
                    profile = field_array[time_idx, :, position]
                else:
                    profile = field_array[time_idx, position, :]

                ax.plot(
                    spatial_coord,
                    profile,
                    color=color,
                    alpha=0.8,
                    linewidth=2,
                    label=f"t = {time_val:.1f} days",
                    linestyle=["-", "--", ":"][j % 3],
                )

            ax.set_xlabel(coord_label)
            ax.set_ylabel(field_name)
            ax.set_title(f"{field_name} Profile ({direction}-direction)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle(
            f"Spatial Profiles Along {direction.upper()}-Direction", fontsize=14
        )
        plt.tight_layout()
        return fig
