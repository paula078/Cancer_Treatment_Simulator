"""
ODE Model for Tumor-Immune-Therapy Dynamics

This module implements the ordinary differential equation system for modeling
systemic interactions between tumor cells, immune cells, drugs, and therapies.

Based on mechanistic models from cancer radiobiology and immuno-oncology.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Optional
import pandas as pd


class TumorImmuneODE:
    """
    ODE model for tumor-immune-therapy dynamics.

    State variables:
    - T: Tumor cell population
    - I: Immune cell population
    - D: Drug concentration
    - O: Oxygen concentration
    - R: Radiation dose accumulated
    """

    def __init__(self, parameters: Optional[Dict] = None):
        """Initialize with default or custom parameters."""
        self.params = self._get_default_parameters()
        if parameters:
            self.params.update(parameters)

    def _get_default_parameters(self) -> Dict:
        """Default biologically-motivated parameters."""
        return {
            # Tumor growth parameters
            "r_tumor": 0.1,  # Intrinsic growth rate (1/day)
            "K_tumor": 1e9,  # Carrying capacity (cells)
            "alpha_tumor": 0.8,  # Tumor growth exponent
            # Immune parameters
            "r_immune": 0.05,  # Immune recruitment rate (1/day)
            "k_kill": 1e-9,  # Immune kill rate (1/(cell*day))
            "delta_immune": 0.02,  # Immune decay rate (1/day)
            "theta_immune": 0.1,  # Immune activation threshold
            # Drug parameters
            "k_in": 0.5,  # Drug input rate (mg/L/day)
            "k_out": 0.2,  # Drug elimination rate (1/day)
            "IC50": 1.0,  # Half-maximal inhibitory concentration (mg/L)
            "hill_drug": 2,  # Hill coefficient for drug effect
            # Radiotherapy parameters
            "alpha_rt": 0.3,  # Linear component of radiation (1/Gy)
            "beta_rt": 0.03,  # Quadratic component of radiation (1/Gy²)
            "lambda_repair": 0.5,  # DNA repair rate (1/hour)
            "OER": 3.0,  # Oxygen enhancement ratio
            # Immunotherapy parameters
            "k_immuno": 0.1,  # Immunotherapy boost factor
            "tau_immuno": 7,  # Immunotherapy half-life (days)
            # Oxygen parameters
            "O_max": 160,  # Maximum oxygen (mmHg)
            "k_oxygen": 0.1,  # Oxygen consumption rate
            "D_oxygen": 0.01,  # Oxygen diffusion coefficient
            # Interaction parameters
            "synergy_rt_immuno": 1.2,  # RT-immunotherapy synergy factor
            "abscopal_factor": 0.1,  # Abscopal effect strength
        }

    def system(self, y: np.ndarray, t: float, therapy_schedule: Dict) -> np.ndarray:
        """
        ODE system for tumor-immune-therapy dynamics.

        Args:
            y: State vector [T, I, D, O, R]
            t: Time
            therapy_schedule: Dictionary with therapy timing and doses

        Returns:
            dydt: Derivatives of state variables
        """
        tumor_cells, immune_cells, drug_conc, oxygen_level, radiation_dose = y

        # Prevent negative values
        tumor_cells = max(tumor_cells, 0)
        immune_cells = max(immune_cells, 0)
        drug_conc = max(drug_conc, 0)
        oxygen_level = max(oxygen_level, 0)
        radiation_dose = max(radiation_dose, 0)

        # Get therapy doses at current time
        rt_dose = self._get_therapy_dose(t, therapy_schedule.get("radiotherapy", {}))
        immuno_dose = self._get_therapy_dose(
            t, therapy_schedule.get("immunotherapy", {})
        )
        drug_dose = self._get_therapy_dose(t, therapy_schedule.get("chemotherapy", {}))

        # Tumor dynamics
        growth_term = (
            self.params["r_tumor"]
            * tumor_cells
            * (1 - (tumor_cells / self.params["K_tumor"]) ** self.params["alpha_tumor"])
        )

        # Improved immune kill term with saturation and resistance
        immune_efficiency = immune_cells / (
            immune_cells + 1e4
        )  # Saturation at high immune density
        tumor_resistance = 1 / (1 + tumor_cells / 1e7)  # Tumor resistance with size
        immune_kill = (
            self.params["k_kill"]
            * immune_cells
            * tumor_cells
            * immune_efficiency
            * tumor_resistance
        )

        # Enhanced drug cytotoxicity with resistance evolution
        drug_effect = drug_conc ** self.params["hill_drug"] / (
            self.params["IC50"] ** self.params["hill_drug"]
            + drug_conc ** self.params["hill_drug"]
        )
        # Add resistance factor that depends on accumulated dose
        resistance_factor = 1 / (
            1 + 0.1 * np.sum([drug_conc]) * 0.01
        )  # Simplified resistance
        drug_kill = drug_effect * tumor_cells * 0.15 * resistance_factor

        # Enhanced radiation effect with fractionation and repair
        oer_factor = (self.params["OER"] * oxygen_level / self.params["O_max"] + 1) / (
            self.params["OER"] + 1
        )
        # Include sublethal damage repair between fractions
        repair_factor = np.exp(-radiation_dose * 0.01)  # Repair reduces effectiveness
        rt_effect = (
            (self.params["alpha_rt"] * rt_dose + self.params["beta_rt"] * rt_dose**2)
            * oer_factor
            * repair_factor
        )
        rt_kill = rt_effect * tumor_cells

        # Add radiation-induced bystander effect
        bystander_kill = (
            0.02 * rt_dose * tumor_cells * (radiation_dose / (radiation_dose + 1))
        )

        # Repair term for radiation damage
        repair_term = (
            self.params["lambda_repair"] * radiation_dose / 24
        )  # Convert to daily rate

        dT_dt = growth_term - immune_kill - drug_kill - rt_kill - bystander_kill

        # Enhanced immune dynamics with memory and exhaustion
        tumor_antigen_load = tumor_cells / 1e7  # Normalized antigen presentation
        immune_recruitment = self.params["r_immune"] * (
            1 + self.params["abscopal_factor"] * rt_dose + 0.5 * tumor_antigen_load
        )

        # Immunotherapy effect with T-cell expansion
        immuno_expansion = (
            immuno_dose
            * self.params["k_immuno"]
            * immune_cells
            * (1 + tumor_antigen_load)
        )

        # Immune exhaustion from prolonged activation
        exhaustion_factor = 1 / (
            1 + tumor_cells / 5e6
        )  # High tumor burden causes exhaustion
        immune_activation = immuno_expansion * exhaustion_factor

        immune_decay = self.params["delta_immune"] * immune_cells

        # Immunogenic cell death from RT and drugs
        immunogenic_boost = 0.02 * (
            rt_kill + drug_kill * 0.5
        )  # RT more immunogenic than chemo

        dI_dt = (
            immune_recruitment + immune_activation + immunogenic_boost - immune_decay
        )

        # Drug pharmacokinetics
        drug_input = drug_dose * self.params["k_in"]
        drug_elimination = self.params["k_out"] * drug_conc

        dD_dt = drug_input - drug_elimination

        # Enhanced oxygen dynamics with vascular supply and consumption
        # Vascular oxygen supply (higher at boundaries, lower in tumor core)
        tumor_burden = tumor_cells / self.params["K_tumor"]
        vascular_supply = self.params["O_max"] * 0.1 * (1 - tumor_burden * 0.5)

        # Michaelis-Menten oxygen consumption by tumor cells
        km_oxygen = 50  # Half-saturation constant
        oxygen_consumption = (
            self.params["k_oxygen"]
            * tumor_cells
            * oxygen_level
            / (km_oxygen + oxygen_level)
        )

        dO_dt = vascular_supply - oxygen_consumption

        # Radiation dose accumulation and repair
        dR_dt = rt_dose - repair_term

        return np.array([dT_dt, dI_dt, dD_dt, dO_dt, dR_dt])

    def _get_therapy_dose(self, t: float, therapy_config: Dict) -> float:
        """
        Calculate therapy dose at time t based on schedule.

        Args:
            t: Current time (days)
            therapy_config: Therapy configuration dict

        Returns:
            Current dose rate
        """
        if not therapy_config:
            return 0.0

        start_time = therapy_config.get("start_time", 0)
        end_time = therapy_config.get("end_time", np.inf)
        dose = therapy_config.get("dose", 0)
        frequency = therapy_config.get("frequency", 1)  # Daily by default
        duration = therapy_config.get("duration", 1)  # 1 day duration

        if t < start_time or t > end_time:
            return 0.0

        # Check if we're in a treatment cycle
        cycle_time = (t - start_time) % frequency
        if cycle_time < duration:
            return dose
        else:
            return 0.0

    def simulate(
        self,
        initial_conditions: np.ndarray,
        time_points: np.ndarray,
        therapy_schedule: Dict,
    ) -> pd.DataFrame:
        """
        Simulate the ODE system.

        Args:
            initial_conditions: Initial state [T0, I0, D0, O0, R0]
            time_points: Time points for simulation
            therapy_schedule: Therapy schedule configuration

        Returns:
            DataFrame with simulation results
        """
        solution = odeint(
            self.system,
            initial_conditions,
            time_points,
            args=(therapy_schedule,),
            rtol=1e-8,
            atol=1e-10,
        )

        # Ensure non-negative values
        solution = np.maximum(solution, 0)

        results = pd.DataFrame(
            {
                "time": time_points,
                "tumor_cells": solution[:, 0],
                "immune_cells": solution[:, 1],
                "drug_concentration": solution[:, 2],
                "oxygen_level": solution[:, 3],
                "radiation_dose": solution[:, 4],
            }
        )

        # Add therapy indicators
        results["rt_dose"] = [
            self._get_therapy_dose(t, therapy_schedule.get("radiotherapy", {}))
            for t in time_points
        ]
        results["immuno_dose"] = [
            self._get_therapy_dose(t, therapy_schedule.get("immunotherapy", {}))
            for t in time_points
        ]
        results["chemo_dose"] = [
            self._get_therapy_dose(t, therapy_schedule.get("chemotherapy", {}))
            for t in time_points
        ]

        return results

    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate treatment outcome metrics."""
        final_tumor = results["tumor_cells"].iloc[-1]
        initial_tumor = results["tumor_cells"].iloc[0]
        min_tumor = results["tumor_cells"].min()

        # Tumor Control Probability (simplified)
        tcp = np.exp(-final_tumor / 1e6)

        # Immune Cell Viability
        avg_immune = results["immune_cells"].mean()
        icv = min(avg_immune / 1e5, 1.0)

        # Treatment response
        tumor_reduction = (initial_tumor - min_tumor) / initial_tumor

        return {
            "tumor_control_probability": tcp,
            "immune_cell_viability": icv,
            "tumor_reduction": tumor_reduction,
            "final_tumor_burden": final_tumor,
            "treatment_duration": results["time"].iloc[-1],
        }

    def fit_to_data(
        self,
        data: pd.DataFrame,
        therapy_schedule: Dict,
        param_bounds: Optional[Dict] = None,
    ) -> Dict:
        """
        Fit model parameters to patient data using least squares.

        Args:
            data: DataFrame with columns ['time', 'tumor_volume']
            therapy_schedule: Treatment schedule
            param_bounds: Parameter bounds for optimization

        Returns:
            Fitted parameters and metrics
        """
        from scipy.optimize import minimize

        # Convert tumor volume to cell count (rough approximation)
        data = data.copy()
        data["tumor_cells"] = data["tumor_volume"] * 1e6  # Assume 1mm³ = 1e6 cells

        def objective(params_array):
            # Update parameters
            param_names = ["r_tumor", "K_tumor", "k_kill", "alpha_rt"]
            temp_params = self.params.copy()
            for i, name in enumerate(param_names):
                temp_params[name] = params_array[i]

            # Temporarily update parameters
            old_params = self.params.copy()
            self.params = temp_params

            try:
                # Simulate with current parameters
                initial_conditions = np.array(
                    [data["tumor_cells"].iloc[0], 1e5, 0, 100, 0]
                )
                results = self.simulate(
                    initial_conditions, data["time"].values, therapy_schedule
                )

                # Calculate error
                error = np.sum((results["tumor_cells"] - data["tumor_cells"]) ** 2)

            except Exception:
                error = 1e10  # Large penalty for failed simulations
            finally:
                # Restore original parameters
                self.params = old_params

            return error

        # Initial guess and bounds
        x0 = [0.1, 1e9, 1e-9, 0.3]
        bounds = [(0.01, 1.0), (1e8, 1e10), (1e-12, 1e-6), (0.1, 1.0)]

        if param_bounds:
            for i, name in enumerate(["r_tumor", "K_tumor", "k_kill", "alpha_rt"]):
                if name in param_bounds:
                    bounds[i] = param_bounds[name]

        # Optimize
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        # Update parameters with fitted values
        param_names = ["r_tumor", "K_tumor", "k_kill", "alpha_rt"]
        fitted_params = {}
        for i, name in enumerate(param_names):
            fitted_params[name] = result.x[i]
            self.params[name] = result.x[i]

        return {
            "fitted_parameters": fitted_params,
            "fit_error": result.fun,
            "success": result.success,
            "optimization_result": result,
        }
