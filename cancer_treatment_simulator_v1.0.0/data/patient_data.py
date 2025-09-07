"""
Data Handler for Patient Data Integration

This module handles loading, processing, and fitting of patient data
for the cancer treatment simulator.

Supports synthetic data generation and real data integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import requests
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class PatientDataHandler:
    """
    Handler for patient data loading, processing, and model fitting.
    """

    def __init__(self):
        """Initialize data handler."""
        self.patient_data = {}
        self.synthetic_patients = {}
        self.data_sources = {
            "tcia": "https://services.cancerimagingarchive.net/services/v4/TCIA/query",
            "seer": None,  # Requires special access
            "synthetic": True,
        }

    def generate_synthetic_patient(
        self, patient_id: str, scenario: str = "standard"
    ) -> pd.DataFrame:
        """
        Generate synthetic patient data for testing.

        Args:
            patient_id: Unique patient identifier
            scenario: Clinical scenario type

        Returns:
            DataFrame with time series data
        """
        scenarios = {
            "standard": {
                "initial_volume": 5.0,  # cm³
                "growth_rate": 0.05,  # 1/day
                "treatment_response": 0.7,
                "noise_level": 0.1,
            },
            "aggressive": {
                "initial_volume": 8.0,
                "growth_rate": 0.12,
                "treatment_response": 0.4,
                "noise_level": 0.15,
            },
            "responsive": {
                "initial_volume": 4.0,
                "growth_rate": 0.03,
                "treatment_response": 0.9,
                "noise_level": 0.08,
            },
            "resistant": {
                "initial_volume": 6.0,
                "growth_rate": 0.08,
                "treatment_response": 0.2,
                "noise_level": 0.12,
            },
        }

        params = scenarios.get(scenario, scenarios["standard"])

        # Generate time points (daily for 6 months)
        time_points = np.arange(0, 180, 1)

        # Generate tumor volume trajectory
        initial_vol = params["initial_volume"]
        growth_rate = params["growth_rate"]
        response = params["treatment_response"]
        noise = params["noise_level"]

        # Pre-treatment growth
        volumes = []
        current_vol = initial_vol

        for i, t in enumerate(time_points):
            if t < 30:  # First 30 days: growth
                current_vol *= 1 + growth_rate + np.random.normal(0, noise * 0.1)
            elif t < 90:  # Days 30-90: treatment effect
                treatment_effect = response * (1 - np.exp(-(t - 30) / 20))
                growth_factor = growth_rate * (1 - treatment_effect)
                current_vol *= 1 + growth_factor + np.random.normal(0, noise)
            else:  # Days 90+: follow-up
                # Possible regrowth or continued response
                if response > 0.6:  # Good responder
                    growth_factor = growth_rate * 0.3
                else:  # Poor responder - regrowth
                    growth_factor = growth_rate * 0.8
                current_vol *= 1 + growth_factor + np.random.normal(0, noise)

            # Ensure positive volume
            current_vol = max(current_vol, 0.1)
            volumes.append(current_vol)

        # Generate additional clinical markers
        psa_levels = []  # Prostate-specific antigen (example marker)
        immune_markers = []

        for i, vol in enumerate(volumes):
            # PSA roughly correlates with tumor volume
            psa = vol * 2 + np.random.normal(0, 0.5)
            psa_levels.append(max(psa, 0.1))

            # Immune marker (simplified)
            if time_points[i] > 30 and time_points[i] < 90:
                # Treatment boosts immune response
                immune_level = 1.5 + np.random.normal(0, 0.2)
            else:
                immune_level = 1.0 + np.random.normal(0, 0.2)
            immune_markers.append(max(immune_level, 0.1))

        # Create DataFrame
        data = pd.DataFrame(
            {
                "time": time_points,
                "tumor_volume": volumes,
                "psa_level": psa_levels,
                "immune_marker": immune_markers,
                "patient_id": patient_id,
                "scenario": scenario,
            }
        )

        # Add treatment indicators
        data["radiation_dose"] = 0.0
        data["immunotherapy_dose"] = 0.0
        data["chemotherapy_dose"] = 0.0

        # Standard treatment schedule
        treatment_days = (data["time"] >= 30) & (data["time"] <= 65)
        data.loc[treatment_days, "radiation_dose"] = 2.0  # 2 Gy daily

        immuno_days = (data["time"] >= 35) & (data["time"] <= 95)
        data.loc[immuno_days, "immunotherapy_dose"] = 1.0

        self.synthetic_patients[patient_id] = data
        return data

    def load_patient_data(self, file_path: str) -> pd.DataFrame:
        """
        Load patient data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with patient data
        """
        try:
            data = pd.read_csv(file_path)

            # Validate required columns
            required_cols = ["time", "tumor_volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Ensure proper data types
            data["time"] = pd.to_numeric(data["time"])
            data["tumor_volume"] = pd.to_numeric(data["tumor_volume"])

            return data

        except Exception as e:
            raise ValueError(f"Error loading patient data: {str(e)}")

    def fetch_tcia_data(self, collection: str = "TCGA-LUAD") -> Optional[pd.DataFrame]:
        """
        Fetch data from The Cancer Imaging Archive (TCIA).

        Args:
            collection: TCIA collection name

        Returns:
            DataFrame with imaging data or None if failed
        """
        try:
            # This is a simplified example - real TCIA API requires authentication
            url = f"{self.data_sources['tcia']}/getCollectionValues"
            params = {"Collection": collection, "format": "json"}

            # Note: In practice, you'd need API keys and proper authentication
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                # Process response (simplified)
                # Real implementation would parse DICOM metadata
                print(f"Successfully connected to TCIA for collection: {collection}")
                return None  # Placeholder
            else:
                print(f"Failed to fetch TCIA data: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching TCIA data: {str(e)}")
            return None

    def create_patient_cohort(self, n_patients: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Create a cohort of synthetic patients for testing.

        Args:
            n_patients: Number of patients to generate

        Returns:
            Dictionary of patient data
        """
        scenarios = ["standard", "aggressive", "responsive", "resistant"]
        cohort = {}

        for i in range(n_patients):
            patient_id = f"PATIENT_{i + 1:03d}"
            scenario = scenarios[i % len(scenarios)]

            # Add some variation
            if i % 3 == 0:
                scenario = "aggressive"
            elif i % 5 == 0:
                scenario = "resistant"

            data = self.generate_synthetic_patient(patient_id, scenario)
            cohort[patient_id] = data

        return cohort

    def fit_ode_to_patient(
        self, patient_data: pd.DataFrame, ode_model, therapy_schedule: Dict
    ) -> Dict:
        """
        Fit ODE model parameters to patient data.

        Args:
            patient_data: Patient time series data
            ode_model: ODE model instance
            therapy_schedule: Treatment schedule

        Returns:
            Fitting results
        """
        # Extract tumor volume data
        time_data = patient_data["time"].values
        volume_data = patient_data["tumor_volume"].values

        # Convert volume to cell count (rough approximation)
        cell_data = volume_data * 1e6  # Assume 1 cm³ ≈ 1e6 cells

        def objective(params):
            """Objective function for parameter fitting."""
            # Update model parameters
            param_names = ["r_tumor", "K_tumor", "alpha_rt", "k_kill"]
            old_params = ode_model.params.copy()

            for i, name in enumerate(param_names):
                ode_model.params[name] = params[i]

            try:
                # Simulate with current parameters
                initial_conditions = np.array([cell_data[0], 1e5, 0, 100, 0])
                results = ode_model.simulate(
                    initial_conditions, time_data, therapy_schedule
                )

                # Calculate error
                predicted_cells = results["tumor_cells"].values
                error = np.sum((predicted_cells - cell_data) ** 2)

            except Exception:
                error = 1e10  # Large penalty for failed simulations
            finally:
                # Restore original parameters
                ode_model.params = old_params

            return error

        # Parameter bounds and initial guess
        bounds = [
            (0.01, 0.5),  # r_tumor
            (1e8, 1e10),  # K_tumor
            (0.1, 1.0),  # alpha_rt
            (1e-12, 1e-6),  # k_kill
        ]

        x0 = [0.1, 1e9, 0.3, 1e-9]

        # Optimize
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        # Update model with fitted parameters
        param_names = ["r_tumor", "K_tumor", "alpha_rt", "k_kill"]
        fitted_params = {}
        for i, name in enumerate(param_names):
            fitted_params[name] = result.x[i]
            ode_model.params[name] = result.x[i]

        # Calculate fit quality
        final_error = result.fun
        r_squared = 1 - (final_error / np.var(cell_data))

        return {
            "fitted_parameters": fitted_params,
            "fit_error": final_error,
            "r_squared": max(0, r_squared),  # Ensure non-negative
            "optimization_success": result.success,
            "patient_data": patient_data,
            "optimization_result": result,
        }

    def compare_treatment_protocols(
        self, patient_data: pd.DataFrame, protocols: Dict[str, Dict]
    ) -> Dict:
        """
        Compare different treatment protocols for a patient.

        Args:
            patient_data: Patient baseline data
            protocols: Dictionary of treatment protocols

        Returns:
            Comparison results
        """
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from models.ode_model import TumorImmuneODE

        results = {}

        for protocol_name, therapy_schedule in protocols.items():
            # Create fresh model instance
            model = TumorImmuneODE()

            # Fit to patient data first
            fit_result = self.fit_ode_to_patient(patient_data, model, therapy_schedule)

            # Simulate treatment outcome
            initial_conditions = np.array(
                [
                    patient_data["tumor_volume"].iloc[0] * 1e6,  # Convert to cells
                    1e5,  # Initial immune cells
                    0,  # No initial drug
                    100,  # Normal oxygen
                    0,  # No initial radiation
                ]
            )

            time_points = np.arange(0, 365)  # Simulate 1 year
            simulation = model.simulate(
                initial_conditions, time_points, therapy_schedule
            )

            # Calculate metrics
            metrics = model.calculate_metrics(simulation)

            results[protocol_name] = {
                "simulation": simulation,
                "metrics": metrics,
                "fit_quality": fit_result["r_squared"],
                "fitted_parameters": fit_result["fitted_parameters"],
            }

        return results

    def export_results(self, results: Dict, filename: str):
        """
        Export simulation results to file.

        Args:
            results: Simulation results dictionary
            filename: Output filename
        """
        if filename.endswith(".csv"):
            # Export as CSV
            if "simulation" in results:
                results["simulation"].to_csv(filename, index=False)
            else:
                # Multiple simulations
                combined_data = []
                for name, data in results.items():
                    if "simulation" in data:
                        df = data["simulation"].copy()
                        df["protocol"] = name
                        combined_data.append(df)

                if combined_data:
                    final_df = pd.concat(combined_data, ignore_index=True)
                    final_df.to_csv(filename, index=False)

        elif filename.endswith(".json"):
            import json

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict("records")
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_numpy(results)

            with open(filename, "w") as f:
                json.dump(serializable_results, f, indent=2)

    def plot_patient_data(
        self, patient_data: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot patient data overview.

        Args:
            patient_data: Patient data DataFrame
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Tumor volume
        axes[0, 0].plot(patient_data["time"], patient_data["tumor_volume"], "b-o")
        axes[0, 0].set_xlabel("Time (days)")
        axes[0, 0].set_ylabel("Tumor Volume (cm³)")
        axes[0, 0].set_title("Tumor Volume Trajectory")
        axes[0, 0].grid(True)

        # Treatment indicators
        if "radiation_dose" in patient_data.columns:
            ax_twin = axes[0, 1].twinx()
            axes[0, 1].bar(
                patient_data["time"],
                patient_data["radiation_dose"],
                alpha=0.6,
                color="red",
                label="Radiation",
            )
            if "immunotherapy_dose" in patient_data.columns:
                ax_twin.bar(
                    patient_data["time"],
                    patient_data["immunotherapy_dose"],
                    alpha=0.6,
                    color="green",
                    label="Immunotherapy",
                )
            axes[0, 1].set_xlabel("Time (days)")
            axes[0, 1].set_ylabel("Radiation Dose (Gy)")
            ax_twin.set_ylabel("Immunotherapy Dose")
            axes[0, 1].set_title("Treatment Schedule")
            axes[0, 1].legend()

        # Biomarkers (if available)
        if "psa_level" in patient_data.columns:
            axes[1, 0].plot(patient_data["time"], patient_data["psa_level"], "g-o")
            axes[1, 0].set_xlabel("Time (days)")
            axes[1, 0].set_ylabel("PSA Level")
            axes[1, 0].set_title("Biomarker Levels")
            axes[1, 0].grid(True)

        if "immune_marker" in patient_data.columns:
            axes[1, 1].plot(patient_data["time"], patient_data["immune_marker"], "m-o")
            axes[1, 1].set_xlabel("Time (days)")
            axes[1, 1].set_ylabel("Immune Marker")
            axes[1, 1].set_title("Immune Response")
            axes[1, 1].grid(True)

        plt.tight_layout()
        return fig

    def generate_treatment_protocols(self) -> Dict[str, Dict]:
        """
        Generate standard treatment protocol templates.

        Returns:
            Dictionary of treatment protocols
        """
        protocols = {
            "standard_rt": {
                "radiotherapy": {
                    "start_time": 30,
                    "end_time": 65,
                    "dose": 2.0,
                    "frequency": 1,
                    "duration": 1,
                }
            },
            "rt_plus_immuno": {
                "radiotherapy": {
                    "start_time": 30,
                    "end_time": 65,
                    "dose": 2.0,
                    "frequency": 1,
                    "duration": 1,
                },
                "immunotherapy": {
                    "start_time": 35,
                    "end_time": 95,
                    "dose": 1.0,
                    "frequency": 7,
                    "duration": 1,
                },
            },
            "hypofractionated_rt": {
                "radiotherapy": {
                    "start_time": 30,
                    "end_time": 44,
                    "dose": 4.0,
                    "frequency": 2,
                    "duration": 1,
                }
            },
            "sequential_therapy": {
                "chemotherapy": {
                    "start_time": 20,
                    "end_time": 50,
                    "dose": 1.5,
                    "frequency": 7,
                    "duration": 1,
                },
                "radiotherapy": {
                    "start_time": 55,
                    "end_time": 90,
                    "dose": 2.0,
                    "frequency": 1,
                    "duration": 1,
                },
                "immunotherapy": {
                    "start_time": 95,
                    "end_time": 155,
                    "dose": 1.0,
                    "frequency": 14,
                    "duration": 1,
                },
            },
        }

        return protocols
