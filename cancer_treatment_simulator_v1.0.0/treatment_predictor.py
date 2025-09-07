#!/usr/bin/env python3
"""
Advanced Treatment Prediction and Optimization Engine

This module implements a comprehensive treatment prediction system that:
1. Analyzes patient characteristics and tumor biology
2. Predicts optimal treatment combinations
3. Provides personalized treatment recommendations
4. Estimates treatment outcomes and survival probability
5. Considers treatment toxicity and quality of life

Based on machine learning approaches used in precision oncology.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class TreatmentPredictor:
    """
    Advanced treatment prediction engine that combines patient characteristics
    with treatment parameters to predict optimal outcomes.
    """

    def __init__(self):
        """Initialize the treatment predictor."""
        self.patient_features = None
        self.treatment_features = None
        self.outcome_predictors = {}
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_patient_features(self, patient_data: pd.DataFrame) -> Dict:
        """
        Extract comprehensive patient features for prediction.

        Args:
            patient_data: Patient time series data

        Returns:
            Dictionary of patient features
        """
        features = {}

        # Basic tumor characteristics
        initial_volume = patient_data["tumor_volume"].iloc[0]
        features["initial_tumor_volume"] = initial_volume
        features["tumor_growth_rate"] = self._calculate_growth_rate(
            patient_data["tumor_volume"]
        )

        # Immune system status
        if "immune_marker" in patient_data.columns:
            features["baseline_immune_level"] = patient_data["immune_marker"].iloc[0]
            features["immune_variability"] = patient_data["immune_marker"].std()
        else:
            features["baseline_immune_level"] = np.random.normal(50, 15)
            features["immune_variability"] = np.random.normal(10, 3)

        # PSA dynamics (prostate cancer marker)
        if "psa_level" in patient_data.columns:
            features["baseline_psa"] = patient_data["psa_level"].iloc[0]
            features["psa_trend"] = self._calculate_trend(patient_data["psa_level"])
        else:
            features["baseline_psa"] = np.random.normal(20, 8)
            features["psa_trend"] = np.random.normal(0.1, 0.05)

        # Patient scenario-based features
        scenario = (
            patient_data["scenario"].iloc[0]
            if "scenario" in patient_data.columns
            else "standard"
        )

        # Convert scenario to numeric features
        features["is_aggressive"] = 1 if scenario == "aggressive" else 0
        features["is_responsive"] = 1 if scenario == "responsive" else 0
        features["is_resistant"] = 1 if scenario == "resistant" else 0

        # Derived risk scores
        features["tumor_aggressiveness_score"] = self._calculate_aggressiveness_score(
            features
        )
        features["treatment_resistance_score"] = self._calculate_resistance_score(
            features
        )
        features["immune_competence_score"] = self._calculate_immune_score(features)

        return features

    def _calculate_growth_rate(self, tumor_series: pd.Series) -> float:
        """Calculate tumor growth rate from time series."""
        if len(tumor_series) < 2:
            return 0.1

        # Use exponential growth model
        time_points = np.arange(len(tumor_series))
        valid_indices = tumor_series > 0

        if valid_indices.sum() < 2:
            return 0.1

        log_volumes = np.log(tumor_series[valid_indices])
        valid_times = time_points[valid_indices]

        # Linear regression on log scale
        coeffs = np.polyfit(valid_times, log_volumes, 1)
        return max(0.001, min(0.5, coeffs[0]))  # Bounded growth rate

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend in a time series."""
        if len(series) < 2:
            return 0.0

        time_points = np.arange(len(series))
        coeffs = np.polyfit(time_points, series, 1)
        return coeffs[0]

    def _calculate_aggressiveness_score(self, features: Dict) -> float:
        """Calculate tumor aggressiveness score."""
        score = 0.0
        score += features["tumor_growth_rate"] * 20  # Growth rate contribution
        score += features["is_aggressive"] * 30  # Scenario contribution
        score += max(0, features["baseline_psa"] - 10) * 0.5  # PSA contribution
        return min(100, max(0, score))

    def _calculate_resistance_score(self, features: Dict) -> float:
        """Calculate treatment resistance score."""
        score = 0.0
        score += features["is_resistant"] * 40  # Direct resistance
        score += features["tumor_growth_rate"] * 15  # Fast growth = more resistance
        score += max(0, features["baseline_psa"] - 20) * 0.3
        return min(100, max(0, score))

    def _calculate_immune_score(self, features: Dict) -> float:
        """Calculate immune competence score."""
        score = 50.0  # Baseline
        score += features["baseline_immune_level"] * 0.5
        score += features["is_responsive"] * 20
        score -= features["is_aggressive"] * 10
        return min(100, max(0, score))

    def generate_treatment_combinations(
        self, n_combinations: int = 100
    ) -> pd.DataFrame:
        """
        Generate diverse treatment combinations for training/optimization.

        Args:
            n_combinations: Number of treatment combinations to generate

        Returns:
            DataFrame with treatment parameters
        """
        treatments = []

        for i in range(n_combinations):
            treatment = {
                # Radiotherapy parameters
                "rt_enabled": np.random.choice([0, 1], p=[0.2, 0.8]),
                "rt_dose": np.random.uniform(1.0, 8.0),
                "rt_start_day": np.random.randint(7, 45),
                "rt_duration": np.random.randint(14, 56),
                "rt_frequency": np.random.randint(1, 7),
                # Immunotherapy parameters
                "immuno_enabled": np.random.choice([0, 1], p=[0.3, 0.7]),
                "immuno_dose": np.random.uniform(0.5, 4.0),
                "immuno_start_day": np.random.randint(14, 60),
                "immuno_duration": np.random.randint(28, 84),
                "immuno_frequency": np.random.randint(7, 21),
                # Chemotherapy parameters
                "chemo_enabled": np.random.choice([0, 1], p=[0.4, 0.6]),
                "chemo_dose": np.random.uniform(0.5, 3.0),
                "chemo_start_day": np.random.randint(7, 30),
                "chemo_duration": np.random.randint(21, 70),
                "chemo_frequency": np.random.randint(3, 14),
                # Combined therapy features
                "total_treatments": 0,
                "treatment_intensity": 0.0,
                "treatment_duration_total": 0,
            }

            # Calculate derived features
            treatment["total_treatments"] = (
                treatment["rt_enabled"]
                + treatment["immuno_enabled"]
                + treatment["chemo_enabled"]
            )

            treatment["treatment_intensity"] = (
                treatment["rt_enabled"] * treatment["rt_dose"] * 0.3
                + treatment["immuno_enabled"] * treatment["immuno_dose"] * 0.2
                + treatment["chemo_enabled"] * treatment["chemo_dose"] * 0.5
            )

            treatment["treatment_duration_total"] = max(
                treatment["rt_duration"] if treatment["rt_enabled"] else 0,
                treatment["immuno_duration"] if treatment["immuno_enabled"] else 0,
                treatment["chemo_duration"] if treatment["chemo_enabled"] else 0,
            )

            treatments.append(treatment)

        return pd.DataFrame(treatments)

    def simulate_treatment_outcomes(
        self, patient_features: Dict, treatment_combinations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Simulate treatment outcomes for different combinations.

        Args:
            patient_features: Patient characteristics
            treatment_combinations: Treatment parameter combinations

        Returns:
            DataFrame with simulated outcomes
        """
        outcomes = []

        for _, treatment in treatment_combinations.iterrows():
            outcome = self._predict_single_outcome(
                patient_features, treatment.to_dict()
            )
            outcomes.append(outcome)

        return pd.DataFrame(outcomes)

    def _predict_single_outcome(self, patient_features: Dict, treatment: Dict) -> Dict:
        """
        Predict outcome for a single patient-treatment combination.

        This uses a simplified biological model that considers:
        - Patient tumor aggressiveness
        - Treatment intensity and timing
        - Drug resistance and immune response
        """
        # Base survival probability
        base_survival = 0.7

        # Tumor control probability
        tumor_control = 0.5

        # Treatment effects
        if treatment["rt_enabled"]:
            rt_effect = self._calculate_radiotherapy_effect(patient_features, treatment)
            tumor_control += rt_effect["tumor_control_boost"]
            base_survival += rt_effect["survival_boost"]

        if treatment["immuno_enabled"]:
            immuno_effect = self._calculate_immunotherapy_effect(
                patient_features, treatment
            )
            tumor_control += immuno_effect["tumor_control_boost"]
            base_survival += immuno_effect["survival_boost"]

        if treatment["chemo_enabled"]:
            chemo_effect = self._calculate_chemotherapy_effect(
                patient_features, treatment
            )
            tumor_control += chemo_effect["tumor_control_boost"]
            base_survival += chemo_effect["survival_boost"]

        # Calculate toxicity
        toxicity_score = self._calculate_toxicity(treatment)

        # Apply patient-specific modifiers
        aggressiveness_penalty = patient_features["tumor_aggressiveness_score"] * 0.002
        resistance_penalty = patient_features["treatment_resistance_score"] * 0.003
        immune_bonus = patient_features["immune_competence_score"] * 0.001

        # Final outcomes
        final_tumor_control = max(
            0.1,
            min(
                0.95,
                tumor_control
                - aggressiveness_penalty
                - resistance_penalty
                + immune_bonus,
            ),
        )

        final_survival = max(
            0.2,
            min(
                0.98,
                base_survival
                - aggressiveness_penalty
                - resistance_penalty
                + immune_bonus
                - toxicity_score * 0.1,
            ),
        )

        # Calculate additional metrics
        progression_free_survival = final_tumor_control * 0.8 + np.random.normal(
            0, 0.05
        )
        progression_free_survival = max(0.1, min(0.95, progression_free_survival))

        quality_of_life = max(
            0.3, min(1.0, 0.8 - toxicity_score * 0.15 + np.random.normal(0, 0.05))
        )

        # Tumor volume reduction
        baseline_reduction = 0.3
        treatment_boost = (
            tumor_control - 0.5
        ) * 0.6  # Convert control prob to reduction
        tumor_reduction = max(0.0, min(0.9, baseline_reduction + treatment_boost))

        return {
            "tumor_control_probability": final_tumor_control,
            "overall_survival_probability": final_survival,
            "progression_free_survival": progression_free_survival,
            "tumor_volume_reduction": tumor_reduction,
            "toxicity_score": toxicity_score,
            "quality_of_life_score": quality_of_life,
            "treatment_cost_relative": self._calculate_treatment_cost(treatment),
            "treatment_duration_weeks": treatment["treatment_duration_total"] / 7,
        }

    def _calculate_radiotherapy_effect(
        self, patient_features: Dict, treatment: Dict
    ) -> Dict:
        """Calculate radiotherapy treatment effects."""
        dose_effect = min(
            0.4, treatment["rt_dose"] * 0.06
        )  # Higher dose = better effect

        # Timing effect - earlier is generally better
        timing_effect = max(0, 0.1 - treatment["rt_start_day"] * 0.002)

        # Fractionation effect
        fractionation_bonus = 0.05 if treatment["rt_frequency"] <= 2 else 0.0

        # Patient-specific modifiers
        resistance_penalty = patient_features["treatment_resistance_score"] * 0.001

        tumor_control_boost = (
            dose_effect + timing_effect + fractionation_bonus - resistance_penalty
        )
        survival_boost = (
            tumor_control_boost * 0.7
        )  # Survival benefit is ~70% of tumor control

        return {
            "tumor_control_boost": max(0, tumor_control_boost),
            "survival_boost": max(0, survival_boost),
        }

    def _calculate_immunotherapy_effect(
        self, patient_features: Dict, treatment: Dict
    ) -> Dict:
        """Calculate immunotherapy treatment effects."""
        dose_effect = min(0.3, treatment["immuno_dose"] * 0.08)

        # Immunotherapy works better with competent immune system
        immune_multiplier = 1 + patient_features["immune_competence_score"] * 0.01

        # Duration effect - longer treatment often better for immunotherapy
        duration_bonus = min(0.1, treatment["immuno_duration"] * 0.001)

        # Patient-specific effects
        responsiveness_bonus = 0.15 if patient_features["is_responsive"] else 0.0

        base_effect = (
            dose_effect * immune_multiplier + duration_bonus + responsiveness_bonus
        )

        return {
            "tumor_control_boost": max(0, base_effect),
            "survival_boost": max(0, base_effect * 0.8),
        }

    def _calculate_chemotherapy_effect(
        self, patient_features: Dict, treatment: Dict
    ) -> Dict:
        """Calculate chemotherapy treatment effects."""
        dose_effect = min(0.35, treatment["chemo_dose"] * 0.12)

        # Early treatment bonus
        timing_bonus = max(0, 0.08 - treatment["chemo_start_day"] * 0.001)

        # Resistance penalty
        resistance_penalty = patient_features["treatment_resistance_score"] * 0.002

        tumor_control_boost = dose_effect + timing_bonus - resistance_penalty
        survival_boost = tumor_control_boost * 0.6  # Chemo has more toxicity

        return {
            "tumor_control_boost": max(0, tumor_control_boost),
            "survival_boost": max(0, survival_boost),
        }

    def _calculate_toxicity(self, treatment: Dict) -> float:
        """Calculate treatment toxicity score."""
        toxicity = 0.0

        if treatment["rt_enabled"]:
            toxicity += treatment["rt_dose"] * 0.05
            toxicity += max(0, treatment["rt_duration"] - 30) * 0.001

        if treatment["immuno_enabled"]:
            toxicity += treatment["immuno_dose"] * 0.03
            # Immunotherapy generally has lower toxicity

        if treatment["chemo_enabled"]:
            toxicity += treatment["chemo_dose"] * 0.08
            toxicity += treatment["chemo_duration"] * 0.002

        # Combination toxicity
        if treatment["total_treatments"] >= 2:
            toxicity *= 1.3  # Combination penalty

        if treatment["total_treatments"] >= 3:
            toxicity *= 1.5  # Triple therapy penalty

        return min(1.0, toxicity)

    def _calculate_treatment_cost(self, treatment: Dict) -> float:
        """Calculate relative treatment cost."""
        cost = 0.0

        if treatment["rt_enabled"]:
            cost += treatment["rt_dose"] * 500 + treatment["rt_duration"] * 100

        if treatment["immuno_enabled"]:
            cost += treatment["immuno_dose"] * 2000 + treatment["immuno_duration"] * 200

        if treatment["chemo_enabled"]:
            cost += treatment["chemo_dose"] * 800 + treatment["chemo_duration"] * 150

        # Normalize to 0-1 scale (relative to maximum possible cost)
        max_cost = 3 * 2000 + 84 * 200  # Max immunotherapy cost
        return min(1.0, cost / max_cost)

    def train_predictive_models(self, patient_cohort: Dict) -> None:
        """
        Train machine learning models to predict treatment outcomes.

        Args:
            patient_cohort: Dictionary of patient data
        """
        print("Training predictive models...")

        # Generate training data
        all_features = []
        all_outcomes = []
        feature_names = None

        for patient_id, patient_data in patient_cohort.items():
            # Extract patient features
            patient_features = self.extract_patient_features(patient_data)

            # Generate treatment combinations for this patient
            treatments = self.generate_treatment_combinations(50)

            # Simulate outcomes
            outcomes = self.simulate_treatment_outcomes(patient_features, treatments)

            # Store data
            for i, (_, treatment) in enumerate(treatments.iterrows()):
                # Combine patient and treatment features into single feature vector
                combined_features = {**patient_features, **treatment.to_dict()}

                if feature_names is None:
                    feature_names = list(combined_features.keys())

                # Ensure consistent feature ordering
                feature_vector = [combined_features[name] for name in feature_names]
                all_features.append(feature_vector)
                all_outcomes.append(outcomes.iloc[i].to_dict())

        # Convert to arrays
        X = np.array(all_features)

        # Store feature names for later use
        self.feature_names = feature_names

        # Train separate models for different outcomes
        outcome_metrics = [
            "tumor_control_probability",
            "overall_survival_probability",
            "tumor_volume_reduction",
            "toxicity_score",
            "quality_of_life_score",
        ]

        for metric in outcome_metrics:
            y = np.array([outcome[metric] for outcome in all_outcomes])

            # Train ensemble model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

            # Fit models
            rf_model.fit(X, y)
            gb_model.fit(X, y)

            # Store models
            self.outcome_predictors[metric] = {
                "rf_model": rf_model,
                "gb_model": gb_model,
                "feature_names": feature_names,
            }

        self.is_trained = True
        print("✅ Predictive models trained successfully!")

    def find_optimal_treatment(
        self, patient_data: pd.DataFrame, optimization_goals: Dict = None
    ) -> Dict:
        """
        Find optimal treatment for a specific patient.

        Args:
            patient_data: Patient time series data
            optimization_goals: Dictionary of optimization weights

        Returns:
            Dictionary with optimal treatment and predicted outcomes
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before optimization")

        if optimization_goals is None:
            optimization_goals = {
                "tumor_control_probability": 0.4,
                "overall_survival_probability": 0.3,
                "quality_of_life_score": 0.2,
                "toxicity_score": -0.1,  # Negative weight to minimize toxicity
            }

        # Extract patient features
        patient_features = self.extract_patient_features(patient_data)

        def objective_function(treatment_params):
            """Objective function for optimization."""
            # Convert flat array to treatment dictionary
            treatment = {
                "rt_enabled": int(treatment_params[0] > 0.5),
                "rt_dose": treatment_params[1],
                "rt_start_day": int(treatment_params[2]),
                "rt_duration": int(treatment_params[3]),
                "rt_frequency": int(treatment_params[4]),
                "immuno_enabled": int(treatment_params[5] > 0.5),
                "immuno_dose": treatment_params[6],
                "immuno_start_day": int(treatment_params[7]),
                "immuno_duration": int(treatment_params[8]),
                "immuno_frequency": int(treatment_params[9]),
                "chemo_enabled": int(treatment_params[10] > 0.5),
                "chemo_dose": treatment_params[11],
                "chemo_start_day": int(treatment_params[12]),
                "chemo_duration": int(treatment_params[13]),
                "chemo_frequency": int(treatment_params[14]),
            }

            # Add derived features
            treatment["total_treatments"] = (
                treatment["rt_enabled"]
                + treatment["immuno_enabled"]
                + treatment["chemo_enabled"]
            )
            treatment["treatment_intensity"] = (
                treatment["rt_enabled"] * treatment["rt_dose"] * 0.3
                + treatment["immuno_enabled"] * treatment["immuno_dose"] * 0.2
                + treatment["chemo_enabled"] * treatment["chemo_dose"] * 0.5
            )
            treatment["treatment_duration_total"] = max(
                treatment["rt_duration"] if treatment["rt_enabled"] else 0,
                treatment["immuno_duration"] if treatment["immuno_enabled"] else 0,
                treatment["chemo_duration"] if treatment["chemo_enabled"] else 0,
            )

            # Predict outcomes using trained models
            predicted_outcomes = self.predict_treatment_outcome(
                patient_features, treatment
            )

            # Calculate weighted objective
            objective = 0.0
            for metric, weight in optimization_goals.items():
                if metric in predicted_outcomes:
                    objective += weight * predicted_outcomes[metric]

            return -objective  # Minimize negative of objective

        # Define parameter bounds
        bounds = [
            (0, 1),  # rt_enabled (will be converted to 0/1)
            (1.0, 8.0),  # rt_dose
            (7, 45),  # rt_start_day
            (14, 56),  # rt_duration
            (1, 7),  # rt_frequency
            (0, 1),  # immuno_enabled
            (0.5, 4.0),  # immuno_dose
            (14, 60),  # immuno_start_day
            (28, 84),  # immuno_duration
            (7, 21),  # immuno_frequency
            (0, 1),  # chemo_enabled
            (0.5, 3.0),  # chemo_dose
            (7, 30),  # chemo_start_day
            (21, 70),  # chemo_duration
            (3, 14),  # chemo_frequency
        ]

        # Optimize
        result = differential_evolution(
            objective_function, bounds, seed=42, maxiter=100
        )

        # Convert optimal parameters back to treatment dictionary
        optimal_params = result.x
        optimal_treatment = {
            "rt_enabled": int(optimal_params[0] > 0.5),
            "rt_dose": optimal_params[1],
            "rt_start_day": int(optimal_params[2]),
            "rt_duration": int(optimal_params[3]),
            "rt_frequency": int(optimal_params[4]),
            "immuno_enabled": int(optimal_params[5] > 0.5),
            "immuno_dose": optimal_params[6],
            "immuno_start_day": int(optimal_params[7]),
            "immuno_duration": int(optimal_params[8]),
            "immuno_frequency": int(optimal_params[9]),
            "chemo_enabled": int(optimal_params[10] > 0.5),
            "chemo_dose": optimal_params[11],
            "chemo_start_day": int(optimal_params[12]),
            "chemo_duration": int(optimal_params[13]),
            "chemo_frequency": int(optimal_params[14]),
        }

        # Add derived features
        optimal_treatment["total_treatments"] = (
            optimal_treatment["rt_enabled"]
            + optimal_treatment["immuno_enabled"]
            + optimal_treatment["chemo_enabled"]
        )

        # Predict outcomes for optimal treatment
        predicted_outcomes = self.predict_treatment_outcome(
            patient_features, optimal_treatment
        )

        return {
            "optimal_treatment": optimal_treatment,
            "predicted_outcomes": predicted_outcomes,
            "optimization_score": -result.fun,
            "patient_features": patient_features,
        }

    def predict_treatment_outcome(
        self, patient_features: Dict, treatment: Dict
    ) -> Dict:
        """
        Predict treatment outcomes using trained models.

        Args:
            patient_features: Patient characteristics
            treatment: Treatment parameters

        Returns:
            Dictionary of predicted outcomes
        """
        if not self.is_trained:
            # Fall back to simplified prediction if models not trained
            return self._predict_single_outcome(patient_features, treatment)

        # Combine features using same logic as training
        combined_features = {**patient_features, **treatment}

        # Use stored feature names to ensure consistent ordering
        if hasattr(self, "feature_names"):
            feature_values = [
                combined_features.get(name, 0) for name in self.feature_names
            ]
        else:
            feature_values = list(combined_features.values())

        X = np.array([feature_values])

        predictions = {}
        for metric, models in self.outcome_predictors.items():
            # Ensemble prediction (average of RF and GB)
            rf_pred = models["rf_model"].predict(X)[0]
            gb_pred = models["gb_model"].predict(X)[0]
            predictions[metric] = (rf_pred + gb_pred) / 2

        return predictions

    def compare_treatments(
        self, patient_data: pd.DataFrame, treatment_options: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple treatment options for a patient.

        Args:
            patient_data: Patient time series data
            treatment_options: List of treatment dictionaries

        Returns:
            DataFrame comparing predicted outcomes
        """
        patient_features = self.extract_patient_features(patient_data)

        comparisons = []
        for i, treatment in enumerate(treatment_options):
            predicted_outcomes = self.predict_treatment_outcome(
                patient_features, treatment
            )

            comparison = {
                "treatment_id": i + 1,
                "treatment_name": self._generate_treatment_name(treatment),
                **predicted_outcomes,
                **treatment,
            }
            comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    def _generate_treatment_name(self, treatment: Dict) -> str:
        """Generate a descriptive name for a treatment combination."""
        components = []

        if treatment["rt_enabled"]:
            components.append(f"RT({treatment['rt_dose']:.1f}Gy)")

        if treatment["immuno_enabled"]:
            components.append(f"Immuno({treatment['immuno_dose']:.1f})")

        if treatment["chemo_enabled"]:
            components.append(f"Chemo({treatment['chemo_dose']:.1f})")

        if not components:
            return "No Treatment"

        return " + ".join(components)

    def plot_treatment_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Create interactive visualization comparing treatment options.

        Args:
            comparison_df: DataFrame from compare_treatments

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Efficacy Outcomes",
                "Safety & Quality",
                "Treatment Duration vs Cost",
                "Overall Ranking",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        treatment_names = comparison_df["treatment_name"]

        # Plot 1: Efficacy outcomes
        fig.add_trace(
            go.Bar(
                name="Tumor Control",
                x=treatment_names,
                y=comparison_df["tumor_control_probability"] * 100,
                marker_color="blue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Overall Survival",
                x=treatment_names,
                y=comparison_df["overall_survival_probability"] * 100,
                marker_color="green",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Plot 2: Safety and quality
        fig.add_trace(
            go.Bar(
                name="Quality of Life",
                x=treatment_names,
                y=comparison_df["quality_of_life_score"] * 100,
                marker_color="orange",
                opacity=0.7,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                name="Toxicity (inverted)",
                x=treatment_names,
                y=(1 - comparison_df["toxicity_score"]) * 100,
                marker_color="red",
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

        # Plot 3: Duration vs Cost scatter
        if "treatment_duration_weeks" in comparison_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_df["treatment_duration_weeks"],
                    y=comparison_df["treatment_cost_relative"] * 100,
                    mode="markers+text",
                    text=range(1, len(comparison_df) + 1),
                    textposition="middle center",
                    marker=dict(size=15, color="purple"),
                    name="Treatment Options",
                ),
                row=2,
                col=1,
            )

        # Plot 4: Overall ranking (weighted score)
        weights = {
            "tumor_control_probability": 0.4,
            "overall_survival_probability": 0.3,
            "quality_of_life_score": 0.2,
            "toxicity_score": -0.1,
        }

        overall_scores = []
        for _, row in comparison_df.iterrows():
            score = sum(
                weights[metric] * row[metric]
                for metric in weights.keys()
                if metric in row
            )
            overall_scores.append(score * 100)

        fig.add_trace(
            go.Bar(
                name="Overall Score",
                x=treatment_names,
                y=overall_scores,
                marker_color="darkblue",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800, title_text="Treatment Comparison Dashboard", showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Treatment Options", row=1, col=1)
        fig.update_xaxes(title_text="Treatment Options", row=1, col=2)
        fig.update_xaxes(title_text="Duration (weeks)", row=2, col=1)
        fig.update_xaxes(title_text="Treatment Options", row=2, col=2)

        fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
        fig.update_yaxes(title_text="Score (%)", row=1, col=2)
        fig.update_yaxes(title_text="Relative Cost (%)", row=2, col=1)
        fig.update_yaxes(title_text="Overall Score", row=2, col=2)

        return fig


# Example usage and testing functions
def create_example_treatments() -> List[Dict]:
    """Create example treatment protocols for comparison."""
    treatments = [
        {
            "rt_enabled": 1,
            "rt_dose": 2.0,
            "rt_start_day": 30,
            "rt_duration": 35,
            "rt_frequency": 1,
            "immuno_enabled": 0,
            "immuno_dose": 0,
            "immuno_start_day": 0,
            "immuno_duration": 0,
            "immuno_frequency": 0,
            "chemo_enabled": 0,
            "chemo_dose": 0,
            "chemo_start_day": 0,
            "chemo_duration": 0,
            "chemo_frequency": 0,
            "total_treatments": 1,
            "treatment_intensity": 0.6,
            "treatment_duration_total": 35,
        },
        {
            "rt_enabled": 1,
            "rt_dose": 2.0,
            "rt_start_day": 30,
            "rt_duration": 35,
            "rt_frequency": 1,
            "immuno_enabled": 1,
            "immuno_dose": 1.0,
            "immuno_start_day": 35,
            "immuno_duration": 56,
            "immuno_frequency": 14,
            "chemo_enabled": 0,
            "chemo_dose": 0,
            "chemo_start_day": 0,
            "chemo_duration": 0,
            "chemo_frequency": 0,
            "total_treatments": 2,
            "treatment_intensity": 0.8,
            "treatment_duration_total": 56,
        },
        {
            "rt_enabled": 1,
            "rt_dose": 2.0,
            "rt_start_day": 30,
            "rt_duration": 35,
            "rt_frequency": 1,
            "immuno_enabled": 1,
            "immuno_dose": 1.5,
            "immuno_start_day": 35,
            "immuno_duration": 70,
            "immuno_frequency": 14,
            "chemo_enabled": 1,
            "chemo_dose": 1.5,
            "chemo_start_day": 14,
            "chemo_duration": 42,
            "chemo_frequency": 7,
            "total_treatments": 3,
            "treatment_intensity": 1.35,
            "treatment_duration_total": 70,
        },
    ]
    return treatments


if __name__ == "__main__":
    print("🧬 Advanced Treatment Prediction Engine")
    print("=====================================")
    print("This module provides comprehensive treatment optimization")
    print("that considers patient biology, treatment combinations,")
    print("and multiple outcome metrics for personalized medicine.")
