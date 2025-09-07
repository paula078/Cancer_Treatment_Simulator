"""
Main Streamlit GUI Application for Cancer Treatment Simulator

This is the main interface for the biomedical cancer treatment dynamics simulator.
Provides interactive controls, visualizations, and analysis tools.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict
import time
import traceback
import json
import io

# Import model components
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ode_model import TumorImmuneODE
from models.pde_model import SpatialTumorPDE
from models.abm_model import TumorImmuneABM3D
from models.abm_model_optimized import TumorImmuneABM3DOptimized, CellType  # 🔧 FIX: Import CellType from optimized model
from data.patient_data import PatientDataHandler
from treatment_predictor import TreatmentPredictor, create_example_treatments


class CancerSimulatorGUI:
    """Main GUI class for the cancer treatment simulator."""

    def __init__(self):
        """Initialize the GUI application."""
        self.setup_page_config()
        self.initialize_session_state()
        self.data_handler = PatientDataHandler()

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Cancer Treatment Dynamics Simulator",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS for better styling
        st.markdown(
            """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin: 0.5rem 0;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "simulation_results" not in st.session_state:
            st.session_state.simulation_results = {}

        if "patient_data" not in st.session_state:
            st.session_state.patient_data = None

        if "current_model" not in st.session_state:
            st.session_state.current_model = "ODE"

        if "therapy_schedule" not in st.session_state:
            # Initialize with default therapy schedule
            st.session_state.therapy_schedule = {
                "radiotherapy": {
                    "start_time": 30,
                    "end_time": 65,
                    "dose": 2.0,
                    "frequency": 1,
                    "duration": 1,
                }
            }

    def render_header(self):
        """Render the main application header."""
        st.markdown(
            '<h1 class="main-header">🧬 Cancer Treatment Dynamics Simulator</h1>',
            unsafe_allow_html=True,
        )

        st.markdown("""
        **A comprehensive interactive simulator for modeling cancer treatment dynamics using multiscale mathematical models.**
        
        - **ODE Models**: Systemic tumor-immune-drug interactions
        - **PDE Models**: Spatial diffusion and transport processes  
        - **ABM Models**: Cell-level behaviors and interactions
        - **Patient Data Integration**: Real/synthetic patient datasets
        """)

        st.divider()

    def render_sidebar(self):
        """Render the simplified sidebar with essential controls."""
        with st.sidebar:
            st.header("🎛️ Model & Data Selection")

            # Model selection
            model_type = st.selectbox(
                "Select Model Type",
                options=["ODE", "PDE", "ABM"],
                index=["ODE", "PDE", "ABM"].index(st.session_state.current_model),
                help="Choose the mathematical modeling approach:\n• ODE: Fast, continuous dynamics\n• PDE: Spatial tumor growth\n• ABM: Agent-based micro-interactions",
                key="sidebar_model_selection"
            )
            # Immediately update session state
            if model_type != st.session_state.current_model:
                st.session_state.current_model = model_type
                st.rerun()  # Force refresh to update UI

            # Performance optimization toggle for ABM
            if model_type == "ABM":
                if "abm_optimized" not in st.session_state:
                    st.session_state.abm_optimized = True
                
                st.session_state.abm_optimized = st.checkbox(
                    "🚀 High-Performance ABM", 
                    value=st.session_state.abm_optimized,
                    help="Use optimized ABM with vectorized operations and spatial indexing for 2-5x faster simulations"
                )
                
                if st.session_state.abm_optimized:
                    st.success("⚡ Optimized mode: Faster simulations")
                else:
                    st.info("🐌 Standard mode: Detailed dynamics")

            st.divider()

            # Patient selection
            st.subheader("👤 Patient Data")
            patient_source = st.radio(
                "Data Source",
                options=["Synthetic", "Upload File", "Cohort Analysis"],
                help="Choose patient data source",
            )

            if patient_source == "Synthetic":
                self.render_synthetic_patient_controls()
            elif patient_source == "Upload File":
                self.render_file_upload()
            else:
                self.render_cohort_controls()

            st.divider()

            # Model differences explanation
            with st.expander("ℹ️ Model Differences"):
                st.write("""
                **ODE Models:** 
                - Extended treatment schedules for sustained effects
                - Continuous dosing optimization
                
                **PDE Models:**
                - Spatial drug distribution limitations  
                - Delayed response kinetics
                - Higher radiation doses for spatial coverage
                
                **ABM Models:**
                - Individual cell interactions
                - Stochastic treatment responses
                - Micro-environment effects
                """)

            st.info("🎯 Configure treatment parameters in the **Simulation** tab.")

    def render_synthetic_patient_controls(self):
        """Render controls for synthetic patient generation."""
        scenario = st.selectbox(
            "Patient Scenario",
            options=["standard", "aggressive", "responsive", "resistant"],
            help="Select clinical scenario type",
        )

        patient_id = st.text_input("Patient ID", value="PATIENT_001")

        if st.button("Generate Synthetic Patient", type="primary"):
            with st.spinner("Generating synthetic patient data..."):
                patient_data = self.data_handler.generate_synthetic_patient(
                    patient_id, scenario
                )
                st.session_state.patient_data = patient_data
                st.success(f"Generated data for {patient_id} ({scenario} scenario)")

    def render_file_upload(self):
        """Render file upload interface."""
        uploaded_file = st.file_uploader(
            "Upload Patient Data (CSV)",
            type=["csv"],
            help="CSV file with columns: time, tumor_volume",
        )

        if uploaded_file is not None:
            try:
                patient_data = pd.read_csv(uploaded_file)
                st.session_state.patient_data = patient_data
                st.success("Patient data loaded successfully!")

                # Show data preview
                st.write("**Data Preview:**")
                st.dataframe(patient_data.head())

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    def render_cohort_controls(self):
        """Render cohort analysis controls with full functionality."""
        st.write("### Patient Cohort Generation")
        
        # Cohort generation
        n_patients = st.slider("Number of Patients", 5, 50, 10)

        if st.button("Generate Patient Cohort", type="primary"):
            with st.spinner("Generating patient cohort..."):
                cohort = self.data_handler.create_patient_cohort(n_patients)
                st.session_state.patient_cohort = cohort
                st.success(f"Generated cohort of {n_patients} patients")

        # Display existing cohort status
        if "patient_cohort" in st.session_state and st.session_state.patient_cohort:
            cohort = st.session_state.patient_cohort
            st.success(f"✅ Cohort of {len(cohort)} patients ready")
            
            st.divider()
            st.write("### Quick Cohort Simulation")
            
            # Model selection for cohort analysis (using global cohort_model from render_cohort_simulation_controls)
            cohort_model = st.selectbox(
                "Select Model for Cohort Analysis:",
                ["ODE", "PDE", "ABM"],
                key="cohort_model"
            )
            
            # Quick simulation settings
            simulation_time = st.slider("Simulation Time (days)", 30, 365, 120, key="sidebar_sim_time")
            
            # Patient selection
            patient_options = ["All Patients"] + list(cohort.keys())
            selected_patients = st.multiselect(
                "Select Patients:",
                options=patient_options,
                default=["All Patients"],
                key="sidebar_patients"
            )
            
            if "All Patients" in selected_patients:
                selected_patients = list(cohort.keys())
            
            # Quick run button
            if st.button("� Quick Cohort Simulation", key="sidebar_cohort_run"):
                self.run_cohort_simulation(selected_patients, cohort_model, simulation_time)
            
            st.info("💡 For detailed analysis, use the 'Cohort' tab in the main window")
        else:
            st.info("Generate a patient cohort to enable cohort analysis in the main window.")

    def render_cohort_display(self):
        """Display generated patient cohort data and analysis."""
        st.write("### Generated Patient Cohort")
        
        cohort = st.session_state.patient_cohort
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", len(cohort))
        
        # Count scenarios
        scenarios = {}
        for patient_id, data in cohort.items():
            scenario = data['scenario'].iloc[0]
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
        
        with col2:
            st.metric("Scenarios", len(scenarios))
        with col3:
            total_timepoints = sum(len(data) for data in cohort.values())
            st.metric("Total Data Points", total_timepoints)
        
        # Scenario distribution
        st.write("**Patient Scenario Distribution:**")
        scenario_df = pd.DataFrame(list(scenarios.items()), columns=['Scenario', 'Count'])
        st.dataframe(scenario_df, use_container_width=True)
        
        # Patient list with basic info
        st.write("**Patient Details:**")
        patient_info = []
        for patient_id, data in cohort.items():
            initial_volume = data['tumor_volume'].iloc[0]
            final_volume = data['tumor_volume'].iloc[-1]
            scenario = data['scenario'].iloc[0]
            
            patient_info.append({
                'Patient ID': patient_id,
                'Scenario': scenario,
                'Initial Volume (cm³)': f"{initial_volume:.2f}",
                'Final Volume (cm³)': f"{final_volume:.2f}",
                'Volume Change': f"{((final_volume - initial_volume) / initial_volume * 100):.1f}%"
            })
        
        patient_df = pd.DataFrame(patient_info)
        st.dataframe(patient_df, use_container_width=True)
        
        # Visualization options
        st.write("**Cohort Visualization:**")
        viz_option = st.selectbox(
            "Select visualization:",
            ["Tumor Volume Trajectories", "PSA Level Trajectories", "Immune Response", "Treatment Response Analysis"]
        )
        
        if viz_option == "Tumor Volume Trajectories":
            self.plot_cohort_tumor_volumes(cohort)
        elif viz_option == "PSA Level Trajectories":
            self.plot_cohort_psa_levels(cohort)
        elif viz_option == "Immune Response":
            self.plot_cohort_immune_response(cohort)
        elif viz_option == "Treatment Response Analysis":
            self.plot_cohort_treatment_response(cohort)

    def plot_cohort_tumor_volumes(self, cohort):
        """Plot tumor volume trajectories for the cohort."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {
            'standard': '#1f77b4',
            'aggressive': '#d62728', 
            'responsive': '#2ca02c',
            'resistant': '#ff7f0e'
        }
        
        for patient_id, data in cohort.items():
            scenario = data['scenario'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['tumor_volume'],
                    mode='lines',
                    name=f"{patient_id} ({scenario})",
                    line=dict(color=colors.get(scenario, '#1f77b4')),
                    opacity=0.7
                )
            )
        
        fig.update_layout(
            title="Tumor Volume Trajectories by Patient",
            xaxis_title="Time (days)",
            yaxis_title="Tumor Volume (cm³)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_cohort_psa_levels(self, cohort):
        """Plot PSA level trajectories for the cohort."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {
            'standard': '#1f77b4',
            'aggressive': '#d62728', 
            'responsive': '#2ca02c',
            'resistant': '#ff7f0e'
        }
        
        for patient_id, data in cohort.items():
            scenario = data['scenario'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['psa_level'],
                    mode='lines',
                    name=f"{patient_id} ({scenario})",
                    line=dict(color=colors.get(scenario, '#1f77b4')),
                    opacity=0.7
                )
            )
        
        fig.update_layout(
            title="PSA Level Trajectories by Patient",
            xaxis_title="Time (days)",
            yaxis_title="PSA Level",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_cohort_immune_response(self, cohort):
        """Plot immune response trajectories for the cohort."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {
            'standard': '#1f77b4',
            'aggressive': '#d62728', 
            'responsive': '#2ca02c',
            'resistant': '#ff7f0e'
        }
        
        for patient_id, data in cohort.items():
            scenario = data['scenario'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['immune_marker'],
                    mode='lines',
                    name=f"{patient_id} ({scenario})",
                    line=dict(color=colors.get(scenario, '#1f77b4')),
                    opacity=0.7
                )
            )
        
        fig.update_layout(
            title="Immune Response Trajectories by Patient",
            xaxis_title="Time (days)",
            yaxis_title="Immune Marker Level",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_cohort_treatment_response(self, cohort):
        """Plot treatment response analysis for the cohort."""
        import plotly.graph_objects as go
        
        # Calculate response metrics
        response_data = []
        for patient_id, data in cohort.items():
            scenario = data['scenario'].iloc[0]
            
            # Pre-treatment volume (day 30)
            pre_idx = data['time'].sub(30).abs().idxmin()
            pre_treatment = data.loc[pre_idx, 'tumor_volume']
            
            # End of treatment volume (day 90)
            end_idx = data['time'].sub(90).abs().idxmin()
            end_treatment = data.loc[end_idx, 'tumor_volume']
            
            # Response ratio
            response_ratio = (pre_treatment - end_treatment) / pre_treatment * 100
            
            response_data.append({
                'Patient': patient_id,
                'Scenario': scenario,
                'Pre-treatment Volume': pre_treatment,
                'End-treatment Volume': end_treatment,
                'Volume Reduction (%)': response_ratio
            })
        
        response_df = pd.DataFrame(response_data)
        
        # Create bar chart
        fig = go.Figure()
        
        colors = {
            'standard': '#1f77b4',
            'aggressive': '#d62728', 
            'responsive': '#2ca02c',
            'resistant': '#ff7f0e'
        }
        
        for scenario in response_df['Scenario'].unique():
            scenario_data = response_df[response_df['Scenario'] == scenario]
            fig.add_trace(
                go.Bar(
                    x=scenario_data['Patient'],
                    y=scenario_data['Volume Reduction (%)'],
                    name=scenario,
                    marker_color=colors.get(scenario, '#1f77b4')
                )
            )
        
        fig.update_layout(
            title="Treatment Response by Patient",
            xaxis_title="Patient ID",
            yaxis_title="Volume Reduction (%)",
            height=500,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show response statistics table
        st.write("**Treatment Response Statistics:**")
        st.dataframe(response_df, use_container_width=True)

    def render_cohort_simulation_controls(self):
        """Render controls for running simulations on the cohort."""
        st.write("### Cohort Simulation & Analysis")
        
        cohort = st.session_state.patient_cohort
        
        # Model selection for cohort analysis
        cohort_model = st.selectbox(
            "Select Model for Cohort Analysis:",
            ["ODE", "PDE", "ABM"],
            key="cohort_model"
        )
        
        # Patient selection
        patient_options = ["All Patients"] + list(cohort.keys())
        selected_patients = st.multiselect(
            "Select Patients to Analyze:",
            options=patient_options,
            default=["All Patients"]
        )
        
        if "All Patients" in selected_patients:
            selected_patients = list(cohort.keys())
        
        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            simulation_time = st.number_input("Simulation Time (days)", 30, 365, 120)
        with col2:
            comparison_metric = st.selectbox(
                "Primary Comparison Metric:",
                ["Tumor Volume Reduction", "Final Tumor Burden", "Treatment Response", "Survival Probability"]
            )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Run Cohort Simulation", type="primary"):
                self.run_cohort_simulation(selected_patients, cohort_model, simulation_time)
        
        with col2:
            if st.button("Compare Treatment Protocols"):
                self.run_cohort_treatment_comparison(selected_patients)
        
        with col3:
            if st.button("Export Cohort Results"):
                self.export_cohort_results()
        
        # Treatment Parameter Analysis Section
        st.write("### Treatment Parameter Analysis")
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Parameter Sensitivity", "Dose Response", "Timing Optimization", "Custom Parameter Sweep"]
        )
        
        if analysis_type == "Parameter Sensitivity":
            self.render_parameter_sensitivity_analysis()
        elif analysis_type == "Dose Response":
            self.render_dose_response_analysis()
        elif analysis_type == "Timing Optimization":
            self.render_timing_optimization_analysis()
        elif analysis_type == "Custom Parameter Sweep":
            self.render_custom_parameter_sweep()
        
        # Display cohort simulation results if available
        if "cohort_simulation_results" in st.session_state and st.session_state.cohort_simulation_results:
            self.display_cohort_simulation_results(comparison_metric)

    def run_cohort_simulation(self, selected_patients, model_type, simulation_time):
        """Run simulation on selected patients from the cohort."""
        cohort = st.session_state.patient_cohort
        
        with st.spinner(f"Running {model_type} simulation on {len(selected_patients)} patients..."):
            results = {}
            
            for i, patient_id in enumerate(selected_patients):
                patient_data = cohort[patient_id]
                
                # Update progress
                progress = (i + 1) / len(selected_patients)
                st.progress(progress)
                
                try:
                    # Fit model parameters to patient data
                    fitted_params = self.fit_model_to_patient(patient_data, model_type)
                    
                    # Run simulation with fitted parameters
                    if model_type == "ODE":
                        result = self.run_ode_simulation_for_patient(patient_data, fitted_params, simulation_time)
                    elif model_type == "PDE":
                        result = self.run_pde_simulation_for_patient(patient_data, fitted_params, simulation_time)
                    elif model_type == "ABM":
                        result = self.run_abm_simulation_for_patient(patient_data, fitted_params, simulation_time)
                    
                    results[patient_id] = {
                        'simulation_results': result,
                        'patient_data': patient_data,
                        'fitted_parameters': fitted_params,
                        'scenario': patient_data['scenario'].iloc[0]
                    }
                    
                except Exception as e:
                    st.error(f"Simulation failed for {patient_id}: {str(e)}")
                    results[patient_id] = {'error': str(e)}
            
            st.session_state.cohort_simulation_results = results
            st.success(f"Completed simulation for {len(selected_patients)} patients!")

    def fit_model_to_patient(self, patient_data, model_type):
        """Fit model parameters to individual patient data."""
        # Basic parameter fitting based on patient characteristics
        scenario = patient_data['scenario'].iloc[0]
        
        # Default parameters based on scenario
        if scenario == "aggressive":
            params = {
                "r_tumor": 0.15,
                "treatment_response": 0.4,
                "immune_strength": 0.8
            }
        elif scenario == "responsive":
            params = {
                "r_tumor": 0.03,
                "treatment_response": 0.9,
                "immune_strength": 1.5
            }
        elif scenario == "resistant":
            params = {
                "r_tumor": 0.08,
                "treatment_response": 0.2,
                "immune_strength": 0.6
            }
        else:  # standard
            params = {
                "r_tumor": 0.05,
                "treatment_response": 0.7,
                "immune_strength": 1.0
            }
        
        return params

    def run_ode_simulation_for_patient(self, patient_data, fitted_params, simulation_time):
        """Run ODE simulation for a specific patient."""
        from models.ode_model import TumorImmuneODE
        
        model = TumorImmuneODE()
        
        # Use the therapy schedule from session state but modify based on ODE-specific characteristics
        therapy_schedule = st.session_state.therapy_schedule.copy()
        
        # ODE-specific modifications: More systematic dosing, better drug kinetics
        if "immunotherapy" in therapy_schedule:
            # ODE model handles continuous dosing better - increase frequency
            original_dose = therapy_schedule["immunotherapy"]["dose"]
            therapy_schedule["immunotherapy"]["dose"] = original_dose * fitted_params["immune_strength"]
            # ODE model: extend treatment duration for sustained effect
            therapy_schedule["immunotherapy"]["end_time"] = min(
                therapy_schedule["immunotherapy"]["end_time"] + 30,
                simulation_time - 10
            )
        
        if "radiotherapy" in therapy_schedule:
            # ODE model: fractioned radiotherapy is more effective
            original_dose = therapy_schedule["radiotherapy"]["dose"] 
            therapy_schedule["radiotherapy"]["dose"] = original_dose * 0.8  # Lower per-fraction dose
            # Extend treatment period for fractionation
            therapy_schedule["radiotherapy"]["end_time"] = min(
                therapy_schedule["radiotherapy"]["end_time"] + 20,
                simulation_time - 5
            )
        
        # Initial conditions based on patient data
        initial_volume = patient_data['tumor_volume'].iloc[0]
        initial_conditions = np.array([
            initial_volume * 1e6,  # Convert to cells
            1e5,  # Initial immune cells
            0,    # Initial drug concentration
            100,  # Initial oxygen
            0     # Initial radiation dose
        ])
        
        time_points = np.linspace(0, simulation_time, int(simulation_time) + 1)
        results = model.simulate(initial_conditions, time_points, therapy_schedule)
        
        return results

    def run_pde_simulation_for_patient(self, patient_data, fitted_params, simulation_time):
        """Run PDE simulation for a specific patient."""
        from models.pde_model import SpatialTumorPDE
        
        model = SpatialTumorPDE(grid_size=(50, 50))
        
        initial_volume = patient_data['tumor_volume'].iloc[0]
        tumor_radius = (initial_volume * 3 / (4 * np.pi)) ** (1/3)  # Convert volume to radius
        
        model.initialize_conditions(tumor_center=(5, 5), tumor_radius=tumor_radius)
        
        # Use the therapy schedule from session state but modify for PDE-specific spatial characteristics
        therapy_schedule = st.session_state.therapy_schedule.copy()
        
        # PDE-specific modifications: Better spatial drug distribution, heterogeneous response
        if "immunotherapy" in therapy_schedule:
            # PDE model: spatial diffusion of immunotherapy is limited
            original_dose = therapy_schedule["immunotherapy"]["dose"]
            therapy_schedule["immunotherapy"]["dose"] = original_dose * fitted_params["immune_strength"] * 0.7  # Reduced efficacy due to spatial constraints
            # PDE model: delayed response due to spatial kinetics
            therapy_schedule["immunotherapy"]["start_time"] = therapy_schedule["immunotherapy"]["start_time"] + 5
        
        # Convert 3D therapy schedule to 2D for PDE model if needed
        if "radiotherapy" in therapy_schedule:
            # PDE model: spatial radiation distribution with edge effects
            original_rad = therapy_schedule["radiotherapy"]
            therapy_schedule["radiotherapy"] = {
                "start_time": original_rad.get("start_time", 30),
                "end_time": original_rad.get("end_time", 65),
                "center": (5, 5),  # Fixed center for PDE
                "width": 3.0,
                "dose": original_rad.get("dose", 2.0) * 1.2,  # Higher dose needed for spatial model
            }
        
        results = model.simulate(
            total_time=simulation_time,
            dt=0.1,
            therapy_schedule=therapy_schedule
        )
        
        # Include spatial model data for cohort analysis
        if hasattr(results, 'update') and isinstance(results, dict):
            results['spatial_model'] = model
        else:
            # If results is not a dict, create one
            results = {
                'time': getattr(results, 'time', []),
                'tumor_volume': getattr(results, 'tumor_volume', []),
                'avg_oxygen': getattr(results, 'avg_oxygen', []),
                'total_drug': getattr(results, 'total_drug', []),
                'total_radiation': getattr(results, 'total_radiation', []),
                'spatial_model': model
            }
        
        return results

    def run_abm_simulation_for_patient(self, patient_data, fitted_params, simulation_time):
        """Run ABM simulation for a specific patient."""
        # Use optimized ABM if enabled
        use_optimized = st.session_state.get('abm_optimized', True)
        
        if use_optimized:
            model = TumorImmuneABM3DOptimized(grid_size=(50, 50, 25))
            model.initialize_tumor_optimized(center=(25, 25, 12), radius=max(3, int((patient_data['tumor_volume'].iloc[0] * 3 / (4 * np.pi)) ** (1/3))))
            model.initialize_immune_cells_optimized(count=int(fitted_params["immune_strength"] * 50))
        else:
            from models.abm_model import TumorImmuneABM3D
            model = TumorImmuneABM3D(grid_size=(50, 50, 25))
            
            initial_volume = patient_data['tumor_volume'].iloc[0]
            tumor_radius = max(3, int((initial_volume * 3 / (4 * np.pi)) ** (1/3)))
            
            model.initialize_tumor(center=(25, 25, 12), radius=tumor_radius)
            model.initialize_immune_cells(count=int(fitted_params["immune_strength"] * 50))
        
        # Get therapy schedule from session state
        therapy_schedule = st.session_state.therapy_schedule
        
        # Run simulation
        results = {
            'time': [],
            'tumor_volume': [],
            'immune_cells': [],
            'dead_cells': []
        }
        
        for day in range(simulation_time):
            # Apply therapy based on session state schedule
            therapy_input = {}
            
            # Check for radiation therapy
            if "radiotherapy" in therapy_schedule:
                rad_schedule = therapy_schedule["radiotherapy"]
                if rad_schedule["start_time"] <= day <= rad_schedule["end_time"]:
                    therapy_input["radiation"] = {
                        "center": (25, 25, 12),
                        "radius": 15,
                        "dose": rad_schedule["dose"]
                    }
            
            # Check for drug therapy (immunotherapy)
            if "immunotherapy" in therapy_schedule:
                immuno_schedule = therapy_schedule["immunotherapy"]
                if immuno_schedule["start_time"] <= day <= immuno_schedule["end_time"]:
                    therapy_input["drug_dose"] = immuno_schedule["dose"] * fitted_params["treatment_response"]
            
            model.step(dt=1.0, therapy_input=therapy_input)
            
            # Record results
            results['time'].append(day)
            # 🔧 FIX: Use string keys instead of CellType enum for accessing cell_counts
            results['tumor_volume'].append(model.cell_counts.get('tumor', 0) * 0.001)  # Convert to volume
            results['immune_cells'].append(model.cell_counts.get('immune', 0))
            results['dead_cells'].append(model.cell_counts.get('dead', 0))
        
        return results

    def run_cohort_treatment_comparison(self, selected_patients):
        """Compare different treatment protocols on the cohort."""
        cohort = st.session_state.patient_cohort
        
        # Define treatment protocols to compare
        protocols = {
            "Standard Care": {
                "radiotherapy": {"start_time": 30, "end_time": 65, "dose": 2.0},
                "immunotherapy": {"start_time": 35, "end_time": 95, "dose": 1.0}
            },
            "Aggressive Protocol": {
                "radiotherapy": {"start_time": 14, "end_time": 49, "dose": 3.0},
                "immunotherapy": {"start_time": 21, "end_time": 120, "dose": 1.5}
            },
            "Immunotherapy First": {
                "immunotherapy": {"start_time": 7, "end_time": 60, "dose": 2.0},
                "radiotherapy": {"start_time": 30, "end_time": 65, "dose": 2.0}
            }
        }
        
        with st.spinner("Comparing treatment protocols..."):
            comparison_results = {}
            
            for protocol_name, therapy_schedule in protocols.items():
                protocol_results = {}
                
                for patient_id in selected_patients:
                    patient_data = cohort[patient_id]
                    
                    try:
                        # Use ODE model for protocol comparison
                        result = self.data_handler.compare_treatment_protocols(
                            patient_data, {protocol_name: therapy_schedule}
                        )
                        protocol_results[patient_id] = result[protocol_name]
                        
                    except Exception as e:
                        st.error(f"Protocol comparison failed for {patient_id}: {str(e)}")
                
                comparison_results[protocol_name] = protocol_results
            
            st.session_state.protocol_comparison_results = comparison_results
            st.success("Treatment protocol comparison completed!")
            
            # Display comparison results
            self.display_protocol_comparison_results(comparison_results)

    def display_cohort_simulation_results(self, comparison_metric):
        """Display comprehensive results from cohort simulation with rich visualizations."""
        results = st.session_state.cohort_simulation_results
        
        st.write("### 🧬 Comprehensive Cohort Simulation Results")
        
        # Summary statistics and scenario analysis
        scenarios = {}
        metrics = {}
        detailed_results = {}
        
        for patient_id, patient_result in results.items():
            if 'error' in patient_result:
                continue
                
            scenario = patient_result['scenario']
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
            
            # Store detailed results for advanced analysis
            if scenario not in detailed_results:
                detailed_results[scenario] = []
            detailed_results[scenario].append(patient_result)
            
            # Calculate comprehensive metrics
            sim_results = patient_result['simulation_results']
            
            if comparison_metric == "Tumor Volume Reduction":
                if 'tumor_cells' in sim_results:
                    tumor_data = sim_results['tumor_cells']
                    initial_vol = tumor_data[0] if len(tumor_data) > 0 else 0
                    final_vol = tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') and len(tumor_data) > 0 else (tumor_data[-1] if len(tumor_data) > 0 else 0)
                else:
                    tumor_data = sim_results['tumor_volume']
                    initial_vol = tumor_data[0] if len(tumor_data) > 0 else 0
                    final_vol = tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') and len(tumor_data) > 0 else (tumor_data[-1] if len(tumor_data) > 0 else 0)
                
                if initial_vol > 0:
                    metric_value = (initial_vol - final_vol) / initial_vol * 100
                else:
                    metric_value = 0
                    
            elif comparison_metric == "Final Tumor Burden":
                if 'tumor_cells' in sim_results:
                    tumor_data = sim_results['tumor_cells']
                    metric_value = tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') and len(tumor_data) > 0 else (tumor_data[-1] if len(tumor_data) > 0 else 0)
                else:
                    tumor_data = sim_results['tumor_volume']
                    metric_value = tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') and len(tumor_data) > 0 else (tumor_data[-1] if len(tumor_data) > 0 else 0)
            else:
                metric_value = 0
            
            if scenario not in metrics:
                metrics[scenario] = []
            metrics[scenario].append(metric_value)
        
        # Create rich dashboard layout
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write("#### 📊 Treatment Effectiveness by Scenario")
            effectiveness_data = []
            for scenario, values in metrics.items():
                if values:
                    effectiveness_data.append({
                        'Scenario': scenario.title(),
                        'Mean': f"{np.mean(values):.2f}",
                        'Std Dev': f"{np.std(values):.2f}",
                        'Min': f"{np.min(values):.2f}",
                        'Max': f"{np.max(values):.2f}",
                        'Patients': len(values)
                    })
            
            if effectiveness_data:
                effectiveness_df = pd.DataFrame(effectiveness_data)
                st.dataframe(effectiveness_df, use_container_width=True)
        
        with col2:
            st.write("#### 🎯 Response Distribution")
            scenario_colors = {
                'standard': '#1f77b4',
                'aggressive': '#d62728', 
                'responsive': '#2ca02c',
                'resistant': '#ff7f0e'
            }
            
            # Create pie chart for patient distribution
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(scenarios.keys()),
                values=list(scenarios.values()),
                marker_colors=[scenario_colors.get(s, '#1f77b4') for s in scenarios.keys()],
                hole=0.3
            )])
            fig_pie.update_layout(
                title="Patient Distribution by Scenario",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col3:
            st.write("#### 📈 Summary Stats")
            total_patients = sum(scenarios.values())
            st.metric("Total Patients", total_patients)
            
            all_values = [v for values in metrics.values() for v in values]
            if all_values:
                st.metric("Overall Mean", f"{np.mean(all_values):.2f}")
                st.metric("Best Response", f"{np.max(all_values):.2f}")
                st.metric("Response Range", f"{np.max(all_values) - np.min(all_values):.2f}")
        
        # Advanced visualizations
        st.write("---")
        
        # Create tabs for different views
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "🔬 Tumor Dynamics", "📊 Statistical Analysis", "🎯 Treatment Response", "⚡ Advanced Metrics"
        ])
        
        with viz_tab1:
            self.render_cohort_tumor_dynamics(results, comparison_metric)
            
        with viz_tab2:
            self.render_cohort_statistical_analysis(detailed_results, metrics)
            
        with viz_tab3:
            self.render_cohort_treatment_response_analysis(detailed_results)
            
        with viz_tab4:
            self.render_cohort_advanced_metrics(results)

    def plot_cohort_simulation_comparison(self, results, comparison_metric):
        """Plot comparison of cohort simulation results."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {
            'standard': '#1f77b4',
            'aggressive': '#d62728', 
            'responsive': '#2ca02c',
            'resistant': '#ff7f0e'
        }
        
        # Plot first few patients' tumor trajectories
        for i, (patient_id, patient_result) in enumerate(results.items()):
            if 'error' in patient_result or i >= 10:  # Limit to 10 patients for clarity
                continue
                
            try:
                scenario = patient_result['scenario']
                sim_results = patient_result['simulation_results']
                
                # Get data with proper error checking
                if 'tumor_cells' in sim_results and len(sim_results['tumor_cells']) > 0:
                    y_data = sim_results['tumor_cells']
                elif 'tumor_volume' in sim_results and len(sim_results['tumor_volume']) > 0:
                    y_data = sim_results['tumor_volume']
                else:
                    continue  # Skip if no valid tumor data
                
                if 'time' in sim_results and len(sim_results['time']) > 0:
                    x_data = sim_results['time']
                else:
                    continue  # Skip if no valid time data
                
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines',
                        name=f"{patient_id} ({scenario})",
                        line=dict(color=colors.get(scenario, '#1f77b4')),
                        opacity=0.7
                    )
                )
            except Exception as e:
                # Skip this patient if there's any plotting error
                continue
        
        fig.update_layout(
            title="Tumor Evolution - Cohort Simulation Results",
            xaxis_title="Time (days)",
            yaxis_title="Tumor Volume/Cells",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_cohort_tumor_dynamics(self, results, comparison_metric):
        """Render detailed tumor dynamics visualization for cohort."""
        st.write("#### Tumor Volume Evolution - All Patients")
        
        # Create comprehensive tumor dynamics plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Individual Tumor Trajectories",
                "Scenario-Based Trajectories", 
                "Treatment Effect Timeline",
                "Response Distribution"
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}]
            ]
        )
        
        colors = {
            'standard': '#1f77b4',
            'aggressive': '#d62728', 
            'responsive': '#2ca02c',
            'resistant': '#ff7f0e'
        }
        
        scenario_data = {}
        
        # Plot individual trajectories and collect scenario data
        for i, (patient_id, patient_result) in enumerate(results.items()):
            if 'error' in patient_result:
                continue
                
            try:
                scenario = patient_result['scenario']
                sim_results = patient_result['simulation_results']
                
                # Get tumor data
                if 'tumor_cells' in sim_results and len(sim_results['tumor_cells']) > 0:
                    y_data = sim_results['tumor_cells']
                elif 'tumor_volume' in sim_results and len(sim_results['tumor_volume']) > 0:
                    y_data = sim_results['tumor_volume']
                else:
                    continue
                
                if 'time' in sim_results and len(sim_results['time']) > 0:
                    x_data = sim_results['time']
                else:
                    continue
                
                # Individual trajectories (first subplot)
                fig.add_trace(
                    go.Scatter(
                        x=x_data, y=y_data,
                        mode='lines',
                        name=f"{patient_id}",
                        line=dict(color=colors.get(scenario, '#1f77b4'), width=1),
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Collect data by scenario
                if scenario not in scenario_data:
                    scenario_data[scenario] = {'x': [], 'y': []}
                scenario_data[scenario]['x'].append(x_data)
                scenario_data[scenario]['y'].append(y_data)
                
            except Exception:
                continue
        
        # Plot scenario averages (second subplot)
        for scenario, data in scenario_data.items():
            if len(data['x']) > 0:
                # Calculate average trajectory
                max_len = max(len(x) for x in data['x'])
                avg_x = data['x'][0] if data['x'] else []
                
                # Interpolate all trajectories to same length
                interpolated_y = []
                for x_vals, y_vals in zip(data['x'], data['y']):
                    if len(y_vals) > 0:
                        interpolated_y.append(np.interp(avg_x, x_vals, y_vals))
                
                if interpolated_y:
                    avg_y = np.mean(interpolated_y, axis=0)
                    std_y = np.std(interpolated_y, axis=0)
                    
                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=avg_x, y=avg_y,
                            mode='lines',
                            name=f"{scenario.title()} Mean",
                            line=dict(color=colors.get(scenario, '#1f77b4'), width=3)
                        ),
                        row=1, col=2
                    )
                    
                    # Confidence interval
                    fig.add_trace(
                        go.Scatter(
                            x=list(avg_x) + list(avg_x[::-1]),
                            y=list(avg_y + std_y) + list((avg_y - std_y)[::-1]),
                            fill='toself',
                            fillcolor=colors.get(scenario, '#1f77b4'),
                            opacity=0.2,
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f"{scenario.title()} ±1σ"
                        ),
                        row=1, col=2
                    )
        
        # Treatment timeline (third subplot) - show therapy schedule if available
        if hasattr(st.session_state, 'therapy_schedule') and st.session_state.therapy_schedule:
            therapy = st.session_state.therapy_schedule
            
            if 'radiotherapy' in therapy:
                rt = therapy['radiotherapy']
                fig.add_trace(
                    go.Scatter(
                        x=[rt['start_time'], rt['end_time']],
                        y=[rt['dose'], rt['dose']],
                        mode='lines+markers',
                        name="Radiotherapy",
                        line=dict(color='purple', width=4)
                    ),
                    row=2, col=1
                )
            
            if 'immunotherapy' in therapy:
                immuno = therapy['immunotherapy']
                fig.add_trace(
                    go.Scatter(
                        x=[immuno['start_time'], immuno['end_time']],
                        y=[immuno['dose'], immuno['dose']],
                        mode='lines+markers',
                        name="Immunotherapy",
                        line=dict(color='orange', width=4),
                        yaxis='y2'
                    ),
                    row=2, col=1, secondary_y=True
                )
        
        # Response distribution (fourth subplot)
        response_data = []
        for scenario, values in scenario_data.items():
            for y_vals in values['y']:
                if len(y_vals) > 1:
                    # Handle both pandas Series and numpy arrays
                    initial_val = y_vals.iloc[0] if hasattr(y_vals, 'iloc') else y_vals[0]
                    final_val = y_vals.iloc[-1] if hasattr(y_vals, 'iloc') else y_vals[-1]
                    reduction = (initial_val - final_val) / initial_val * 100 if initial_val > 0 else 0
                    response_data.append({'scenario': scenario, 'reduction': reduction})
        
        if response_data:
            df_response = pd.DataFrame(response_data)
            for scenario in df_response['scenario'].unique():
                scenario_responses = df_response[df_response['scenario'] == scenario]['reduction']
                fig.add_trace(
                    go.Histogram(
                        x=scenario_responses,
                        name=f"{scenario.title()}",
                        opacity=0.7,
                        nbinsx=10,
                        marker_color=colors.get(scenario, '#1f77b4')
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title_text="Comprehensive Cohort Tumor Dynamics Analysis",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (days)", row=1, col=1)
        fig.update_xaxes(title_text="Time (days)", row=1, col=2)
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Tumor Reduction (%)", row=2, col=2)
        
        fig.update_yaxes(title_text="Tumor Volume", row=1, col=1)
        fig.update_yaxes(title_text="Tumor Volume", row=1, col=2)
        fig.update_yaxes(title_text="Radiation Dose (Gy)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)

    def render_cohort_statistical_analysis(self, detailed_results, metrics):
        """Render statistical analysis of cohort results."""
        st.write("#### Statistical Analysis & Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Box Plot Analysis")
            
            # Create box plot data
            box_data = []
            for scenario, values in metrics.items():
                for value in values:
                    box_data.append({'Scenario': scenario.title(), 'Response': value})
            
            if box_data:
                df_box = pd.DataFrame(box_data)
                fig_box = px.box(
                    df_box, 
                    x='Scenario', 
                    y='Response',
                    title="Response Distribution by Scenario",
                    color='Scenario'
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            st.write("##### Statistical Tests")
            
            # Perform basic statistical tests
            scenario_names = list(metrics.keys())
            if len(scenario_names) >= 2:
                from scipy import stats
                
                st.write("**Pairwise Comparisons (t-test p-values):**")
                comparison_results = []
                
                for i, scenario1 in enumerate(scenario_names):
                    for scenario2 in scenario_names[i+1:]:
                        if len(metrics[scenario1]) > 1 and len(metrics[scenario2]) > 1:
                            t_stat, p_value = stats.ttest_ind(metrics[scenario1], metrics[scenario2])
                            comparison_results.append({
                                'Comparison': f"{scenario1.title()} vs {scenario2.title()}",
                                'p-value': f"{p_value:.4f}",
                                'Significant': "Yes" if p_value < 0.05 else "No"
                            })
                
                if comparison_results:
                    comp_df = pd.DataFrame(comparison_results)
                    st.dataframe(comp_df, use_container_width=True)
                
                # ANOVA if more than 2 groups
                if len(scenario_names) > 2:
                    scenario_values = [metrics[s] for s in scenario_names if len(metrics[s]) > 1]
                    if len(scenario_values) > 2:
                        f_stat, p_anova = stats.f_oneway(*scenario_values)
                        st.write(f"**ANOVA F-statistic:** {f_stat:.3f}")
                        st.write(f"**ANOVA p-value:** {p_anova:.4f}")
                        st.write(f"**Overall significance:** {'Yes' if p_anova < 0.05 else 'No'}")

    def render_cohort_treatment_response_analysis(self, detailed_results):
        """Render treatment response analysis."""
        st.write("#### Treatment Response Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Response Categories")
            
            # Categorize responses
            response_categories = {
                'Excellent (>75% reduction)': 0,
                'Good (50-75% reduction)': 0,
                'Moderate (25-50% reduction)': 0,
                'Poor (<25% reduction)': 0,
                'Progressive (tumor growth)': 0
            }
            
            for scenario, patients in detailed_results.items():
                for patient in patients:
                    sim_results = patient['simulation_results']
                    
                    # Calculate response
                    if 'tumor_cells' in sim_results:
                        tumor_data = sim_results['tumor_cells']
                    else:
                        tumor_data = sim_results.get('tumor_volume', [])
                    
                    if len(tumor_data) > 1:
                        # Handle both pandas Series and numpy arrays
                        initial = tumor_data.iloc[0] if hasattr(tumor_data, 'iloc') else tumor_data[0]
                        final = tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') else tumor_data[-1]
                        
                        initial = initial if initial > 0 else 1
                        reduction = (initial - final) / initial * 100
                        
                        if reduction > 75:
                            response_categories['Excellent (>75% reduction)'] += 1
                        elif reduction > 50:
                            response_categories['Good (50-75% reduction)'] += 1
                        elif reduction > 25:
                            response_categories['Moderate (25-50% reduction)'] += 1
                        elif reduction > 0:
                            response_categories['Poor (<25% reduction)'] += 1
                        else:
                            response_categories['Progressive (tumor growth)'] += 1
            
            # Display as pie chart
            fig_response = go.Figure(data=[go.Pie(
                labels=list(response_categories.keys()),
                values=list(response_categories.values()),
                hole=0.3
            )])
            fig_response.update_layout(title="Treatment Response Categories")
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            st.write("##### Time to Response")
            
            # Calculate time to 50% reduction for each patient
            time_to_response = {}
            
            for scenario, patients in detailed_results.items():
                time_to_response[scenario] = []
                
                for patient in patients:
                    sim_results = patient['simulation_results']
                    
                    if 'tumor_cells' in sim_results:
                        tumor_data = sim_results['tumor_cells']
                    else:
                        tumor_data = sim_results.get('tumor_volume', [])
                    
                    time_data = sim_results.get('time', [])
                    
                    if len(tumor_data) > 1 and len(time_data) > 1:
                        # Handle both pandas Series and numpy arrays
                        initial = tumor_data.iloc[0] if hasattr(tumor_data, 'iloc') else tumor_data[0]
                        target = initial * 0.5  # 50% of initial
                        
                        # Find first time point where tumor <= target
                        for i, vol in enumerate(tumor_data):
                            vol_val = vol if not hasattr(vol, 'iloc') else vol
                            time_val = time_data.iloc[i] if hasattr(time_data, 'iloc') else time_data[i]
                            if vol_val <= target:
                                time_to_response[scenario].append(time_val)
                                break
                        else:
                            # If never reached 50% reduction
                            time_to_response[scenario].append(None)
            
            # Plot time to response
            time_data = []
            for scenario, times in time_to_response.items():
                valid_times = [t for t in times if t is not None]
                for t in valid_times:
                    time_data.append({'Scenario': scenario.title(), 'Time to 50% Reduction': t})
            
            if time_data:
                df_time = pd.DataFrame(time_data)
                fig_time = px.box(
                    df_time, 
                    x='Scenario', 
                    y='Time to 50% Reduction',
                    title="Time to 50% Tumor Reduction"
                )
                st.plotly_chart(fig_time, use_container_width=True)

    def render_cohort_advanced_metrics(self, results):
        """Render advanced metrics and analysis."""
        st.write("#### Advanced Metrics & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Survival Analysis")
            
            # Define "survival" as maintaining tumor reduction
            survival_data = []
            
            for patient_id, patient_result in results.items():
                if 'error' in patient_result:
                    continue
                
                sim_results = patient_result['simulation_results']
                scenario = patient_result['scenario']
                
                # Simplified survival: tumor stays below 50% of initial for >30 days
                if 'tumor_cells' in sim_results:
                    tumor_data = sim_results['tumor_cells']
                else:
                    tumor_data = sim_results.get('tumor_volume', [])
                
                time_data = sim_results.get('time', [])
                
                if len(tumor_data) > 1 and len(time_data) > 1:
                    initial = tumor_data[0]
                    threshold = initial * 0.5
                    
                    # Count days below threshold after day 30
                    survival_days = 0
                    for i, (time, vol) in enumerate(zip(time_data, tumor_data)):
                        if time > 30 and vol < threshold:
                            survival_days += 1
                    
                    survival_data.append({
                        'Patient': patient_id,
                        'Scenario': scenario.title(),
                        'Survival Days': survival_days,
                        'Total Days': len([t for t in time_data if t > 30])
                    })
            
            if survival_data:
                df_survival = pd.DataFrame(survival_data)
                df_survival['Survival Rate'] = df_survival['Survival Days'] / df_survival['Total Days']
                
                fig_survival = px.scatter(
                    df_survival,
                    x='Total Days',
                    y='Survival Rate',
                    color='Scenario',
                    title="Tumor Control Probability",
                    hover_data=['Patient']
                )
                st.plotly_chart(fig_survival, use_container_width=True)
        
        with col2:
            st.write("##### Resistance Analysis")
            
            # Analyze treatment resistance patterns
            resistance_data = []
            
            for patient_id, patient_result in results.items():
                if 'error' in patient_result:
                    continue
                
                sim_results = patient_result['simulation_results']
                scenario = patient_result['scenario']
                
                if 'tumor_cells' in sim_results:
                    tumor_data = sim_results['tumor_cells']
                else:
                    tumor_data = sim_results.get('tumor_volume', [])
                
                if len(tumor_data) > 10:  # Need enough data points
                    # Look for resistance (tumor growth after initial reduction)
                    min_vol = min(tumor_data)
                    min_idx = list(tumor_data).index(min_vol)
                    
                    if min_idx < len(tumor_data) - 5:  # Check if there's growth after minimum
                        final_segment = tumor_data[min_idx:]
                        # Handle pandas Series properly
                        final_val = final_segment.iloc[-1] if hasattr(final_segment, 'iloc') else final_segment[-1]
                        growth_trend = final_val > min_vol * 1.2  # 20% growth from minimum
                        
                        resistance_data.append({
                            'Patient': patient_id,
                            'Scenario': scenario.title(),
                            'Resistance': 'Yes' if growth_trend else 'No',
                            'Min Volume': min_vol,
                            'Final Volume': tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') else tumor_data[-1]
                        })
            
            if resistance_data:
                df_resistance = pd.DataFrame(resistance_data)
                
                # Count resistance by scenario
                resistance_counts = df_resistance.groupby(['Scenario', 'Resistance']).size().reset_index(name='Count')
                
                fig_resistance = px.bar(
                    resistance_counts,
                    x='Scenario',
                    y='Count',
                    color='Resistance',
                    title="Treatment Resistance by Scenario",
                    barmode='group'
                )
                st.plotly_chart(fig_resistance, use_container_width=True)

    def render_enhanced_cohort_results(self, selected_patients, model_type, comparison_metric):
        """Render enhanced cohort results with spatial and temporal distributions like single patient mode."""
        results = st.session_state.cohort_simulation_results
        
        # Create main tabs for different analysis types
        overview_tab, spatial_tab, temporal_tab, individual_tab = st.tabs([
            "📊 Overview & Statistics", 
            "🗺️ Spatial Analysis", 
            "📈 Temporal Dynamics", 
            "👤 Individual Results"
        ])
        
        with overview_tab:
            # Keep the existing comprehensive dashboard
            self.display_cohort_simulation_results(comparison_metric)
        
        with spatial_tab:
            self.render_cohort_spatial_analysis(results, model_type)
        
        with temporal_tab:
            self.render_cohort_temporal_analysis(results, model_type)
        
        with individual_tab:
            self.render_cohort_individual_results(results, model_type)

    def render_cohort_spatial_analysis(self, results, model_type):
        """Render spatial analysis for cohort results."""
        st.write("#### Spatial Distribution Analysis")
        
        if model_type == "PDE":
            self.render_cohort_pde_spatial(results)
        elif model_type == "ABM":
            self.render_cohort_abm_spatial(results)
        else:  # ODE
            st.info("🔬 ODE models focus on temporal dynamics. Switch to PDE or ABM models for spatial analysis.")
            
    def render_cohort_pde_spatial(self, results):
        """Render PDE spatial analysis for cohort."""
        st.write("##### PDE Spatial Fields - Final States")
        
        # Collect spatial data from all patients with improved detection
        spatial_data = []
        for patient_id, patient_result in results.items():
            if 'error' not in patient_result and 'simulation_results' in patient_result:
                sim_data = patient_result['simulation_results']
                
                # Check for spatial model data in different possible locations
                spatial_model = None
                if 'spatial_model' in sim_data:
                    spatial_model = sim_data['spatial_model']
                elif hasattr(sim_data, 'spatial_model'):
                    spatial_model = sim_data.spatial_model
                
                if spatial_model is not None:
                    spatial_data.append({
                        'patient_id': patient_id,
                        'scenario': patient_result.get('scenario', 'unknown'),
                        'model': spatial_model
                    })
        
        if not spatial_data:
            st.warning("No spatial data available. This may be due to:")
            st.info("• PDE simulations not including spatial model data in results\n• Check that PDE model includes spatial_model in returned data\n• Verify PDE simulation completed successfully")
            
            # Show what data is actually available for debugging
            st.write("**Available simulation data keys:**")
            for patient_id, patient_result in list(results.items())[:3]:  # Show first 3 patients
                if 'error' not in patient_result and 'simulation_results' in patient_result:
                    st.write(f"Patient {patient_id}: {list(patient_result['simulation_results'].keys())}")
            return
            
        # Display spatial statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tumor Density Distributions**")
            # Create average tumor density heatmap
            if len(spatial_data) > 0:
                # Average tumor density across all patients
                try:
                    avg_tumor = np.zeros_like(spatial_data[0]['model'].tumor_density)
                    for data in spatial_data:
                        avg_tumor += data['model'].tumor_density
                    avg_tumor /= len(spatial_data)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=avg_tumor,
                        colorscale='Reds',
                        colorbar=dict(title="Density")
                    ))
                    fig.update_layout(
                        title="Average Tumor Density (All Patients)",
                        xaxis_title="X Position", 
                        yaxis_title="Y Position",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Spatial visualization not available for this PDE model configuration.")
        
        with col2:
            st.write("**Drug Distribution Patterns**")
            if len(spatial_data) > 0:
                try:
                    # Average drug distribution
                    avg_drug = np.zeros_like(spatial_data[0]['model'].drug_concentration)
                    for data in spatial_data:
                        avg_drug += data['model'].drug_concentration
                    avg_drug /= len(spatial_data)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=avg_drug,
                        colorscale='Blues',
                        colorbar=dict(title="Concentration")
                    ))
                    fig.update_layout(
                        title="Average Drug Concentration",
                        xaxis_title="X Position",
                        yaxis_title="Y Position", 
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Drug distribution visualization not available.")

    def render_cohort_abm_spatial(self, results):
        """Render ABM spatial analysis for cohort."""
        st.write("##### ABM Spatial Cell Distributions")
        
        # Collect ABM spatial data
        spatial_data = []
        for patient_id, patient_result in results.items():
            if 'error' not in patient_result and 'simulation_results' in patient_result:
                sim_data = patient_result['simulation_results']
                if 'final_positions' in sim_data:
                    spatial_data.append({
                        'patient_id': patient_id,
                        'scenario': patient_result.get('scenario', 'unknown'),
                        'positions': sim_data['final_positions']
                    })
        
        if not spatial_data:
            st.info("ABM spatial data not available. This requires model-specific implementation.")
            return
        
        # Show available spatial statistics
        st.write("**Cell Distribution Statistics**")
        stats_data = []
        for data in spatial_data:
            if 'tumor_cells' in data['positions']:
                tumor_count = len(data['positions']['tumor_cells'])
                stats_data.append({
                    'Patient': data['patient_id'][:12],
                    'Scenario': data['scenario'], 
                    'Final Tumor Cells': tumor_count
                })
        
        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)

    def render_cohort_temporal_analysis(self, results, model_type):
        """Render temporal dynamics analysis for cohort."""
        st.write("#### Temporal Dynamics Analysis")
        
        # Collect time series data
        time_series_data = []
        for patient_id, patient_result in results.items():
            if 'error' not in patient_result and 'simulation_results' in patient_result:
                sim_data = patient_result['simulation_results']
                time_series_data.append({
                    'patient_id': patient_id,
                    'scenario': patient_result.get('scenario', 'unknown'),
                    'time': sim_data.get('time', []),
                    'tumor_volume': sim_data.get('tumor_volume', sim_data.get('tumor_cells', [])),
                    'immune_cells': sim_data.get('immune_cells', []),
                    'drug_concentration': sim_data.get('total_drug', sim_data.get('drug_concentration', [])),
                    'oxygen_level': sim_data.get('avg_oxygen', [])
                })
        
        if not time_series_data:
            st.warning("No temporal data available.")
            return
        
        # Multi-patient time series visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tumor Volume Evolution**")
            fig = go.Figure()
            
            scenarios = list(set([data['scenario'] for data in time_series_data]))
            colors = {'standard': '#1f77b4', 'aggressive': '#d62728', 'responsive': '#2ca02c', 'resistant': '#ff7f0e'}
            
            for data in time_series_data:
                if len(data['time']) > 0 and len(data['tumor_volume']) > 0:
                    fig.add_trace(go.Scatter(
                        x=data['time'], 
                        y=data['tumor_volume'],
                        mode='lines',
                        name=f"{data['patient_id'][:8]} ({data['scenario']})",
                        line=dict(color=colors.get(data['scenario'], '#888888'), width=1),
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title="Individual Patient Tumor Trajectories",
                xaxis_title="Time (days)",
                yaxis_title="Tumor Volume",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Treatment Response Patterns**")
            # Show immune cell dynamics
            fig = go.Figure()
            
            for data in time_series_data:
                if len(data['time']) > 0 and len(data['immune_cells']) > 0:
                    fig.add_trace(go.Scatter(
                        x=data['time'],
                        y=data['immune_cells'],
                        mode='lines',
                        name=f"{data['patient_id'][:8]}",
                        line=dict(width=1),
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title="Immune Cell Dynamics",
                xaxis_title="Time (days)", 
                yaxis_title="Immune Cell Count",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_cohort_individual_results(self, results, model_type):
        """Render detailed individual patient results."""
        st.write("#### Individual Patient Analysis")
        
        # Patient selection
        patient_options = [pid for pid in results.keys() if 'error' not in results[pid]]
        if not patient_options:
            st.warning("No successful simulation results available.")
            return
            
        selected_patient = st.selectbox("Select Patient for Detailed Analysis:", patient_options)
        
        if selected_patient:
            patient_result = results[selected_patient]
            sim_data = patient_result['simulation_results']
            
            st.write(f"##### Results for {selected_patient}")
            
            # Display patient info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Scenario", patient_result.get('scenario', 'unknown').title())
            with col2:
                if 'tumor_volume' in sim_data and len(sim_data['tumor_volume']) > 1:
                    tumor_data = sim_data['tumor_volume']
                    initial_vol = tumor_data.iloc[0] if hasattr(tumor_data, 'iloc') else tumor_data[0]
                    final_vol = tumor_data.iloc[-1] if hasattr(tumor_data, 'iloc') else tumor_data[-1]
                    reduction = (initial_vol - final_vol) / initial_vol * 100 if initial_vol > 0 else 0
                    st.metric("Volume Reduction", f"{reduction:.1f}%")
            with col3:
                if 'time' in sim_data and len(sim_data['time']) > 0:
                    time_data = sim_data['time']
                    duration = time_data.iloc[-1] if hasattr(time_data, 'iloc') else time_data[-1]
                    st.metric("Simulation Duration", f"{duration:.0f} days")
            
            # Show the same detailed visualizations as single patient mode
            st.write("**Detailed Analysis (Same as Single Patient Mode)**")
            
            try:
                if model_type == "PDE" and 'spatial_model' in sim_data:
                    # Wrap data in expected format for PDE results
                    pde_results = {'results': sim_data, 'spatial_model': sim_data.get('spatial_model')}
                    self.render_pde_results(pde_results)
                elif model_type == "ABM":
                    # ABM results function now handles both formats
                    self.render_abm_results(sim_data)
                else:  # ODE
                    # ODE results function now handles both formats
                    self.render_ode_results(sim_data)
            except Exception as e:
                st.warning(f"Detailed visualization not available: {str(e)}")
                st.info("Basic simulation data is available in the Overview tab.")

    def display_protocol_comparison_results(self, comparison_results):
        """Display treatment protocol comparison results."""
        st.write("### Treatment Protocol Comparison Results")
        
        # Create comparison table
        comparison_data = []
        
        for protocol_name, protocol_results in comparison_results.items():
            for patient_id, result in protocol_results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    comparison_data.append({
                        'Protocol': protocol_name,
                        'Patient': patient_id,
                        'Final Tumor Burden': f"{metrics.get('final_tumor_burden', 0):.2e}",
                        'Tumor Reduction': f"{metrics.get('tumor_reduction', 0):.1%}",
                        'Control Probability': f"{metrics.get('tumor_control_probability', 0):.3f}"
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Plot protocol comparison
            self.plot_protocol_comparison_chart(comparison_df)
        else:
            st.warning("No comparison results available.")

    def plot_protocol_comparison_chart(self, comparison_df):
        """Plot protocol comparison chart."""
        import plotly.express as px
        
        # Convert tumor reduction to numeric
        comparison_df['Tumor Reduction (%)'] = comparison_df['Tumor Reduction'].str.rstrip('%').astype(float)
        
        fig = px.box(
            comparison_df,
            x='Protocol',
            y='Tumor Reduction (%)',
            title='Treatment Protocol Effectiveness Comparison',
            color='Protocol'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def export_cohort_results(self):
        """Export cohort analysis results."""
        if "cohort_simulation_results" not in st.session_state:
            st.warning("No cohort simulation results to export. Run a cohort simulation first.")
            return
        
        results = st.session_state.cohort_simulation_results
        
        # Prepare export data
        export_data = []
        for patient_id, patient_result in results.items():
            if 'error' not in patient_result:
                sim_results = patient_result['simulation_results']
                export_data.append({
                    'patient_id': patient_id,
                    'scenario': patient_result['scenario'],
                    'simulation_results': sim_results,
                    'fitted_parameters': patient_result['fitted_parameters']
                })
        
        # Convert to JSON for download
        export_json = json.dumps(export_data, default=str, indent=2)
        
        st.download_button(
            label="Download Cohort Results (JSON)",
            data=export_json,
            file_name=f"cohort_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("Cohort results prepared for download!")

    def render_parameter_sensitivity_analysis(self):
        """Render parameter sensitivity analysis controls."""
        st.write("**Parameter Sensitivity Analysis**")
        st.write("See how sensitive each patient scenario is to treatment parameter changes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            param_to_vary = st.selectbox(
                "Parameter to Vary:",
                ["Radiation Dose", "Treatment Duration", "Immunotherapy Strength", "Treatment Start Time"]
            )
            
        with col2:
            variation_range = st.slider("Variation Range (%)", 10, 100, 50)
        
        if st.button("Run Sensitivity Analysis", key="sensitivity"):
            self.run_parameter_sensitivity_analysis(param_to_vary, variation_range)

    def render_dose_response_analysis(self):
        """Render dose-response analysis controls."""
        st.write("**Dose-Response Analysis**")
        st.write("Analyze how different radiation or drug doses affect treatment outcomes.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dose_type = st.selectbox("Dose Type:", ["Radiation", "Immunotherapy", "Both"])
        
        with col2:
            min_dose = st.number_input("Min Dose", 0.5, 10.0, 1.0, 0.5)
            max_dose = st.number_input("Max Dose", 1.0, 20.0, 5.0, 0.5)
        
        with col3:
            dose_steps = st.number_input("Number of Dose Levels", 3, 10, 5)
        
        if st.button("Run Dose-Response Analysis", key="dose_response"):
            self.run_dose_response_analysis(dose_type, min_dose, max_dose, dose_steps)

    def render_timing_optimization_analysis(self):
        """Render treatment timing optimization controls."""
        st.write("**Treatment Timing Optimization**")
        st.write("Find optimal treatment start times and schedules for different patients.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            timing_param = st.selectbox(
                "Timing Parameter:",
                ["Treatment Start Day", "Treatment Duration", "Inter-treatment Interval", "Sequential vs Concurrent"]
            )
        
        with col2:
            optimization_metric = st.selectbox(
                "Optimization Goal:",
                ["Maximize Tumor Reduction", "Minimize Final Burden", "Maximize Survival", "Minimize Toxicity"]
            )
        
        if st.button("Run Timing Optimization", key="timing"):
            self.run_timing_optimization_analysis(timing_param, optimization_metric)

    def render_custom_parameter_sweep(self):
        """Render custom parameter sweep controls."""
        st.write("**Custom Parameter Sweep**")
        st.write("Define your own parameter ranges to explore treatment space.")
        
        # Parameter 1
        col1, col2, col3 = st.columns(3)
        with col1:
            param1_name = st.selectbox("Parameter 1:", ["radiation_dose", "immuno_dose", "treatment_start", "treatment_duration"])
        with col2:
            param1_min = st.number_input("Min Value", 0.1, 100.0, 1.0, key="p1_min")
        with col3:
            param1_max = st.number_input("Max Value", 1.0, 200.0, 10.0, key="p1_max")
        
        # Parameter 2
        col1, col2, col3 = st.columns(3)
        with col1:
            param2_name = st.selectbox("Parameter 2:", ["treatment_duration", "radiation_dose", "immuno_dose", "treatment_start"])
        with col2:
            param2_min = st.number_input("Min Value", 0.1, 100.0, 5.0, key="p2_min")
        with col3:
            param2_max = st.number_input("Max Value", 1.0, 200.0, 50.0, key="p2_max")
        
        sweep_resolution = st.slider("Sweep Resolution", 3, 15, 5)
        
        if st.button("Run Parameter Sweep", key="param_sweep"):
            self.run_custom_parameter_sweep(
                param1_name, param1_min, param1_max,
                param2_name, param2_min, param2_max,
                sweep_resolution
            )

    def run_parameter_sensitivity_analysis(self, param_to_vary, variation_range):
        """Run parameter sensitivity analysis on the cohort."""
        try:
            cohort = st.session_state.patient_cohort
            
            # Define parameter variations
            variations = []
            base_multiplier = 1.0
            var_range = variation_range / 100.0  # Convert percentage to decimal if not already done
            
            for i in range(5):  # 5 different parameter values
                multiplier = base_multiplier + (var_range * (i - 2) / 2)  # -var_range to +var_range
                # Ensure positive values
                multiplier = max(0.1, multiplier)
                variations.append(multiplier)
            
            with st.spinner(f"Running sensitivity analysis for {param_to_vary}..."):
                sensitivity_results = {}
                
                for patient_id, patient_data in cohort.items():
                    patient_results = {}
                    
                    # Get scenario safely
                    if isinstance(patient_data, dict) and 'scenario' in patient_data:
                        scenario = str(patient_data['scenario'])
                    elif hasattr(patient_data, 'get'):
                        scenario = str(patient_data.get('scenario', 'unknown'))
                    elif isinstance(patient_data, pd.DataFrame) and 'scenario' in patient_data.columns:
                        scenario = str(patient_data['scenario'].iloc[0])
                    else:
                        scenario = 'standard'
                    
                    for i, multiplier in enumerate(variations):
                        # Create modified therapy schedule
                        therapy_schedule = self.create_modified_therapy_schedule(param_to_vary, multiplier)
                        
                        try:
                            # Run ODE simulation with modified parameters
                            result = self.run_ode_simulation_for_patient_with_schedule(
                                patient_data, therapy_schedule, 120
                            )
                            
                            # Calculate outcome metric
                            if 'tumor_cells' in result.columns:
                                initial_vol = result['tumor_cells'].iloc[0]
                                final_vol = result['tumor_cells'].iloc[-1]
                            elif 'tumor_volume' in result.columns:
                                initial_vol = result['tumor_volume'].iloc[0] 
                                final_vol = result['tumor_volume'].iloc[-1]
                            else:
                                # Fallback - use first available numeric column
                                numeric_cols = result.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 1:  # Skip time column
                                    col_name = numeric_cols[1]  # Use second numeric column (first is usually time)
                                    initial_vol = result[col_name].iloc[0]
                                    final_vol = result[col_name].iloc[-1]
                                else:
                                    raise ValueError("No suitable tumor data found in simulation results")
                            
                            reduction = (initial_vol - final_vol) / initial_vol * 100
                            
                            patient_results[f"{param_to_vary}_{multiplier:.2f}"] = {
                                'reduction': reduction,
                                'final_burden': final_vol,
                                'parameter_value': multiplier
                            }
                            
                        except Exception as e:
                            st.warning(f"Simulation failed for patient {patient_id}, multiplier {multiplier:.2f}: {str(e)}")
                            continue
                    
                    sensitivity_results[patient_id] = {
                        'results': patient_results,
                        'scenario': scenario
                    }
                
                if sensitivity_results:
                    st.session_state.sensitivity_results = sensitivity_results
                    st.success("Sensitivity analysis completed!")
                    
                    # Display results
                    self.display_sensitivity_results(param_to_vary, sensitivity_results)
                    return sensitivity_results
                else:
                    st.error("No valid sensitivity analysis results generated.")
                    return None
                    
        except Exception as e:
            st.error(f"Sensitivity analysis failed: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return None

    def run_dose_response_analysis(self, dose_type, min_dose, max_dose, dose_steps):
        """Run dose-response analysis on the cohort."""
        cohort = st.session_state.patient_cohort
        
        # Create dose levels
        dose_levels = np.linspace(min_dose, max_dose, dose_steps)
        
        with st.spinner(f"Running dose-response analysis for {dose_type}..."):
            dose_response_results = {}
            
            for patient_id, patient_data in cohort.items():
                patient_results = {}
                scenario = patient_data['scenario'].iloc[0]
                
                for dose in dose_levels:
                    # Create therapy schedule with specific dose
                    therapy_schedule = self.create_dose_specific_therapy_schedule(dose_type, dose)
                    
                    try:
                        result = self.run_ode_simulation_for_patient_with_schedule(
                            patient_data, therapy_schedule, 120
                        )
                        
                        # Calculate metrics
                        initial_vol = result['tumor_cells'].iloc[0] if hasattr(result['tumor_cells'], 'iloc') else result['tumor_cells'][0]
                        final_vol = result['tumor_cells'].iloc[-1] if hasattr(result['tumor_cells'], 'iloc') else result['tumor_cells'][-1]
                        reduction = (initial_vol - final_vol) / initial_vol * 100
                        
                        patient_results[f"dose_{dose:.1f}"] = {
                            'dose': dose,
                            'reduction': reduction,
                            'final_burden': final_vol
                        }
                        
                    except Exception as e:
                        st.error(f"Dose-response analysis failed for {patient_id}: {str(e)}")
                
                dose_response_results[patient_id] = {
                    'results': patient_results,
                    'scenario': scenario
                }
            
            st.session_state.dose_response_results = dose_response_results
            st.success("Dose-response analysis completed!")
            
            # Display results
            self.display_dose_response_results(dose_type, dose_response_results)

    def run_timing_optimization_analysis(self, timing_param, optimization_metric):
        """Run treatment timing optimization analysis."""
        try:
            cohort = st.session_state.patient_cohort
            
            # Define timing variations based on parameter
            if timing_param == "Treatment Start Day":
                timing_values = [7, 14, 21, 30, 45, 60]
            elif timing_param == "Treatment Duration":
                timing_values = [14, 21, 30, 42, 56, 70]
            else:
                timing_values = [5, 10, 15, 20, 25, 30]
            
            with st.spinner(f"Running timing optimization for {timing_param}..."):
                timing_results = {}
                
                for patient_id, patient_data in cohort.items():
                    patient_results = {}
                    
                    # Get scenario safely
                    if isinstance(patient_data, dict) and 'scenario' in patient_data:
                        scenario = patient_data['scenario']
                    elif hasattr(patient_data, 'get'):
                        scenario = patient_data.get('scenario', 'unknown')
                    elif isinstance(patient_data, pd.DataFrame) and 'scenario' in patient_data.columns:
                        scenario = patient_data['scenario'].iloc[0]
                    else:
                        scenario = 'standard'
                    
                    best_result = None
                    # Initialize based on whether we want to maximize or minimize
                    if "Reduction" in optimization_metric or "reduction" in optimization_metric.lower():
                        best_value = float('-inf')  # Maximizing reduction
                    elif "Burden" in optimization_metric or "burden" in optimization_metric.lower():
                        best_value = float('inf')   # Minimizing burden
                    else:
                        best_value = float('-inf')  # Default to maximizing
                    
                    for timing_val in timing_values:
                        # Create therapy schedule with specific timing
                        therapy_schedule = self.create_timing_specific_therapy_schedule(timing_param, timing_val)
                        
                        try:
                            result = self.run_ode_simulation_for_patient_with_schedule(
                                patient_data, therapy_schedule, 150
                            )
                            
                            # Calculate optimization metric
                            if 'tumor_cells' in result.columns:
                                initial_vol = result['tumor_cells'].iloc[0]
                                final_vol = result['tumor_cells'].iloc[-1]
                            elif 'tumor_volume' in result.columns:
                                initial_vol = result['tumor_volume'].iloc[0] 
                                final_vol = result['tumor_volume'].iloc[-1]
                            else:
                                # Fallback - use first available numeric column
                                numeric_cols = result.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 1:  # Skip time column
                                    col_name = numeric_cols[1]  # Use second numeric column
                                    initial_vol = result[col_name].iloc[0]
                                    final_vol = result[col_name].iloc[-1]
                                else:
                                    raise ValueError("No suitable tumor data found in simulation results")
                            
                            reduction = (initial_vol - final_vol) / initial_vol * 100
                            
                            # Determine if we want to maximize or minimize based on metric name
                            if "Reduction" in optimization_metric or "reduction" in optimization_metric.lower():
                                metric_value = reduction  # Higher reduction is better
                                is_maximizing = True
                            elif "Burden" in optimization_metric or "burden" in optimization_metric.lower():
                                metric_value = final_vol  # Lower final burden is better
                                is_maximizing = False
                            else:
                                metric_value = reduction  # Default to reduction
                                is_maximizing = True
                            
                            patient_results[f"timing_{timing_val}"] = {
                                'timing_value': timing_val,
                                'reduction': reduction,
                                'final_burden': final_vol,
                                'metric_value': metric_value
                            }
                            
                            # Track best result
                            if (is_maximizing and metric_value > best_value) or \
                               (not is_maximizing and metric_value < best_value):
                                best_value = metric_value
                                best_result = timing_val
                                
                        except Exception as e:
                            st.warning(f"Timing optimization failed for patient {patient_id}, timing {timing_val}: {str(e)}")
                            continue
                    
                    timing_results[patient_id] = {
                        'results': patient_results,
                        'scenario': scenario,
                        'optimal_timing': best_result,
                        'optimal_value': best_value
                    }
                
                if timing_results:
                    st.session_state.timing_results = timing_results
                    st.success("Timing optimization completed!")
                    
                    # Display results
                    self.display_timing_optimization_results(timing_param, optimization_metric, timing_results)
                    return timing_results
                else:
                    st.error("No valid timing optimization results generated.")
                    return None
                    
        except Exception as e:
            st.error(f"Timing optimization failed: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return None

    def run_custom_parameter_sweep(self, param1_name, param1_min, param1_max, param2_name, param2_min, param2_max, resolution):
        """Run custom parameter sweep analysis."""
        cohort = st.session_state.patient_cohort
        
        # Create parameter grids
        param1_values = np.linspace(param1_min, param1_max, resolution)
        param2_values = np.linspace(param2_min, param2_max, resolution)
        
        with st.spinner("Running custom parameter sweep..."):
            sweep_results = {}
            
            for patient_id, patient_data in cohort.items():
                scenario = patient_data['scenario'].iloc[0]
                param_grid = np.zeros((resolution, resolution))
                
                for i, p1_val in enumerate(param1_values):
                    for j, p2_val in enumerate(param2_values):
                        # Create therapy schedule with specific parameters
                        therapy_schedule = self.create_custom_therapy_schedule(
                            param1_name, p1_val, param2_name, p2_val
                        )
                        
                        try:
                            result = self.run_ode_simulation_for_patient_with_schedule(
                                patient_data, therapy_schedule, 120
                            )
                            
                            # Calculate outcome
                            initial_vol = result['tumor_cells'].iloc[0]
                            final_vol = result['tumor_cells'].iloc[-1]
                            reduction = (initial_vol - final_vol) / initial_vol * 100
                            
                            param_grid[i, j] = reduction
                            
                        except:
                            param_grid[i, j] = 0  # Default for failed simulations
                
                sweep_results[patient_id] = {
                    'grid': param_grid,
                    'scenario': scenario,
                    'param1_values': param1_values,
                    'param2_values': param2_values
                }
            
            st.session_state.sweep_results = {
                'results': sweep_results,
                'param1_name': param1_name,
                'param2_name': param2_name
            }
            st.success("Parameter sweep completed!")
            
            # Display results
            self.display_parameter_sweep_results(sweep_results, param1_name, param2_name)

    def create_modified_therapy_schedule(self, param_to_vary, multiplier):
        """Create therapy schedule with modified parameter."""
        base_schedule = {
            "radiotherapy": {"start_time": 30, "end_time": 65, "dose": 2.0},
            "immunotherapy": {"start_time": 35, "end_time": 95, "dose": 1.0}
        }
        
        if param_to_vary == "Radiation Dose":
            base_schedule["radiotherapy"]["dose"] *= multiplier
        elif param_to_vary == "Treatment Duration":
            duration_change = int((multiplier - 1) * 20)
            base_schedule["radiotherapy"]["end_time"] += duration_change
            base_schedule["immunotherapy"]["end_time"] += duration_change
        elif param_to_vary == "Immunotherapy Strength":
            base_schedule["immunotherapy"]["dose"] *= multiplier
        elif param_to_vary == "Treatment Start Time":
            time_change = int((multiplier - 1) * 15)
            base_schedule["radiotherapy"]["start_time"] += time_change
            base_schedule["immunotherapy"]["start_time"] += time_change
        
        return base_schedule

    def create_dose_specific_therapy_schedule(self, dose_type, dose):
        """Create therapy schedule with specific dose."""
        schedule = {
            "radiotherapy": {"start_time": 30, "end_time": 65, "dose": 2.0},
            "immunotherapy": {"start_time": 35, "end_time": 95, "dose": 1.0}
        }
        
        if dose_type == "Radiation":
            schedule["radiotherapy"]["dose"] = dose
        elif dose_type == "Immunotherapy":
            schedule["immunotherapy"]["dose"] = dose
        elif dose_type == "Both":
            schedule["radiotherapy"]["dose"] = dose
            schedule["immunotherapy"]["dose"] = dose
        
        return schedule

    def create_timing_specific_therapy_schedule(self, timing_param, timing_val):
        """Create therapy schedule with specific timing."""
        schedule = {
            "radiotherapy": {"start_time": 30, "end_time": 65, "dose": 2.0},
            "immunotherapy": {"start_time": 35, "end_time": 95, "dose": 1.0}
        }
        
        if timing_param == "Treatment Start Day":
            schedule["radiotherapy"]["start_time"] = timing_val
            schedule["immunotherapy"]["start_time"] = timing_val + 5
        elif timing_param == "Treatment Duration":
            schedule["radiotherapy"]["end_time"] = schedule["radiotherapy"]["start_time"] + timing_val
            schedule["immunotherapy"]["end_time"] = schedule["immunotherapy"]["start_time"] + timing_val
        
        return schedule

    def create_custom_therapy_schedule(self, param1_name, param1_val, param2_name, param2_val):
        """Create therapy schedule with custom parameters."""
        schedule = {
            "radiotherapy": {"start_time": 30, "end_time": 65, "dose": 2.0},
            "immunotherapy": {"start_time": 35, "end_time": 95, "dose": 1.0}
        }
        
        # Apply parameter 1
        if param1_name == "radiation_dose":
            schedule["radiotherapy"]["dose"] = param1_val
        elif param1_name == "immuno_dose":
            schedule["immunotherapy"]["dose"] = param1_val
        elif param1_name == "treatment_start":
            schedule["radiotherapy"]["start_time"] = param1_val
            schedule["immunotherapy"]["start_time"] = param1_val + 5
        elif param1_name == "treatment_duration":
            schedule["radiotherapy"]["end_time"] = schedule["radiotherapy"]["start_time"] + param1_val
        
        # Apply parameter 2
        if param2_name == "radiation_dose":
            schedule["radiotherapy"]["dose"] = param2_val
        elif param2_name == "immuno_dose":
            schedule["immunotherapy"]["dose"] = param2_val
        elif param2_name == "treatment_start":
            schedule["radiotherapy"]["start_time"] = param2_val
            schedule["immunotherapy"]["start_time"] = param2_val + 5
        elif param2_name == "treatment_duration":
            schedule["immunotherapy"]["end_time"] = schedule["immunotherapy"]["start_time"] + param2_val
        
        return schedule

    def run_ode_simulation_for_patient_with_schedule(self, patient_data, therapy_schedule, simulation_time):
        """Run ODE simulation with custom therapy schedule."""
        from models.ode_model import TumorImmuneODE
        
        model = TumorImmuneODE()
        
        # Initial conditions based on patient data
        initial_volume = patient_data['tumor_volume'].iloc[0]
        initial_conditions = np.array([
            initial_volume * 1e6,  # Convert to cells
            1e5,  # Initial immune cells
            0,    # Initial drug concentration
            100,  # Initial oxygen
            0     # Initial radiation dose
        ])
        
        time_points = np.linspace(0, simulation_time, int(simulation_time) + 1)
        results = model.simulate(initial_conditions, time_points, therapy_schedule)
        
        return results

    def display_sensitivity_results(self, param_to_vary, sensitivity_results):
        """Display parameter sensitivity analysis results."""
        st.write(f"### Sensitivity Analysis Results - {param_to_vary}")
        
        # Create sensitivity plot
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {'standard': '#1f77b4', 'aggressive': '#d62728', 'responsive': '#2ca02c', 'resistant': '#ff7f0e'}
        
        for patient_id, patient_result in sensitivity_results.items():
            scenario = patient_result['scenario']
            
            # Ensure scenario is a string
            if hasattr(scenario, 'iloc') and len(scenario) > 0:
                scenario = str(scenario.iloc[0])
            elif hasattr(scenario, '__iter__') and not isinstance(scenario, str):
                scenario = str(list(scenario)[0]) if scenario else 'unknown'
            else:
                scenario = str(scenario)
            
            results = patient_result['results']
            
            param_values = [r['parameter_value'] for r in results.values()]
            reductions = [r['reduction'] for r in results.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=param_values,
                    y=reductions,
                    mode='lines+markers',
                    name=f"{patient_id} ({scenario})",
                    line=dict(color=colors.get(scenario, '#1f77b4'))
                )
            )
        
        fig.update_layout(
            title=f"Sensitivity to {param_to_vary}",
            xaxis_title=f"{param_to_vary} (Multiplier)",
            yaxis_title="Tumor Volume Reduction (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def display_dose_response_results(self, dose_type, dose_response_results):
        """Display dose-response analysis results."""
        st.write(f"### Dose-Response Analysis Results - {dose_type}")
        
        # Create dose-response plot
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {'standard': '#1f77b4', 'aggressive': '#d62728', 'responsive': '#2ca02c', 'resistant': '#ff7f0e'}
        
        for patient_id, patient_result in dose_response_results.items():
            scenario = patient_result['scenario']
            
            # Ensure scenario is a string
            if hasattr(scenario, 'iloc') and len(scenario) > 0:
                scenario = str(scenario.iloc[0])
            elif hasattr(scenario, '__iter__') and not isinstance(scenario, str):
                scenario = str(list(scenario)[0]) if scenario else 'unknown'
            else:
                scenario = str(scenario)
            
            results = patient_result['results']
            
            doses = [r['dose'] for r in results.values()]
            reductions = [r['reduction'] for r in results.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=doses,
                    y=reductions,
                    mode='lines+markers',
                    name=f"{patient_id} ({scenario})",
                    line=dict(color=colors.get(scenario, '#1f77b4'))
                )
            )
        
        fig.update_layout(
            title=f"Dose-Response Curves - {dose_type}",
            xaxis_title=f"{dose_type} Dose",
            yaxis_title="Tumor Volume Reduction (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def display_timing_optimization_results(self, timing_param, optimization_metric, timing_results):
        """Display timing optimization results."""
        st.write(f"### Timing Optimization Results - {timing_param}")
        
        # Show optimal timings
        optimal_data = []
        for patient_id, patient_result in timing_results.items():
            # Clean up scenario display
            scenario = patient_result['scenario']
            if hasattr(scenario, 'iloc') and len(scenario) > 0:
                scenario_clean = str(scenario.iloc[0])
            elif hasattr(scenario, '__iter__') and not isinstance(scenario, str):
                scenario_clean = str(list(scenario)[0]) if scenario else 'unknown'
            else:
                scenario_clean = str(scenario)
            
            # Handle optimal timing display
            optimal_timing = patient_result.get('optimal_timing', 'None')
            optimal_value = patient_result.get('optimal_value', float('inf'))
            
            # Format optimal value better
            if optimal_value == float('inf') or optimal_value == float('-inf'):
                optimal_value_str = "No optimum found"
            elif optimal_value is None:
                optimal_value_str = "No data"
            else:
                optimal_value_str = f"{optimal_value:.2f}%"
            
            # Format optimal timing better
            if optimal_timing is None or optimal_timing == 'None':
                optimal_timing_str = "No optimum found"
            else:
                optimal_timing_str = f"{optimal_timing:.1f}"
            
            optimal_data.append({
                'Patient': patient_id,
                'Scenario': scenario_clean,
                'Optimal Timing': optimal_timing_str,
                'Optimal Value': optimal_value_str
            })
        
        optimal_df = pd.DataFrame(optimal_data)
        st.write("**Optimal Timing by Patient:**")
        st.dataframe(optimal_df, use_container_width=True)
        
        # Create timing comparison plot
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        colors = {'standard': '#1f77b4', 'aggressive': '#d62728', 'responsive': '#2ca02c', 'resistant': '#ff7f0e'}
        
        for patient_id, patient_result in timing_results.items():
            scenario = patient_result['scenario']
            
            # Ensure scenario is a string
            if hasattr(scenario, 'iloc') and len(scenario) > 0:
                scenario = str(scenario.iloc[0])
            elif hasattr(scenario, '__iter__') and not isinstance(scenario, str):
                scenario = str(list(scenario)[0]) if scenario else 'unknown'
            else:
                scenario = str(scenario)
            
            results = patient_result['results']
            
            timings = [r['timing_value'] for r in results.values()]
            metrics = [r['metric_value'] for r in results.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=timings,
                    y=metrics,
                    mode='lines+markers',
                    name=f"{patient_id} ({scenario})",
                    line=dict(color=colors.get(scenario, '#1f77b4'))
                )
            )
        
        fig.update_layout(
            title=f"Timing Optimization - {timing_param}",
            xaxis_title=timing_param,
            yaxis_title=optimization_metric,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def display_parameter_sweep_results(self, sweep_results, param1_name, param2_name):
        """Display parameter sweep results as heatmaps."""
        st.write(f"### Parameter Sweep Results - {param1_name} vs {param2_name}")
        
        # Create heatmaps for each patient scenario
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        scenarios = list(set(result['scenario'] for result in sweep_results.values()))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{scenario.title()} Scenario" for scenario in scenarios[:4]],
            vertical_spacing=0.15
        )
        
        for i, scenario in enumerate(scenarios[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Find a patient from this scenario
            patient_result = None
            for result in sweep_results.values():
                if result['scenario'] == scenario:
                    patient_result = result
                    break
            
            if patient_result:
                fig.add_trace(
                    go.Heatmap(
                        z=patient_result['grid'],
                        x=patient_result['param2_values'],
                        y=patient_result['param1_values'],
                        colorscale='RdYlBu_r',
                        showscale=(i == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Treatment Parameter Optimization Heatmaps",
            height=800
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def run_timing_grid_search_optimization(self):
        """Run grid search optimization for timing parameters."""
        try:
            if "patient_cohort" not in st.session_state or not st.session_state.patient_cohort:
                return None
            
            cohort = st.session_state.patient_cohort
            results = {}
            
            # Grid search parameters
            start_times = np.linspace(10, 50, 5)  # Treatment start days
            dose_levels = np.linspace(0.5, 3.0, 4)  # Dose multipliers
            
            for patient_id, patient_data in cohort.items():
                patient_results = {}
                best_outcome = float('-inf')
                optimal_params = None
                
                for start_time in start_times:
                    for dose_level in dose_levels:
                        # Create therapy schedule
                        schedule = self.get_base_therapy_schedule()
                        schedule["radiotherapy"]["start_time"] = start_time
                        schedule["radiotherapy"]["dose"] *= dose_level
                        
                        try:
                            # Run simulation
                            sim_results = self.run_ode_simulation_for_patient_with_schedule(
                                patient_data, schedule, 120
                            )
                            
                            # Calculate outcome metric (tumor reduction)
                            initial_volume = sim_results['tumor_volume'][0] if len(sim_results['tumor_volume']) > 0 else 1
                            final_volume = sim_results['tumor_volume'].iloc[-1] if hasattr(sim_results['tumor_volume'], 'iloc') and len(sim_results['tumor_volume']) > 0 else (sim_results['tumor_volume'][-1] if len(sim_results['tumor_volume']) > 0 else 1)
                            reduction = ((initial_volume - final_volume) / initial_volume) * 100
                            
                            param_key = f"start_{start_time:.1f}_dose_{dose_level:.2f}"
                            patient_results[param_key] = {
                                'start_time': start_time,
                                'dose_level': dose_level,
                                'reduction': reduction,
                                'outcome': reduction
                            }
                            
                            if reduction > best_outcome:
                                best_outcome = reduction
                                optimal_params = {'start_time': start_time, 'dose_level': dose_level}
                                
                        except Exception as e:
                            continue
                
                results[patient_id] = {
                    'results': patient_results,
                    'optimal_params': optimal_params,
                    'best_outcome': best_outcome,
                    'scenario': 'grid_search'
                }
            
            return results
            
        except Exception as e:
            st.error(f"Grid search optimization failed: {str(e)}")
            return None

    def run_toxicity_optimization_analysis(self, optimization_method):
        """Run optimization to minimize treatment toxicity."""
        try:
            if "patient_cohort" not in st.session_state or not st.session_state.patient_cohort:
                return None
            
            cohort = st.session_state.patient_cohort
            results = {}
            
            for patient_id, patient_data in cohort.items():
                patient_results = {}
                best_toxicity = float('inf')
                optimal_dose = None
                
                # Test different dose levels for toxicity vs efficacy
                dose_levels = np.linspace(0.3, 2.0, 8)
                
                for dose_level in dose_levels:
                    try:
                        # Create therapy schedule with reduced dose
                        schedule = self.get_base_therapy_schedule()
                        schedule["radiotherapy"]["dose"] *= dose_level
                        if "chemotherapy" in schedule:
                            schedule["chemotherapy"]["dose"] *= dose_level
                        
                        # Run simulation
                        sim_results = self.run_ode_simulation_for_patient_with_schedule(
                            patient_data, schedule, 120
                        )
                        
                        # Calculate toxicity score (simplified model)
                        # Higher doses = higher toxicity, but better efficacy
                        toxicity_score = dose_level * 10  # Base toxicity
                        
                        # Calculate efficacy
                        initial_volume = sim_results['tumor_volume'][0] if len(sim_results['tumor_volume']) > 0 else 1
                        final_volume = sim_results['tumor_volume'].iloc[-1] if hasattr(sim_results['tumor_volume'], 'iloc') and len(sim_results['tumor_volume']) > 0 else (sim_results['tumor_volume'][-1] if len(sim_results['tumor_volume']) > 0 else 1)
                        efficacy = ((initial_volume - final_volume) / initial_volume) * 100
                        
                        # Combined score (minimize toxicity while maintaining efficacy)
                        if efficacy > 20:  # Minimum efficacy threshold
                            combined_score = toxicity_score - (efficacy * 0.5)
                        else:
                            combined_score = toxicity_score + 50  # Penalty for low efficacy
                        
                        patient_results[f"dose_{dose_level:.2f}"] = {
                            'dose_level': dose_level,
                            'toxicity_score': toxicity_score,
                            'efficacy': efficacy,
                            'combined_score': combined_score
                        }
                        
                        if combined_score < best_toxicity:
                            best_toxicity = combined_score
                            optimal_dose = dose_level
                            
                    except Exception as e:
                        continue
                
                results[patient_id] = {
                    'results': patient_results,
                    'optimal_dose': optimal_dose,
                    'best_score': best_toxicity,
                    'scenario': 'toxicity_optimization'
                }
            
            return results
            
        except Exception as e:
            st.error(f"Toxicity optimization failed: {str(e)}")
            return None

    def display_optimization_results(self, results, optimization_target, optimization_method):
        """Display optimization results with appropriate visualization."""
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        
        st.write(f"### {optimization_target} Results ({optimization_method})")
        
        if optimization_target == "Optimize Timing" and optimization_method == "Grid Search":
            self.display_grid_search_results(results)
        elif optimization_target == "Minimize Toxicity":
            self.display_toxicity_optimization_results(results)
        else:
            # Default display for tumor reduction optimization
            self.display_standard_optimization_results(results, optimization_target)

    def display_grid_search_results(self, results):
        """Display grid search optimization results."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Summary table
        summary_data = []
        for patient_id, result in results.items():
            if result['optimal_params']:
                summary_data.append({
                    'Patient': patient_id,
                    'Optimal Start Time': f"{result['optimal_params']['start_time']:.1f} days",
                    'Optimal Dose Level': f"{result['optimal_params']['dose_level']:.2f}x",
                    'Best Outcome': f"{result['best_outcome']:.1f}%"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.write("**Grid Search Optimal Parameters:**")
            st.dataframe(summary_df, use_container_width=True)
            
            # Create heatmap for first patient
            if len(results) > 0:
                first_patient = list(results.keys())[0]
                patient_results = results[first_patient]['results']
                
                start_times = sorted(list(set([r['start_time'] for r in patient_results.values()])))
                dose_levels = sorted(list(set([r['dose_level'] for r in patient_results.values()])))
                
                # Create outcome matrix
                outcome_matrix = np.zeros((len(dose_levels), len(start_times)))
                
                for i, dose in enumerate(dose_levels):
                    for j, start in enumerate(start_times):
                        for result in patient_results.values():
                            if abs(result['dose_level'] - dose) < 0.01 and abs(result['start_time'] - start) < 0.1:
                                outcome_matrix[i, j] = result['outcome']
                                break
                
                fig = go.Figure(data=go.Heatmap(
                    z=outcome_matrix,
                    x=[f"{s:.1f}" for s in start_times],
                    y=[f"{d:.2f}" for d in dose_levels],
                    colorscale='RdYlBu_r',
                    colorbar=dict(title="Tumor Reduction (%)")
                ))
                
                fig.update_layout(
                    title=f"Grid Search Results - Patient {first_patient}",
                    xaxis_title="Treatment Start Time (days)",
                    yaxis_title="Dose Level (multiplier)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

    def display_toxicity_optimization_results(self, results):
        """Display toxicity optimization results."""
        import plotly.graph_objects as go
        
        # Summary table
        summary_data = []
        for patient_id, result in results.items():
            if result['optimal_dose']:
                summary_data.append({
                    'Patient': patient_id,
                    'Optimal Dose': f"{result['optimal_dose']:.2f}x",
                    'Best Score': f"{result['best_score']:.1f}",
                    'Scenario': 'Toxicity Minimized'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.write("**Toxicity Optimization Results:**")
            st.dataframe(summary_df, use_container_width=True)
            
            # Plot toxicity vs efficacy trade-offs
            fig = go.Figure()
            
            for patient_id, result in results.items():
                if 'results' in result:
                    patient_results = result['results']
                    doses = [r['dose_level'] for r in patient_results.values()]
                    toxicities = [r['toxicity_score'] for r in patient_results.values()]
                    efficacies = [r['efficacy'] for r in patient_results.values()]
                    
                    fig.add_trace(go.Scatter(
                        x=toxicities,
                        y=efficacies,
                        mode='lines+markers',
                        name=f"Patient {patient_id}",
                        text=[f"Dose: {d:.2f}x" for d in doses],
                        hovertemplate="Toxicity: %{x}<br>Efficacy: %{y:.1f}%<br>%{text}"
                    ))
            
            fig.update_layout(
                title="Toxicity vs Efficacy Trade-offs",
                xaxis_title="Toxicity Score",
                yaxis_title="Efficacy (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def display_standard_optimization_results(self, results, optimization_target):
        """Display standard optimization results."""
        import matplotlib.pyplot as plt
        
        # Summary of optimal settings first
        st.write("### 🎯 Optimal Treatment Settings")
        
        optimal_summary = []
        for patient_id, result in results.items():
            if 'results' in result and result['results']:
                patient_results = result['results']
                
                # Find parameter with best outcome
                best_param = None
                best_value = float('-inf')
                best_outcome = None
                
                for param_key, data in patient_results.items():
                    if 'reduction' in data:
                        outcome = data['reduction']
                    elif 'metric_value' in data:
                        outcome = data['metric_value'] 
                    elif 'outcome' in data:
                        outcome = data['outcome']
                    else:
                        continue
                        
                    if outcome > best_value:
                        best_value = outcome
                        best_param = param_key
                        best_outcome = outcome
                
                if best_param and best_outcome is not None:
                    # Extract parameter details
                    param_data = patient_results[best_param]
                    param_details = []
                    
                    if 'timing_value' in param_data:
                        param_details.append(f"Timing: {param_data['timing_value']:.1f}")
                    if 'dose_level' in param_data:
                        param_details.append(f"Dose: {param_data['dose_level']:.2f}x")
                    if 'start_time' in param_data:
                        param_details.append(f"Start: {param_data['start_time']:.1f} days")
                    if 'parameter_value' in param_data:
                        param_details.append(f"Multiplier: {param_data['parameter_value']:.2f}")
                    
                    optimal_summary.append({
                        'Patient': f"Patient {patient_id}",
                        'Best Parameters': ", ".join(param_details) if param_details else best_param,
                        'Outcome': f"{best_outcome:.1f}%",
                        'Scenario': result.get('scenario', 'unknown')
                    })
        
        if optimal_summary:
            import pandas as pd
            summary_df = pd.DataFrame(optimal_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Show key insights
            if len(optimal_summary) > 1:
                outcomes = [float(s['Outcome'].replace('%', '')) for s in optimal_summary]
                best_overall = max(outcomes)
                worst_overall = min(outcomes)
                avg_outcome = sum(outcomes) / len(outcomes)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Patient Response", f"{best_overall:.1f}%")
                with col2:
                    st.metric("Average Response", f"{avg_outcome:.1f}%")
                with col3:
                    st.metric("Response Range", f"{best_overall - worst_overall:.1f}%")
        
        # Create visualization plots
        st.write("### 📊 Optimization Analysis")
        
        fig, axes = plt.subplots(1, min(len(results), 3), figsize=(5*min(len(results), 3), 5))
        if len(results) == 1:
            axes = [axes]
        elif len(results) > 3:
            st.info("Showing first 3 patients. Use individual patient analysis for detailed view of all patients.")
        
        plot_count = 0
        for patient_id, result in list(results.items())[:3]:  # Limit to 3 plots
            if 'results' in result and result['results']:
                patient_results = result['results']
                
                # Extract x and y values based on available data
                x_values = []
                y_values = []
                x_label = "Parameter Value"
                
                for param_key, data in patient_results.items():
                    if 'timing_value' in data:
                        x_values.append(data['timing_value'])
                        x_label = "Timing Parameter"
                    elif 'dose_level' in data:
                        x_values.append(data['dose_level'])
                        x_label = "Dose Level"
                    elif 'parameter_value' in data:
                        x_values.append(data['parameter_value'])
                        x_label = "Parameter Multiplier"
                    else:
                        x_values.append(plot_count)  # fallback
                    
                    if 'reduction' in data:
                        y_values.append(data['reduction'])
                    elif 'metric_value' in data:
                        y_values.append(data['metric_value'])
                    elif 'outcome' in data:
                        y_values.append(data['outcome'])
                    else:
                        y_values.append(0)
                
                if x_values and y_values:
                    axes[plot_count].plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
                    axes[plot_count].set_xlabel(x_label)
                    axes[plot_count].set_ylabel('Outcome (%)')
                    axes[plot_count].set_title(f'Patient {patient_id}')
                    axes[plot_count].grid(True, alpha=0.3)
                    
                    # Mark optimal point
                    if y_values:
                        best_idx = y_values.index(max(y_values))
                        axes[plot_count].scatter(x_values[best_idx], y_values[best_idx], 
                                               color='red', s=100, zorder=5, label='Optimal')
                        axes[plot_count].legend()
                
                plot_count += 1
        
        plt.tight_layout()
        st.pyplot(fig)

    def get_base_therapy_schedule(self):
        """Get base therapy schedule for optimization."""
        if hasattr(st.session_state, 'therapy_schedule') and st.session_state.therapy_schedule:
            return st.session_state.therapy_schedule.copy()
        else:
            # Default schedule
            return {
                "radiotherapy": {
                    "start_time": 30,
                    "end_time": 65,
                    "dose": 2.0,
                    "frequency": 1,
                    "duration": 1
                }
            }

    def render_therapy_controls(self):
        """Render therapy parameter controls."""
        therapy_schedule = {}

        # Radiotherapy
        st.write("**Radiotherapy**")
        rt_enabled = st.checkbox("Enable Radiotherapy", value=True)

        if rt_enabled:
            col1, col2 = st.columns(2)
            with col1:
                rt_start = st.number_input("RT Start Day", 0, 365, 30)
                rt_dose = st.number_input("RT Dose (Gy)", 0.5, 10.0, 2.0, 0.1)
            with col2:
                rt_end = st.number_input("RT End Day", rt_start, 365, max(65, rt_start + 7))
                rt_frequency = st.number_input("RT Frequency (days)", 1, 7, 1)

            therapy_schedule["radiotherapy"] = {
                "start_time": rt_start,
                "end_time": rt_end,
                "dose": rt_dose,
                "frequency": rt_frequency,
                "duration": 1,
            }

        # Immunotherapy
        st.write("**Immunotherapy**")
        immuno_enabled = st.checkbox("Enable Immunotherapy", value=False)

        if immuno_enabled:
            col1, col2 = st.columns(2)
            with col1:
                immuno_start = st.number_input("Immuno Start Day", 0, 365, 35)
                immuno_dose = st.number_input("Immuno Dose", 0.1, 5.0, 1.0, 0.1)
            with col2:
                immuno_end = st.number_input("Immuno End Day", immuno_start, 365, max(95, immuno_start + 7))
                immuno_frequency = st.number_input("Immuno Frequency (days)", 1, 28, 7)

            therapy_schedule["immunotherapy"] = {
                "start_time": immuno_start,
                "end_time": immuno_end,
                "dose": immuno_dose,
                "frequency": immuno_frequency,
                "duration": 1,
            }

        # Chemotherapy
        st.write("**Chemotherapy**")
        chemo_enabled = st.checkbox("Enable Chemotherapy", value=False)

        if chemo_enabled:
            col1, col2 = st.columns(2)
            with col1:
                chemo_start = st.number_input("Chemo Start Day", 0, 365, 20)
                chemo_dose = st.number_input("Chemo Dose", 0.1, 3.0, 1.5, 0.1)
            with col2:
                chemo_end = st.number_input("Chemo End Day", chemo_start, 365, max(50, chemo_start + 7))
                chemo_frequency = st.number_input("Chemo Frequency (days)", 1, 28, 7)

            therapy_schedule["chemotherapy"] = {
                "start_time": chemo_start,
                "end_time": chemo_end,
                "dose": chemo_dose,
                "frequency": chemo_frequency,
                "duration": 1,
            }

        st.session_state.therapy_schedule = therapy_schedule

    def run_simulation(self, total_time: int):
        """Run the selected simulation model."""
        if st.session_state.patient_data is None:
            st.error("Please load patient data first!")
            return

        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text(
                f"Initializing {st.session_state.current_model} simulation..."
            )
            progress_bar.progress(10)

            if st.session_state.current_model == "ODE":
                status_text.text("Running ODE simulation...")
                progress_bar.progress(30)
                results = self.run_ode_simulation(total_time)
                progress_bar.progress(80)

            elif st.session_state.current_model == "PDE":
                status_text.text("Running PDE simulation...")
                progress_bar.progress(30)
                results = self.run_pde_simulation(total_time)
                progress_bar.progress(80)

            elif st.session_state.current_model == "ABM":
                status_text.text("Running ABM simulation...")
                progress_bar.progress(30)
                results = self.run_abm_simulation(total_time)
                progress_bar.progress(80)

            status_text.text("Processing results...")
            progress_bar.progress(90)

            st.session_state.simulation_results[st.session_state.current_model] = (
                results
            )

            progress_bar.progress(100)
            status_text.text("✅ Simulation completed successfully!")

            # Auto-clear progress after 2 seconds
            import time

            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            st.success(f"{st.session_state.current_model} simulation completed!")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Simulation failed: {str(e)}")
            # Add detailed error information for debugging
            with st.expander("Error Details"):
                import traceback

                st.code(traceback.format_exc())

    def run_ode_simulation(self, total_time: int) -> Dict:
        """Run ODE simulation with current parameters from UI."""
        try:
            model = TumorImmuneODE()
            
            # 🔧 FIX: Get current parameters from session state sliders
            if hasattr(st.session_state, 'ode_tumor_growth'):
                model.params.update({
                    'r_tumor': st.session_state.get('ode_tumor_growth', 0.1),
                    'k_kill': 10**st.session_state.get('ode_immune_kill', -9),  # Convert from log scale
                    'alpha_rt': st.session_state.get('ode_radiation_sensitivity', 0.3),
                    'k_in': st.session_state.get('ode_drug_uptake', 0.5),
                    'IC50': st.session_state.get('ode_drug_ic50', 1.0),
                })

            # Get patient data for initial conditions
            if "patient_data" in st.session_state and st.session_state.patient_data is not None:
                patient_data = st.session_state.patient_data
                initial_volume = patient_data['tumor_volume'].iloc[0]
            else:
                initial_volume = 5.0  # Default

            # Initial conditions
            initial_conditions = np.array(
                [
                    initial_volume * 1e6,  # Convert to cells
                    1e5,  # Initial immune cells
                    0,  # No initial drug
                    100,  # Normal oxygen
                    0,  # No initial radiation
                ]
            )

            # Time points
            time_points = np.linspace(0, total_time, total_time + 1)

            # Run simulation
            results = model.simulate(
                initial_conditions, time_points, st.session_state.therapy_schedule
            )

            # Calculate metrics
            metrics = model.calculate_metrics(results)

            return {
                "model": "ODE",
                "results": results,
                "metrics": metrics,
                "parameters": model.params.copy(),
                "parameters_used": model.params.copy(),
                "success": True
            }
            
        except Exception as e:
            st.error(f"ODE simulation failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_pde_simulation(self, total_time: int) -> Dict:
        """Run PDE simulation with current parameters from UI."""
        try:
            # 🔧 FIX: Get parameters from session state
            pde_params = {}
            if hasattr(st.session_state, 'pde_tumor_diffusion'):
                pde_params.update({
                    'D_tumor': st.session_state.get('pde_tumor_diffusion', 0.01),
                    'D_drug': st.session_state.get('pde_drug_diffusion', 0.1),
                    'r_tumor': st.session_state.get('pde_growth_rate', 0.1),
                    'k_kill': st.session_state.get('pde_drug_kill_rate', 1.0),
                    'alpha_rt': st.session_state.get('pde_radiation_sensitivity', 0.3),
                })
            
            # Get grid parameters if available
            grid_x = st.session_state.get('pde_grid_x', 100)
            grid_y = st.session_state.get('pde_grid_y', 100)
            spatial_step = st.session_state.get('pde_spatial_step', 0.1)
            
            model = SpatialTumorPDE(grid_size=(grid_x, grid_y), spatial_step=spatial_step, parameters=pde_params)

            # Initialize conditions
            model.initialize_conditions()

            # Create spatial therapy schedule
            spatial_schedule = {}
            if "radiotherapy" in st.session_state.therapy_schedule:
                spatial_schedule["radiation"] = {
                    "start_time": st.session_state.therapy_schedule["radiotherapy"][
                        "start_time"
                    ],
                    "end_time": st.session_state.therapy_schedule["radiotherapy"][
                        "end_time"
                    ],
                    "center": (model.nx * model.dx / 2, model.ny * model.dy / 2),
                    "width": 5.0,
                    "dose": st.session_state.therapy_schedule["radiotherapy"]["dose"],
                }

            # Run simulation
            results = model.simulate(total_time, dt=0.1, therapy_schedule=spatial_schedule)

            return {
                "model": "PDE", 
                "results": results, 
                "spatial_model": model,
                "parameters_used": pde_params,
                "success": True
            }
            
        except Exception as e:
            st.error(f"PDE simulation failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_abm_simulation(self, total_time: int) -> Dict:
        """Run ABM simulation with performance optimization and current parameters from UI."""
        try:
            # Use optimized ABM if enabled
            use_optimized = st.session_state.get('abm_optimized', True)
            ignore_therapy = st.session_state.get('abm_ignore_therapy', False)
            
            # 🔧 FIX: Get ABM parameters from session state
            abm_grid_x = st.session_state.get('abm_grid_x', 100)
            abm_grid_y = st.session_state.get('abm_grid_y', 100)
            abm_grid_z = st.session_state.get('abm_grid_z', 50)
            tumor_radius = st.session_state.get('abm_tumor_radius', 10)
            immune_count = st.session_state.get('abm_immune_count', 50)
            
            abm_params = {}
            if hasattr(st.session_state, 'abm_division_rate'):
                abm_params.update({
                    'tumor_growth_rate': st.session_state.get('abm_division_rate', 0.015),  # Fixed parameter name
                    'immune_kill_rate': st.session_state.get('abm_immune_cytotoxicity', 0.12),  # Fixed parameter name
                    'drug_cytotoxicity': st.session_state.get('abm_drug_cytotoxicity', 0.06),
                    'radiation_sensitivity': st.session_state.get('abm_radiation_sensitivity', 0.3),
                })
            
            if use_optimized:
                model = TumorImmuneABM3DOptimized(grid_size=(abm_grid_x, abm_grid_y, abm_grid_z))
                
                # Update parameters if available
                if abm_params:
                    model.params.update(abm_params)
                
                model.initialize_tumor_optimized(center=(abm_grid_x//2, abm_grid_y//2, abm_grid_z//2), radius=tumor_radius)
                model.initialize_immune_cells_optimized(count=immune_count)
                # Snapshot initial grid for preview visualization
                try:
                    initial_grid = model.get_grid_visualization()
                except Exception:
                    initial_grid = None
                
                # Run optimized simulation using direct steps (simpler and more reliable)
                total_hours = int(total_time * 24)
                
                for hour in range(0, total_hours, 6):  # Every 6 hours for performance
                    therapy_input = {}
                    current_day = hour / 24
                    
                    # Apply therapy based on schedule unless preview is ignoring therapy
                    if (not ignore_therapy) and st.session_state.therapy_schedule:
                        for therapy_type, therapy_config in st.session_state.therapy_schedule.items():
                            if (therapy_config.get('start_time', 0) <= current_day <= 
                                therapy_config.get('end_time', float('inf'))):
                                
                                if therapy_type == 'radiotherapy':
                                    therapy_input['radiation'] = {
                                        'center': (abm_grid_x//2, abm_grid_y//2, abm_grid_z//2),
                                        'radius': 15,
                                        'dose': therapy_config.get('dose', 2.0)
                                    }
                                elif therapy_type == 'chemotherapy':
                                    therapy_input['drug_dose'] = therapy_config.get('dose', 1.0)
                    
                    model.step_optimized(dt=6.0, therapy_input=therapy_input)
                
                return {
                    "model": "ABM", 
                    "results": model.history, 
                    "abm_model": model, 
                    "optimized": True,
                    "parameters_used": abm_params,
                    "initial_grid": initial_grid,
                    "success": True
                }
            else:
                model = TumorImmuneABM3D(grid_size=(abm_grid_x, abm_grid_y, abm_grid_z), parameters=abm_params)

                # Initialize tumor and immune cells
                model.initialize_tumor(center=(abm_grid_x//2, abm_grid_y//2, abm_grid_z//2), radius=tumor_radius)
                model.initialize_immune_cells(count=immune_count)
                # Snapshot initial grid for preview visualization
                try:
                    initial_grid = model.get_grid_visualization()
                except Exception:
                    initial_grid = None

                # Convert days to hours for ABM
                total_hours = int(total_time * 24)

                # Run simulation
                for hour in range(total_hours):
                    therapy_input = {}

                    # Apply therapies based on schedule
                    current_day = hour / 24

                    if (not ignore_therapy) and ("radiotherapy" in st.session_state.therapy_schedule):
                        rt_config = st.session_state.therapy_schedule["radiotherapy"]
                        if rt_config["start_time"] <= current_day <= rt_config["end_time"]:
                            if hour % 24 == 12:  # Apply radiation at noon
                                therapy_input["radiation"] = {
                                    "center": (abm_grid_x//2, abm_grid_y//2, abm_grid_z//2),
                                    "radius": 20,
                                    "dose": rt_config["dose"],
                                }

                    if (not ignore_therapy) and ("chemotherapy" in st.session_state.therapy_schedule):
                        chemo_config = st.session_state.therapy_schedule["chemotherapy"]
                        if (
                            chemo_config["start_time"]
                            <= current_day
                            <= chemo_config["end_time"]
                        ):
                            therapy_input["drug_dose"] = (
                                chemo_config["dose"] / 24
                            )  # Hourly dose

                    model.step(dt=1.0, therapy_input=therapy_input)

                return {
                    "model": "ABM",
                    "results": model.history,
                    "final_grid": model.get_grid_visualization(),
                    "abm_model": model,
                    "optimized": False,
                    "parameters_used": abm_params,
                    "initial_grid": initial_grid,
                    "success": True
                }
        
        except Exception as e:
            st.error(f"ABM simulation failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def render_main_content(self):
        """Render the main content area."""
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Simulation", "📈 Analysis", "👥 Cohort", "📁 Export", "🏠 Dashboard"]
        )

        with tab1:
            self.render_simulation_tab()

        with tab2:
            self.render_analysis_tab()

        with tab3:
            self.render_cohort_tab()

        with tab4:
            self.render_export_tab()

        with tab5:
            self.render_dashboard_tab()

    def render_simulation_tab(self):
        """Render unified simulation interface for single and multiple patients."""
        st.markdown('<h2 class="sub-header">Simulation Control Center</h2>', unsafe_allow_html=True)
        
        # Simulation mode selection
        simulation_mode = st.radio(
            "Simulation Mode:",
            ["Single Patient", "Patient Cohort"],
            horizontal=True
        )
        
        if simulation_mode == "Single Patient":
            self.render_single_patient_simulation()
        else:
            self.render_cohort_simulation()

    def render_single_patient_simulation(self):
        """Render single patient simulation interface."""
        st.write("### Single Patient Simulation")
        
        # Check if patient data is available
        if st.session_state.patient_data is None or st.session_state.patient_data.empty:
            st.warning("⚠️ Please load patient data first using the sidebar controls.")
            return
        
        # Display patient info
        st.success("✅ Patient data loaded successfully")
        
        # Create two main sections: Treatment Parameters and Model Configuration
        tab1, tab2 = st.tabs(["💊 Treatment Parameters", "🔬 Model Configuration"])
        
        with tab1:
            st.write("### Treatment Protocol Configuration")
            st.info("💊 Configure the treatment protocol to be used in the simulation.")
            self.render_therapy_controls()
        
        with tab2:
            # Model and simulation controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Configuration**")
                # Use the global model from sidebar to avoid duplication
                selected_model = st.session_state.current_model
                st.info(f"🔬 Current Model: **{selected_model}** (set in sidebar)")
                
                # Debug info (can be removed later)
                if st.checkbox("Show Debug Info", key="debug_model"):
                    st.write(f"Debug - Session state current_model: {st.session_state.current_model}")
                    st.write(f"Debug - Selected model variable: {selected_model}")
                
                simulation_time = st.number_input("Simulation Time (days):", 30, 730, 180, key="main_sim_time")
                
            with col2:
                st.write("**Model Parameters**")
                if selected_model == "ODE":
                    st.info("💡 Systemic tumor-immune dynamics")
                    st.session_state.ode_tumor_growth = st.slider(
                        "Tumor Growth Rate", 0.01, 0.5, 
                        st.session_state.get('ode_tumor_growth', 0.1), 0.01, 
                        key="main_growth",
                        help="Intrinsic tumor growth rate (1/day)"
                    )
                    st.session_state.ode_immune_kill = st.slider(
                        "Immune Kill Rate (log)", -12.0, -6.0, 
                        st.session_state.get('ode_immune_kill', -9.0), 0.1, 
                        key="main_immune",
                        help="Immune cell killing efficiency (log scale)"
                    )
                    st.session_state.ode_radiation_sensitivity = st.slider(
                        "Radiation Sensitivity", 0.1, 1.0, 
                        st.session_state.get('ode_radiation_sensitivity', 0.3), 0.05, 
                        key="main_rad",
                        help="Radiation sensitivity coefficient"
                    )
                    st.session_state.ode_drug_uptake = st.slider(
                        "Drug Uptake Rate", 0.1, 2.0, 
                        st.session_state.get('ode_drug_uptake', 0.5), 0.1, 
                        key="main_drug_uptake",
                        help="Drug uptake rate constant"
                    )
                    st.session_state.ode_drug_ic50 = st.slider(
                        "Drug IC50", 0.1, 5.0, 
                        st.session_state.get('ode_drug_ic50', 1.0), 0.1, 
                        key="main_ic50",
                        help="Drug concentration for 50% effect"
                    )
                elif selected_model == "PDE":
                    st.info("💡 Spatial diffusion processes")
                    st.session_state.pde_grid_x = st.slider(
                        "Grid Size X", 50, 200, 
                        st.session_state.get('pde_grid_x', 100), 10, 
                        key="main_grid_x"
                    )
                    st.session_state.pde_grid_y = st.slider(
                        "Grid Size Y", 50, 200, 
                        st.session_state.get('pde_grid_y', 100), 10, 
                        key="main_grid_y"
                    )
                    st.session_state.pde_spatial_step = st.slider(
                        "Spatial Resolution (mm)", 0.05, 0.5, 
                        st.session_state.get('pde_spatial_step', 0.1), 0.05, 
                        key="main_spatial"
                    )
                    st.session_state.pde_tumor_diffusion = st.slider(
                        "Tumor Diffusion", 0.001, 0.1, 
                        st.session_state.get('pde_tumor_diffusion', 0.01), 0.001, 
                        key="main_tumor_diff"
                    )
                    st.session_state.pde_drug_diffusion = st.slider(
                        "Drug Diffusion", 0.01, 1.0, 
                        st.session_state.get('pde_drug_diffusion', 0.1), 0.01, 
                        key="main_drug_diff"
                    )
                    st.session_state.pde_growth_rate = st.slider(
                        "Growth Rate", 0.01, 0.5, 
                        st.session_state.get('pde_growth_rate', 0.1), 0.01, 
                        key="main_pde_growth"
                    )
                elif selected_model == "ABM":
                    st.info("💡 Cell-level interactions")
                    st.session_state.abm_grid_x = st.slider(
                        "Grid Size X", 50, 150, 
                        st.session_state.get('abm_grid_x', 80), 10, 
                        key="main_abm_x"
                    )
                    st.session_state.abm_grid_y = st.slider(
                        "Grid Size Y", 50, 150, 
                        st.session_state.get('abm_grid_y', 80), 10, 
                        key="main_abm_y"
                    )
                    st.session_state.abm_grid_z = st.slider(
                        "Grid Size Z", 20, 80, 
                        st.session_state.get('abm_grid_z', 40), 5, 
                        key="main_abm_z"
                    )
                    st.session_state.abm_tumor_radius = st.slider(
                        "Initial Tumor Radius", 3, 20, 
                        st.session_state.get('abm_tumor_radius', 10), 1, 
                        key="main_tumor_radius"
                    )
                    st.session_state.abm_immune_count = st.slider(
                        "Initial Immune Cells", 10, 200, 
                        st.session_state.get('abm_immune_count', 50), 10, 
                        key="main_immune_count"
                    )
                    st.session_state.abm_division_rate = st.slider(
                        "Division Rate", 0.001, 0.1, 
                        st.session_state.get('abm_division_rate', 0.015), 0.001, 
                        key="main_division_rate"
                    )
                    st.session_state.abm_immune_cytotoxicity = st.slider(
                        "Immune Cytotoxicity", 0.01, 0.5, 
                        st.session_state.get('abm_immune_cytotoxicity', 0.12), 0.01, 
                        key="main_immune_cyto"
                    )
                    st.session_state.abm_drug_cytotoxicity = st.slider(
                        "Drug Cytotoxicity", 0.01, 0.2, 
                        st.session_state.get('abm_drug_cytotoxicity', 0.06), 0.01, 
                        key="main_drug_cyto"
                    )
                    st.session_state.abm_ignore_therapy = st.checkbox(
                        "Ignore therapy during ABM run (preview)", 
                        value=st.session_state.get('abm_ignore_therapy', True),
                        help="Preview the ABM without applying radiotherapy/chemotherapy so visuals reflect intrinsic dynamics."
                    )
                
                # Add a reset button for parameters
                col_reset1, col_reset2 = st.columns([1, 1])
                with col_reset1:
                    if st.button("🔄 Reset Parameters", key="reset_params_main"):
                        # Clear parameter-related session state
                        param_keys = [k for k in st.session_state.keys() if k.startswith(('ode_', 'pde_', 'abm_'))]
                        for key in param_keys:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                with col_reset2:
                    if st.button("📄 Show Current Parameters", key="show_params_main"):
                        st.session_state.show_params = not st.session_state.get('show_params', False)
                
                # Show current parameters if requested
                if st.session_state.get('show_params', False):
                    st.write("**Current Parameters:**")
                    if selected_model == "ODE":
                        st.json({
                            "tumor_growth": st.session_state.get('ode_tumor_growth', 0.1),
                            "immune_kill_log": st.session_state.get('ode_immune_kill', -9),
                            "radiation_sensitivity": st.session_state.get('ode_radiation_sensitivity', 0.3),
                            "drug_uptake": st.session_state.get('ode_drug_uptake', 0.5),
                            "drug_ic50": st.session_state.get('ode_drug_ic50', 1.0)
                        })
                    elif selected_model == "PDE":
                        st.json({
                            "grid_x": st.session_state.get('pde_grid_x', 100),
                            "grid_y": st.session_state.get('pde_grid_y', 100),
                            "spatial_step": st.session_state.get('pde_spatial_step', 0.1),
                            "tumor_diffusion": st.session_state.get('pde_tumor_diffusion', 0.01),
                            "drug_diffusion": st.session_state.get('pde_drug_diffusion', 0.1),
                            "growth_rate": st.session_state.get('pde_growth_rate', 0.1)
                        })
                    elif selected_model == "ABM":
                        st.json({
                            "grid_x": st.session_state.get('abm_grid_x', 80),
                            "grid_y": st.session_state.get('abm_grid_y', 80),
                            "grid_z": st.session_state.get('abm_grid_z', 40),
                            "tumor_radius": st.session_state.get('abm_tumor_radius', 10),
                            "immune_count": st.session_state.get('abm_immune_count', 50),
                            "division_rate": st.session_state.get('abm_division_rate', 0.015),
                            "immune_cytotoxicity": st.session_state.get('abm_immune_cytotoxicity', 0.12),
                            "drug_cytotoxicity": st.session_state.get('abm_drug_cytotoxicity', 0.06)
                        })
            
            # Run simulation button
            if st.button("🚀 Run Simulation", type="primary", key="main_run_single"):
                st.session_state.current_model = selected_model
                self.run_simulation(simulation_time)
        
        # Display results if available
        if st.session_state.simulation_results and st.session_state.current_model in st.session_state.simulation_results:
            st.write("### Simulation Results")
            
            current_results = st.session_state.simulation_results[st.session_state.current_model]
            
            if st.session_state.current_model == "ODE":
                self.render_ode_results(current_results)
            elif st.session_state.current_model == "PDE":
                self.render_pde_results(current_results)
            elif st.session_state.current_model == "ABM":
                self.render_abm_results(current_results)

    def render_cohort_simulation(self):
        """Render cohort simulation interface."""
        st.write("### Patient Cohort Simulation")
        
        # Check if cohort is available
        if "patient_cohort" not in st.session_state or not st.session_state.patient_cohort:
            st.warning("⚠️ Please generate a patient cohort first using the sidebar controls.")
            return
        
        cohort = st.session_state.patient_cohort
        st.success(f"✅ Cohort loaded: {len(cohort)} patients")
        
        # Create tabs for Treatment Parameters and Simulation Configuration
        tab1, tab2 = st.tabs(["💊 Treatment Parameters", "🔬 Simulation Configuration"])
        
        with tab1:
            st.write("### Treatment Protocol Configuration")
            st.info("🔬 These treatment parameters will be applied to all patients in the cohort during simulation and analysis.")
            self.render_therapy_controls()
        
        with tab2:
            # Simulation configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Simulation Configuration**")
                # Use the cohort model from sidebar to avoid duplication
                cohort_model = st.session_state.get("cohort_model", "ODE")
                st.info(f"📊 Current Model: **{cohort_model}** (set in sidebar)")
                simulation_time = st.number_input("Simulation Time (days):", 30, 730, 120, key="cohort_sim_time")
                
                # Patient selection
                patient_options = ["All Patients"] + list(cohort.keys())
                selected_patients = st.multiselect(
                    "Select Patients:",
                    options=patient_options,
                    default=["All Patients"],
                    key="cohort_patients"
                )
                
                if "All Patients" in selected_patients:
                    selected_patients = list(cohort.keys())
            
            with col2:
                st.write("**Analysis Options**")
                
                comparison_metric = st.selectbox(
                    "Primary Metric:",
                    ["Tumor Volume Reduction", "Final Tumor Burden", "Treatment Response"],
                    key="cohort_metric"
                )
                
                parallel_execution = st.checkbox("Parallel Execution", value=True, 
                                               help="Run simulations in parallel for faster processing")
                
                export_results = st.checkbox("Auto-export Results", value=False,
                                           help="Automatically export results after simulation")
            
            # Simulation actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Run Cohort Simulation", type="primary", key="run_cohort_main"):
                    self.run_cohort_simulation(selected_patients, cohort_model, simulation_time)
            
            with col2:
                if st.button("🔬 Compare Protocols", key="compare_protocols_main"):
                    self.run_cohort_treatment_comparison(selected_patients)
            
            with col3:
                if st.button("📊 Parameter Analysis", key="param_analysis_main"):
                    # Redirect to Analysis tab with cohort context
                    st.info("💡 Switch to the **Analysis** tab for detailed parameter analysis.")
        
        # Display cohort simulation results
        if "cohort_simulation_results" in st.session_state and st.session_state.cohort_simulation_results:
            st.write("### Cohort Simulation Results")
            
            # Enhanced cohort results with detailed visualizations
            self.render_enhanced_cohort_results(selected_patients, cohort_model, comparison_metric)

    def render_cohort_tab(self):
        """Render dedicated cohort analysis tab."""
        st.markdown('<h2 class="sub-header">Patient Cohort Analysis</h2>', unsafe_allow_html=True)
        
        # Check if cohort exists
        if "patient_cohort" not in st.session_state or not st.session_state.patient_cohort:
            st.info("🔄 Generate a patient cohort using the sidebar to enable cohort analysis.")
            return
        
        # Display cohort overview
        self.render_cohort_display()
        
        # Parameter analysis section
        st.write("---")
        st.write("### Treatment Parameter Analysis")
        st.write("Explore how different treatment parameters affect various patient scenarios.")
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Parameter Sensitivity", "Dose Response", "Timing Optimization", "Custom Parameter Sweep"],
            key="main_analysis_type"
        )
        
        if analysis_type == "Parameter Sensitivity":
            self.render_parameter_sensitivity_analysis()
        elif analysis_type == "Dose Response":
            self.render_dose_response_analysis()
        elif analysis_type == "Timing Optimization":
            self.render_timing_optimization_analysis()
        elif analysis_type == "Custom Parameter Sweep":
            self.render_custom_parameter_sweep()

    def render_analysis_tab(self):
        """Render analysis tab with advanced treatment prediction tools."""
        st.markdown('<h2 class="sub-header">🧬 AI-Powered Treatment Optimizer</h2>', unsafe_allow_html=True)
        
        # Initialize treatment predictor if not already done
        if 'treatment_predictor' not in st.session_state:
            st.session_state.treatment_predictor = TreatmentPredictor()
        
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["🎯 Smart Treatment Optimizer", "🔬 Treatment Comparison", "📊 Sensitivity Analysis", "⚖️ Model Comparison"],
            horizontal=True,
            key="analysis_mode_selection"
        )

        if analysis_mode == "🎯 Smart Treatment Optimizer":
            self.render_smart_treatment_optimizer()
        elif analysis_mode == "🔬 Treatment Comparison":
            self.render_treatment_comparison_tool()
        elif analysis_mode == "📊 Sensitivity Analysis":
            self.render_sensitivity_analysis_tools()
        elif analysis_mode == "⚖️ Model Comparison":
            self.render_model_comparison_tools()

    def render_smart_treatment_optimizer(self):
        """Render the AI-powered smart treatment optimizer."""
        st.markdown("### 🎯 Smart Treatment Optimizer")
        st.markdown("""
        This advanced AI system analyzes patient characteristics and predicts the optimal treatment combination 
        to maximize outcomes while minimizing toxicity and considering quality of life.
        """)
        
        # Check data availability
        has_single = st.session_state.patient_data is not None and not st.session_state.patient_data.empty
        has_cohort = "patient_cohort" in st.session_state and st.session_state.patient_cohort
        
        if not has_single and not has_cohort:
            st.warning("📋 Please load patient data or generate a cohort to use the optimizer.")
            st.info("💡 **How to get started:**\n1. Go to the Simulation tab to load/create patient data\n2. Or use the sidebar to generate a patient cohort\n3. Return here for AI-powered treatment optimization")
            return
        
        # Patient selection
        if has_single and has_cohort:
            data_source = st.radio("Optimize for:", ["Current Patient", "Patient Cohort"], horizontal=True)
            if data_source == "Current Patient":
                selected_patients = {"current": st.session_state.patient_data}
            else:
                selected_patients = st.session_state.patient_cohort
        elif has_single:
            selected_patients = {"current": st.session_state.patient_data}
            st.info("🔍 Optimizing for current patient")
        else:
            selected_patients = st.session_state.patient_cohort
            st.info(f"🔍 Optimizing for cohort ({len(st.session_state.patient_cohort)} patients)")
        
        # Optimization goals
        st.markdown("#### 🎪 Optimization Goals")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tumor_control_weight = st.slider("Tumor Control", 0.0, 1.0, 0.4, 0.05, 
                                            help="Weight for tumor control probability")
        with col2:
            survival_weight = st.slider("Overall Survival", 0.0, 1.0, 0.3, 0.05,
                                       help="Weight for overall survival probability")
        with col3:
            quality_life_weight = st.slider("Quality of Life", 0.0, 1.0, 0.2, 0.05,
                                           help="Weight for quality of life score")
        with col4:
            toxicity_weight = st.slider("Avoid Toxicity", 0.0, 1.0, 0.1, 0.05,
                                       help="Weight for avoiding treatment toxicity")
        
        # Normalize weights
        total_weight = tumor_control_weight + survival_weight + quality_life_weight + toxicity_weight
        if total_weight > 0:
            optimization_goals = {
                'tumor_control_probability': tumor_control_weight / total_weight,
                'overall_survival_probability': survival_weight / total_weight,
                'quality_of_life_score': quality_life_weight / total_weight,
                'toxicity_score': -toxicity_weight / total_weight  # Negative to minimize
            }
        else:
            optimization_goals = {
                'tumor_control_probability': 0.4,
                'overall_survival_probability': 0.3,
                'quality_of_life_score': 0.2,
                'toxicity_score': -0.1
            }
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                train_on_cohort = st.checkbox("Train AI on cohort data", value=True,
                                             help="Use cohort data to train predictive models")
                consider_cost = st.checkbox("Consider treatment cost", value=False,
                                           help="Include treatment cost in optimization")
            with col2:
                optimization_method = st.selectbox("Optimization Method:", 
                                                  ["Differential Evolution", "Bayesian Optimization"], 
                                                  help="Algorithm for finding optimal treatment")
                max_treatments = st.selectbox("Max simultaneous treatments:", [1, 2, 3], index=2)
        
        # Run optimization
        if st.button("🚀 Find Optimal Treatment", type="primary"):
            with st.spinner("🧠 AI is analyzing patient data and optimizing treatment..."):
                
                # Train predictor if requested and cohort available
                if train_on_cohort and has_cohort:
                    try:
                        st.session_state.treatment_predictor.train_predictive_models(st.session_state.patient_cohort)
                        st.success("✅ AI models trained on cohort data")
                    except Exception as e:
                        st.warning(f"⚠️ Could not train on cohort: {str(e)}")
                
                # Optimize for each patient
                optimization_results = {}
                
                for patient_id, patient_data in selected_patients.items():
                    try:
                        # Add cost consideration if requested
                        if consider_cost:
                            optimization_goals['treatment_cost_relative'] = -0.1
                        
                        result = st.session_state.treatment_predictor.find_optimal_treatment(
                            patient_data, optimization_goals
                        )
                        optimization_results[patient_id] = result
                        
                    except Exception as e:
                        st.error(f"❌ Optimization failed for patient {patient_id}: {str(e)}")
                
                if optimization_results:
                    st.success(f"🎉 Optimization completed for {len(optimization_results)} patient(s)!")
                    self.display_optimization_results(optimization_results)
    
    def render_treatment_comparison_tool(self):
        """Render treatment comparison tool."""
        st.markdown("### 🔬 Treatment Protocol Comparison")
        st.markdown("""
        Compare different treatment protocols side-by-side to understand their predicted outcomes, 
        toxicity profiles, and cost-effectiveness for your patient(s).
        """)
        
        # Check data availability
        has_single = st.session_state.patient_data is not None and not st.session_state.patient_data.empty
        has_cohort = "patient_cohort" in st.session_state and st.session_state.patient_cohort
        
        if not has_single and not has_cohort:
            st.warning("📋 Please load patient data or generate a cohort first.")
            return
        
        # Patient selection
        if has_single and has_cohort:
            data_source = st.radio("Compare treatments for:", ["Current Patient", "Patient Cohort"], horizontal=True)
            if data_source == "Current Patient":
                selected_patients = {"current": st.session_state.patient_data}
            else:
                # For cohort, let user select specific patients
                patient_options = list(st.session_state.patient_cohort.keys())
                selected_ids = st.multiselect("Select patients:", patient_options, default=patient_options[:3])
                selected_patients = {pid: st.session_state.patient_cohort[pid] for pid in selected_ids}
        elif has_single:
            selected_patients = {"current": st.session_state.patient_data}
        else:
            patient_options = list(st.session_state.patient_cohort.keys())
            selected_ids = st.multiselect("Select patients:", patient_options, default=patient_options[:3])
            selected_patients = {pid: st.session_state.patient_cohort[pid] for pid in selected_ids}
        
        if not selected_patients:
            st.warning("Please select at least one patient.")
            return
        
        # Treatment protocol definition
        st.markdown("#### 🏥 Define Treatment Protocols")
        
        num_protocols = st.slider("Number of protocols to compare:", 2, 5, 3)
        
        treatment_protocols = []
        
        for i in range(num_protocols):
            with st.expander(f"Protocol {i+1}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Radiotherapy**")
                    rt_enabled = st.checkbox(f"Enable RT", key=f"rt_en_{i}")
                    rt_dose = st.slider(f"RT Dose (Gy)", 1.0, 8.0, 2.0, key=f"rt_dose_{i}") if rt_enabled else 0
                    rt_start = st.slider(f"RT Start (day)", 7, 45, 30, key=f"rt_start_{i}") if rt_enabled else 0
                    rt_duration = st.slider(f"RT Duration", 14, 56, 35, key=f"rt_dur_{i}") if rt_enabled else 0
                
                with col2:
                    st.markdown("**Immunotherapy**")
                    immuno_enabled = st.checkbox(f"Enable Immuno", key=f"im_en_{i}")
                    immuno_dose = st.slider(f"Immuno Dose", 0.5, 4.0, 1.0, key=f"im_dose_{i}") if immuno_enabled else 0
                    immuno_start = st.slider(f"Immuno Start", 14, 60, 35, key=f"im_start_{i}") if immuno_enabled else 0
                    immuno_duration = st.slider(f"Immuno Duration", 28, 84, 56, key=f"im_dur_{i}") if immuno_enabled else 0
                
                with col3:
                    st.markdown("**Chemotherapy**")
                    chemo_enabled = st.checkbox(f"Enable Chemo", key=f"ch_en_{i}")
                    chemo_dose = st.slider(f"Chemo Dose", 0.5, 3.0, 1.5, key=f"ch_dose_{i}") if chemo_enabled else 0
                    chemo_start = st.slider(f"Chemo Start", 7, 30, 20, key=f"ch_start_{i}") if chemo_enabled else 0
                    chemo_duration = st.slider(f"Chemo Duration", 21, 70, 42, key=f"ch_dur_{i}") if chemo_enabled else 0
                
                # Create treatment protocol
                protocol = {
                    'rt_enabled': int(rt_enabled),
                    'rt_dose': rt_dose,
                    'rt_start_day': rt_start,
                    'rt_duration': rt_duration,
                    'rt_frequency': 1,
                    'immuno_enabled': int(immuno_enabled),
                    'immuno_dose': immuno_dose,
                    'immuno_start_day': immuno_start,
                    'immuno_duration': immuno_duration,
                    'immuno_frequency': 14,
                    'chemo_enabled': int(chemo_enabled),
                    'chemo_dose': chemo_dose,
                    'chemo_start_day': chemo_start,
                    'chemo_duration': chemo_duration,
                    'chemo_frequency': 7,
                    'total_treatments': int(rt_enabled) + int(immuno_enabled) + int(chemo_enabled),
                    'treatment_intensity': (int(rt_enabled) * rt_dose * 0.3 + 
                                          int(immuno_enabled) * immuno_dose * 0.2 + 
                                          int(chemo_enabled) * chemo_dose * 0.5),
                    'treatment_duration_total': max(rt_duration if rt_enabled else 0,
                                                   immuno_duration if immuno_enabled else 0,
                                                   chemo_duration if chemo_enabled else 0)
                }
                treatment_protocols.append(protocol)
        
        # Compare treatments
        if st.button("🔍 Compare Treatments", type="primary"):
            with st.spinner("Comparing treatment protocols..."):
                
                comparison_results = {}
                
                for patient_id, patient_data in selected_patients.items():
                    try:
                        comparison_df = st.session_state.treatment_predictor.compare_treatments(
                            patient_data, treatment_protocols
                        )
                        comparison_results[patient_id] = comparison_df
                        
                    except Exception as e:
                        st.error(f"❌ Comparison failed for patient {patient_id}: {str(e)}")
                
                if comparison_results:
                    st.success("🎉 Treatment comparison completed!")
                    self.display_treatment_comparison_results(comparison_results)
    
    def display_optimization_results(self, optimization_results):
        """Display optimization results in a comprehensive format."""
        st.markdown("### 🎯 Optimization Results")
        
        for patient_id, result in optimization_results.items():
            with st.expander(f"🧑‍⚕️ Patient {patient_id} - Optimal Treatment Plan", expanded=True):
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 💊 Recommended Treatment")
                    optimal_treatment = result['optimal_treatment']
                    
                    # Display treatment components
                    active_treatments = []
                    
                    if optimal_treatment['rt_enabled']:
                        st.markdown(f"**🔬 Radiotherapy**")
                        st.write(f"• Dose: {optimal_treatment['rt_dose']:.1f} Gy")
                        st.write(f"• Start Day: {optimal_treatment['rt_start_day']}")
                        st.write(f"• Duration: {optimal_treatment['rt_duration']} days")
                        active_treatments.append("Radiotherapy")
                    
                    if optimal_treatment['immuno_enabled']:
                        st.markdown(f"**🛡️ Immunotherapy**")
                        st.write(f"• Dose: {optimal_treatment['immuno_dose']:.1f}")
                        st.write(f"• Start Day: {optimal_treatment['immuno_start_day']}")
                        st.write(f"• Duration: {optimal_treatment['immuno_duration']} days")
                        active_treatments.append("Immunotherapy")
                    
                    if optimal_treatment['chemo_enabled']:
                        st.markdown(f"**💉 Chemotherapy**")
                        st.write(f"• Dose: {optimal_treatment['chemo_dose']:.1f}")
                        st.write(f"• Start Day: {optimal_treatment['chemo_start_day']}")
                        st.write(f"• Duration: {optimal_treatment['chemo_duration']} days")
                        active_treatments.append("Chemotherapy")
                    
                    if not active_treatments:
                        st.info("💡 Optimal treatment: Watchful waiting / No active treatment")
                    else:
                        st.success(f"✅ Combination therapy: {' + '.join(active_treatments)}")
                
                with col2:
                    st.markdown("#### 📊 Predicted Outcomes")
                    outcomes = result['predicted_outcomes']
                    
                    # Create metrics display
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        tumor_control = outcomes.get('tumor_control_probability', 0) * 100
                        st.metric("Tumor Control", f"{tumor_control:.1f}%", 
                                 delta=f"+{tumor_control-50:.1f}%" if tumor_control > 50 else None)
                        
                        survival = outcomes.get('overall_survival_probability', 0) * 100
                        st.metric("Overall Survival", f"{survival:.1f}%",
                                 delta=f"+{survival-70:.1f}%" if survival > 70 else None)
                    
                    with col_b:
                        quality = outcomes.get('quality_of_life_score', 0) * 100
                        st.metric("Quality of Life", f"{quality:.1f}%",
                                 delta=f"+{quality-80:.1f}%" if quality > 80 else None)
                        
                        toxicity = outcomes.get('toxicity_score', 0) * 100
                        st.metric("Toxicity Risk", f"{toxicity:.1f}%",
                                 delta=f"-{20-toxicity:.1f}%" if toxicity < 20 else None,
                                 delta_color="inverse")
                
                # Patient characteristics
                st.markdown("#### 🧬 Patient Profile")
                patient_features = result['patient_features']
                
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    st.write(f"**Tumor Aggressiveness:** {patient_features['tumor_aggressiveness_score']:.1f}/100")
                    st.write(f"**Initial Volume:** {patient_features['initial_tumor_volume']:.1f} cm³")
                
                with col_p2:
                    st.write(f"**Treatment Resistance:** {patient_features['treatment_resistance_score']:.1f}/100")
                    st.write(f"**Growth Rate:** {patient_features['tumor_growth_rate']:.3f} /day")
                
                with col_p3:
                    st.write(f"**Immune Competence:** {patient_features['immune_competence_score']:.1f}/100")
                    st.write(f"**Baseline PSA:** {patient_features['baseline_psa']:.1f}")
                
                # Optimization score
                st.info(f"🎯 **Optimization Score:** {result['optimization_score']:.3f} (higher is better)")
    
    def display_treatment_comparison_results(self, comparison_results):
        """Display treatment comparison results."""
        st.markdown("### 🔬 Treatment Comparison Results")
        
        for patient_id, comparison_df in comparison_results.items():
            with st.expander(f"🧑‍⚕️ Patient {patient_id} Comparison", expanded=True):
                
                # Create interactive plot
                fig = st.session_state.treatment_predictor.plot_treatment_comparison(comparison_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                st.markdown("#### 📋 Detailed Comparison Table")
                
                # Select key columns for display
                display_columns = ['treatment_name', 'tumor_control_probability', 
                                 'overall_survival_probability', 'quality_of_life_score', 
                                 'toxicity_score']
                
                if all(col in comparison_df.columns for col in display_columns):
                    display_df = comparison_df[display_columns].copy()
                    
                    # Format percentages
                    for col in ['tumor_control_probability', 'overall_survival_probability', 
                               'quality_of_life_score', 'toxicity_score']:
                        if col in display_df.columns:
                            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
                    
                    # Rename columns for display
                    display_df.columns = ['Treatment', 'Tumor Control', 'Overall Survival', 
                                        'Quality of Life', 'Toxicity Risk']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Recommendation
                    best_idx = comparison_df['tumor_control_probability'].idxmax()
                    best_treatment = comparison_df.loc[best_idx, 'treatment_name']
                    st.success(f"🏆 **Recommended:** {best_treatment} (highest tumor control probability)")
                else:
                    st.dataframe(comparison_df)

    def render_sensitivity_analysis_tools(self):
        """Render sensitivity analysis tools."""
        st.write("### 📊 Sensitivity Analysis")
        st.info("Analyze how changes in treatment parameters affect patient outcomes.")
        
        # Check data availability
        has_single = (st.session_state.patient_data is not None and not st.session_state.patient_data.empty)
        has_cohort = ("patient_cohort" in st.session_state and st.session_state.patient_cohort)
        
        if not has_single and not has_cohort:
            st.warning("Please load patient data or generate a cohort to run sensitivity analysis.")
            return
        
        # Parameter selection
        param_to_vary = st.selectbox(
            "Parameter to Analyze:",
            ["Radiation Dose", "Treatment Duration", "Immunotherapy Strength", "Treatment Start Time"]
        )
        
        variation_range = st.slider("Variation Range (%)", 10, 100, 50)
        
        if st.button("🔍 Run Sensitivity Analysis", type="primary"):
            if has_cohort:
                results = self.run_parameter_sensitivity_analysis(param_to_vary, variation_range)
                if results:
                    self.display_sensitivity_results(results)
            else:
                st.info("Single patient sensitivity analysis would go here")
        """Render sensitivity analysis tools."""
        st.write("### Sensitivity Analysis")
        
        # Data source selection
        data_source = st.radio(
            "Data Source:",
            ["Current Patient", "Patient Cohort"],
            horizontal=True,
            key="analysis_data_source"
        )
        
        if data_source == "Current Patient":
            if st.session_state.patient_data is None or st.session_state.patient_data.empty:
                st.warning("Please load patient data first.")
                return
            st.info("💡 Single patient sensitivity analysis")
            
            # Single patient sensitivity controls
            col1, col2 = st.columns(2)
            with col1:
                param_to_vary = st.selectbox(
                    "Parameter to Vary:",
                    ["Radiation Dose", "Treatment Duration", "Immunotherapy Strength"],
                    key="single_param_analysis"
                )
            with col2:
                variation_range = st.slider("Variation Range (%)", 10, 100, 50, key="single_range_analysis")
            
            if st.button("Run Single Patient Sensitivity", key="single_sensitivity_analysis"):
                if st.session_state.patient_data is None or st.session_state.patient_data.empty:
                    st.warning("Please load patient data first.")
                    return
                
                with st.spinner("Running parameter sensitivity analysis..."):
                    # For single patient analysis, create a temporary single-patient cohort
                    original_cohort = st.session_state.get('patient_cohort', {})
                    
                    # Create temporary cohort with just current patient
                    st.session_state.patient_cohort = {0: st.session_state.patient_data}
                    
                    try:
                        # Run sensitivity analysis using existing method
                        results = self.run_parameter_sensitivity_analysis(
                            param_to_vary=param_to_vary,
                            variation_range=variation_range
                        )
                        
                        if results is not None:
                            st.success("Analysis completed!")
                            # Display results for single patient
                            patient_results = results.get(0, {}).get('results', {})
                            
                            if patient_results:
                                # Create visualization
                                import matplotlib.pyplot as plt
                                
                                # Prepare data for plotting
                                param_values = []
                                reductions = []
                                
                                for key, data in patient_results.items():
                                    param_values.append(data['parameter_value'])
                                    reductions.append(data['reduction'])
                                
                                # Sort by parameter value
                                sorted_data = sorted(zip(param_values, reductions))
                                param_values, reductions = zip(*sorted_data)
                                
                                # Create plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(param_values, reductions, marker='o', linewidth=2, markersize=6)
                                ax.set_xlabel(f'{param_to_vary} Multiplier')
                                ax.set_ylabel('Tumor Volume Reduction (%)')
                                ax.set_title(f'Parameter Sensitivity: {param_to_vary}')
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                
                                # Show summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Reduction", f"{max(reductions):.1f}%")
                                with col2:
                                    best_idx = reductions.index(max(reductions))
                                    st.metric("Optimal Multiplier", f"{param_values[best_idx]:.2f}")
                                with col3:
                                    st.metric("Range", f"{max(reductions) - min(reductions):.1f}%")
                            else:
                                st.error("No results generated from sensitivity analysis.")
                        else:
                            st.error("Sensitivity analysis failed.")
                    finally:
                        # Restore original cohort
                        if original_cohort:
                            st.session_state.patient_cohort = original_cohort
                        elif 'patient_cohort' in st.session_state:
                            del st.session_state.patient_cohort
                
        else:
            if "patient_cohort" not in st.session_state or not st.session_state.patient_cohort:
                st.warning("Please generate a patient cohort first.")
                return
            st.info("💡 Cohort-wide sensitivity analysis")
            
            # Cohort sensitivity controls
            col1, col2 = st.columns(2)
            with col1:
                param_to_vary = st.selectbox(
                    "Parameter to Vary:",
                    ["Radiation Dose", "Treatment Duration", "Immunotherapy Strength"],
                    key="cohort_param_analysis"
                )
            with col2:
                variation_range = st.slider("Variation Range (%)", 10, 100, 50, key="cohort_range_analysis")
            
            if st.button("Run Cohort Sensitivity", key="cohort_sensitivity_analysis"):
                if "patient_cohort" not in st.session_state or not st.session_state.patient_cohort:
                    st.warning("Please generate a patient cohort first.")
                    return
                
                with st.spinner("Running cohort-wide parameter sensitivity analysis..."):
                    # Run sensitivity analysis using existing method
                    results = self.run_parameter_sensitivity_analysis(
                        param_to_vary=param_to_vary,
                        variation_range=variation_range
                    )
                    
                    if results is not None:
                        st.success(f"Analysis completed for {len(results)} patients!")
                        
                        # Aggregate results across patients
                        import matplotlib.pyplot as plt
                        
                        # Collect all parameter values and reductions
                        all_data = []
                        for patient_id, patient_result in results.items():
                            if 'results' in patient_result:
                                patient_data = st.session_state.patient_cohort[patient_id]
                                scenario = patient_data['scenario'].iloc[0]
                                
                                for key, data in patient_result['results'].items():
                                    all_data.append({
                                        'patient_id': patient_id,
                                        'scenario': scenario,
                                        'parameter_value': data['parameter_value'],
                                        'reduction': data['reduction']
                                    })
                        
                        if all_data:
                            import pandas as pd
                            df = pd.DataFrame(all_data)
                            
                            # Create visualization by scenario
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Plot 1: All patients
                            for scenario in df['scenario'].unique():
                                scenario_data = df[df['scenario'] == scenario]
                                ax1.scatter(scenario_data['parameter_value'], scenario_data['reduction'], 
                                          alpha=0.6, label=scenario, s=50)
                            
                            ax1.set_xlabel(f'{param_to_vary} Multiplier')
                            ax1.set_ylabel('Tumor Volume Reduction (%)')
                            ax1.set_title('Parameter Sensitivity by Patient Scenario')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Plot 2: Average response by parameter value
                            avg_data = df.groupby('parameter_value')['reduction'].agg(['mean', 'std']).reset_index()
                            ax2.errorbar(avg_data['parameter_value'], avg_data['mean'], 
                                       yerr=avg_data['std'], marker='o', capsize=5, linewidth=2)
                            ax2.set_xlabel(f'{param_to_vary} Multiplier')
                            ax2.set_ylabel('Average Tumor Volume Reduction (%)')
                            ax2.set_title('Average Response Across Cohort')
                            ax2.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                best_avg = avg_data.loc[avg_data['mean'].idxmax()]
                                st.metric("Best Average Response", f"{best_avg['mean']:.1f}%")
                            with col2:
                                st.metric("Optimal Multiplier", f"{best_avg['parameter_value']:.2f}")
                            with col3:
                                response_range = avg_data['mean'].max() - avg_data['mean'].min()
                                st.metric("Response Range", f"{response_range:.1f}%")
                        else:
                            st.error("No valid results from cohort analysis.")
                    else:
                        st.error("Cohort sensitivity analysis failed.")

    def render_treatment_optimization_tools(self):
        """Render treatment optimization tools."""
        st.write("### Treatment Optimization")
        
        # Check data availability
        has_single = st.session_state.patient_data is not None and not st.session_state.patient_data.empty
        has_cohort = "patient_cohort" in st.session_state and st.session_state.patient_cohort
        
        if not has_single and not has_cohort:
            st.warning("Please load patient data or generate a cohort to use optimization tools.")
            return
        
        # Data source selection
        data_options = []
        if has_single:
            data_options.append("Current Patient")
        if has_cohort:
            data_options.append("Patient Cohort")
        
        data_source = st.radio("Optimize for:", data_options, horizontal=True, key="optimization_data_source")
        
        col1, col2 = st.columns(2)
        with col1:
            optimization_target = st.selectbox(
                "Optimization Target:",
                ["Maximize Tumor Reduction", "Minimize Final Burden", "Optimize Timing", "Minimize Toxicity"],
                key="optimization_target"
            )
        with col2:
            optimization_method = st.selectbox(
                "Method:",
                ["Grid Search", "Bayesian Optimization", "Genetic Algorithm"],
                key="optimization_method"
            )
        
        if st.button("🎯 Run Optimization", type="primary", key="run_optimization_analysis"):
            # Check data availability
            has_single = (st.session_state.patient_data is not None and not st.session_state.patient_data.empty)
            has_cohort = ("patient_cohort" in st.session_state and st.session_state.patient_cohort)
            
            if not has_single and not has_cohort:
                st.warning("Please load patient data or generate a cohort first.")
                return
            
            with st.spinner(f"Running {optimization_method} optimization for {optimization_target}..."):
                # If working with single patient, create temporary cohort
                if has_single and not has_cohort:
                    original_cohort = st.session_state.get('patient_cohort', {})
                    st.session_state.patient_cohort = {0: st.session_state.patient_data}
                    restore_cohort = True
                else:
                    restore_cohort = False
                
                try:
                    results = None
                    
                    # Route to appropriate optimization method
                    if optimization_target == "Maximize Tumor Reduction":
                        results = self.run_timing_optimization_analysis(
                            timing_param="Treatment Start Day",
                            optimization_metric="Tumor Volume Reduction"
                        )
                        
                    elif optimization_target == "Minimize Final Burden":
                        results = self.run_timing_optimization_analysis(
                            timing_param="Treatment Start Day",
                            optimization_metric="Final Tumor Burden"
                        )
                        
                    elif optimization_target == "Optimize Timing":
                        if optimization_method == "Grid Search":
                            results = self.run_timing_grid_search_optimization()
                        else:
                            results = self.run_timing_optimization_analysis(
                                timing_param="Treatment Start Day",
                                optimization_metric="Tumor Volume Reduction"
                            )
                            
                    elif optimization_target == "Minimize Toxicity":
                        results = self.run_toxicity_optimization_analysis(optimization_method)
                    
                    if results is not None:
                        st.success(f"{optimization_target} optimization completed!")
                        self.display_optimization_results(results, optimization_target, optimization_method)
                    else:
                        st.error("Optimization failed - no results returned.")
                        
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    
                finally:
                    # Restore original cohort if we created a temporary one
                    if restore_cohort:
                        if original_cohort:
                            st.session_state.patient_cohort = original_cohort
                        elif 'patient_cohort' in st.session_state:
                            del st.session_state.patient_cohort

    def render_model_comparison_tools(self):
        """Render model comparison tools."""
        st.write("### Model Comparison")
        
        # Check data availability
        if (st.session_state.patient_data is None or st.session_state.patient_data.empty) and \
           ("patient_cohort" not in st.session_state or not st.session_state.patient_cohort):
            st.warning("Please load patient data or generate a cohort to compare models.")
            return
        
        st.write("Compare how different mathematical models (ODE, PDE, ABM) predict treatment outcomes.")
        
        col1, col2 = st.columns(2)
        with col1:
            models_to_compare = st.multiselect(
                "Models to Compare:",
                ["ODE", "PDE", "ABM"],
                default=["ODE", "ABM"],
                key="models_comparison"
            )
        with col2:
            comparison_metric = st.selectbox(
                "Comparison Metric:",
                ["Tumor Volume Evolution", "Treatment Response", "Spatial Distribution", "Time to Control"],
                key="comparison_metric_analysis"
            )
        
        if st.button("🔬 Compare Models", type="primary", key="run_model_comparison") and len(models_to_compare) >= 2:
            # Check data availability
            has_data = (st.session_state.patient_data is not None and not st.session_state.patient_data.empty)
            
            if not has_data:
                st.warning("Please load patient data first.")
                return
            
            with st.spinner("Running model comparison..."):
                st.success("Model comparison started!")
                
                # Run simulations with different models
                comparison_results = {}
                simulation_time = 120
                
                for model_type in models_to_compare:
                    try:
                        if model_type == "ODE":
                            result = self.run_ode_simulation(simulation_time)
                        elif model_type == "PDE":
                            result = self.run_pde_simulation(simulation_time)
                        elif model_type == "ABM":
                            result = self.run_abm_simulation(simulation_time)
                        
                        comparison_results[model_type] = result
                        
                    except Exception as e:
                        st.error(f"Failed to run {model_type} simulation: {str(e)}")
                
                if len(comparison_results) >= 2:
                    # Display comparison visualization
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    # Plot 1: Tumor volume evolution
                    ax = axes[0]
                    for model_type, result in comparison_results.items():
                        if 'results' in result:
                            results_data = result['results']
                            if 'tumor_cells' in results_data:
                                time_points = range(len(results_data['tumor_cells']))
                                tumor_data = results_data['tumor_cells']
                                ax.plot(time_points, tumor_data, label=model_type, linewidth=2)
                    
                    ax.set_xlabel('Time (days)')
                    ax.set_ylabel('Tumor Cells')
                    ax.set_title('Tumor Volume Evolution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 2: Immune response
                    ax = axes[1]
                    for model_type, result in comparison_results.items():
                        if 'results' in result:
                            results_data = result['results']
                            if 'immune_cells' in results_data:
                                time_points = range(len(results_data['immune_cells']))
                                immune_data = results_data['immune_cells']
                                ax.plot(time_points, immune_data, label=model_type, linewidth=2)
                    
                    ax.set_xlabel('Time (days)')
                    ax.set_ylabel('Immune Cells')
                    ax.set_title('Immune Response')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 3: Treatment effect comparison
                    ax = axes[2]
                    model_names = []
                    final_reductions = []
                    
                    for model_type, result in comparison_results.items():
                        if 'results' in result:
                            results_data = result['results']
                            if 'tumor_cells' in results_data:
                                initial = results_data['tumor_cells'][0] if len(results_data['tumor_cells']) > 0 else 0
                                final = results_data['tumor_cells'][-1] if len(results_data['tumor_cells']) > 0 else 0
                                reduction = (initial - final) / initial * 100 if initial > 0 else 0
                                
                                model_names.append(model_type)
                                final_reductions.append(reduction)
                    
                    if model_names:
                        bars = ax.bar(model_names, final_reductions, alpha=0.7)
                        ax.set_ylabel('Tumor Reduction (%)')
                        ax.set_title('Treatment Effectiveness by Model')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels on bars
                        for bar, reduction in zip(bars, final_reductions):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                   f'{reduction:.1f}%', ha='center', va='bottom')
                    
                    # Plot 4: Model predictions summary
                    ax = axes[3]
                    ax.text(0.1, 0.8, "Model Comparison Summary:", fontsize=14, fontweight='bold', transform=ax.transAxes)
                    
                    y_pos = 0.6
                    for model_type, result in comparison_results.items():
                        if 'metrics' in result:
                            metrics = result['metrics']
                            summary_text = f"{model_type}: "
                            if 'tumor_burden_reduction' in metrics:
                                summary_text += f"Reduction: {metrics['tumor_burden_reduction']:.1f}%"
                            elif model_type in dict(zip(model_names, final_reductions)):
                                idx = model_names.index(model_type)
                                summary_text += f"Reduction: {final_reductions[idx]:.1f}%"
                            
                            ax.text(0.1, y_pos, summary_text, fontsize=12, transform=ax.transAxes)
                            y_pos -= 0.1
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Summary statistics
                    if model_names and final_reductions:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            best_model_idx = final_reductions.index(max(final_reductions))
                            st.metric("Best Model", model_names[best_model_idx])
                        with col2:
                            st.metric("Best Reduction", f"{max(final_reductions):.1f}%")
                        with col3:
                            reduction_range = max(final_reductions) - min(final_reductions)
                            st.metric("Model Variance", f"{reduction_range:.1f}%")
                
                else:
                    st.error("Could not compare models - insufficient simulation results.")

    def render_results_tab(self):
        """Render simulation results."""
        st.markdown(
            '<h2 class="sub-header">Simulation Results</h2>', unsafe_allow_html=True
        )

        if not st.session_state.simulation_results:
            st.info("Run a simulation to see results here.")
            return

        # Show patient data if available
        if st.session_state.patient_data is not None:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Patient Data")
                fig = self.data_handler.plot_patient_data(st.session_state.patient_data)
                st.pyplot(fig)

            with col2:
                st.subheader("Patient Info")
                if "patient_id" in st.session_state.patient_data.columns:
                    st.metric(
                        "Patient ID",
                        st.session_state.patient_data["patient_id"].iloc[0],
                    )
                if "scenario" in st.session_state.patient_data.columns:
                    st.metric(
                        "Scenario", st.session_state.patient_data["scenario"].iloc[0]
                    )

                st.metric("Data Points", len(st.session_state.patient_data))
                st.metric(
                    "Duration (days)",
                    st.session_state.patient_data["time"].max()
                    - st.session_state.patient_data["time"].min(),
                )

        st.divider()

        # Show simulation results
        for model_type, results in st.session_state.simulation_results.items():
            st.subheader(f"{model_type} Model Results")

            if model_type == "ODE":
                self.render_ode_results(results)
            elif model_type == "PDE":
                self.render_pde_results(results)
            elif model_type == "ABM":
                self.render_abm_results(results)

    def render_ode_results(self, results: Dict):
        """Render ODE simulation results."""
        # Handle different data structures (single patient vs cohort)
        if "results" in results:
            data = results["results"]
        else:
            data = results  # Direct simulation data from cohort
        
        # Create interactive plot with plotly
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Tumor Dynamics",
                "Immune Dynamics",
                "Drug Levels",
                "Treatment Schedule",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": True}],
            ],
        )

        # Tumor cells
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["tumor_cells"] / 1e6,
                name="Tumor Volume (cm³)",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

        # Immune cells
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["immune_cells"] / 1e3,
                name="Immune Cells (×10³)",
                line=dict(color="green"),
            ),
            row=1,
            col=2,
        )

        # Drug concentration
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["drug_concentration"],
                name="Drug Conc.",
                line=dict(color="blue"),
            ),
            row=2,
            col=1,
        )

        # Treatment schedule
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["rt_dose"],
                name="Radiation (Gy)",
                line=dict(color="purple"),
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["immuno_dose"],
                name="Immunotherapy",
                line=dict(color="orange"),
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time (days)")

        st.plotly_chart(fig, use_container_width=True)

        # Show metrics
        if "metrics" in results:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Tumor Control Probability",
                    f"{results['metrics']['tumor_control_probability']:.3f}",
                )

            with col2:
                st.metric(
                    "Immune Cell Viability",
                    f"{results['metrics']['immune_cell_viability']:.3f}",
                )

            with col3:
                st.metric(
                    "Tumor Reduction", f"{results['metrics']['tumor_reduction']:.1%}"
                )

            with col4:
                st.metric(
                    "Final Tumor Burden",
                    f"{results['metrics']['final_tumor_burden']:.1e}",
                )

    def render_pde_results(self, results: Dict):
        """Render PDE simulation results with spatiotemporal evolution."""
        st.subheader("📊 PDE Simulation Results")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Time Series",
                "🗺️ Final Spatial Fields",
                "🎬 Spatiotemporal Evolution",
                "📍 Cross-Sections",
            ]
        )

        with tab1:
            # Time series plots
            col1, col2 = st.columns(2)

            with col1:
                # Create interactive plotly figure
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("Tumor Volume", "Average Oxygen Level"),
                    vertical_spacing=0.12,
                )

                fig.add_trace(
                    go.Scatter(
                        x=results["results"]["time"],
                        y=results["results"]["tumor_volume"],
                        mode="lines",
                        name="Tumor Volume",
                        line=dict(color="red"),
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=results["results"]["time"],
                        y=results["results"]["avg_oxygen"],
                        mode="lines",
                        name="Avg Oxygen",
                        line=dict(color="blue"),
                    ),
                    row=2,
                    col=1,
                )

                fig.update_xaxes(title_text="Time (days)", row=2, col=1)
                fig.update_yaxes(title_text="Volume", row=1, col=1)
                fig.update_yaxes(title_text="Oxygen Level", row=2, col=1)
                fig.update_layout(height=500, showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Additional time series
                fig2 = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("Total Drug", "Total Radiation"),
                    vertical_spacing=0.12,
                )

                fig2.add_trace(
                    go.Scatter(
                        x=results["results"]["time"],
                        y=results["results"]["total_drug"],
                        mode="lines",
                        name="Total Drug",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

                fig2.add_trace(
                    go.Scatter(
                        x=results["results"]["time"],
                        y=results["results"]["total_radiation"],
                        mode="lines",
                        name="Total Radiation",
                        line=dict(color="purple"),
                    ),
                    row=2,
                    col=1,
                )

                fig2.update_xaxes(title_text="Time (days)", row=2, col=1)
                fig2.update_yaxes(title_text="Drug Amount", row=1, col=1)
                fig2.update_yaxes(title_text="Radiation Dose", row=2, col=1)
                fig2.update_layout(height=500, showlegend=False)

                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            # Show final spatial distributions with improved visualization
            model = results["spatial_model"]

            # Create interactive spatial plots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Tumor Density",
                    "Drug Concentration",
                    "Oxygen Level",
                    "Radiation Dose",
                ),
                specs=[
                    [{"type": "heatmap"}, {"type": "heatmap"}],
                    [{"type": "heatmap"}, {"type": "heatmap"}],
                ],
            )

            # Tumor density
            fig.add_trace(
                go.Heatmap(z=model.tumor_density.T, colorscale="Reds", name="Tumor"),
                row=1,
                col=1,
            )

            # Drug concentration
            fig.add_trace(
                go.Heatmap(
                    z=model.drug_concentration.T, colorscale="Blues", name="Drug"
                ),
                row=1,
                col=2,
            )

            # Oxygen level
            fig.add_trace(
                go.Heatmap(z=model.oxygen_level.T, colorscale="Greens", name="Oxygen"),
                row=2,
                col=1,
            )

            # Radiation dose
            fig.add_trace(
                go.Heatmap(
                    z=model.radiation_dose.T, colorscale="Purples", name="Radiation"
                ),
                row=2,
                col=2,
            )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Spatiotemporal evolution visualization
            if "tumor_evolution" in results["results"]:
                st.write("**🎬 Complete Spatiotemporal Evolution**")

                # Time slider for interactive exploration
                if "saved_times" in results["results"]:
                    saved_times = results["results"]["saved_times"]

                    selected_time_idx = st.slider(
                        "Select time point:",
                        min_value=0,
                        max_value=len(saved_times) - 1,
                        value=len(saved_times) // 2,
                        help=f"Time range: {saved_times[0]:.1f} - {saved_times[-1]:.1f} days",
                    )

                    current_time = saved_times[selected_time_idx]
                    st.write(f"**Current time: {current_time:.1f} days**")

                    # Display spatial fields at selected time
                    col1, col2 = st.columns(2)

                    with col1:
                        # Tumor and Drug at selected time
                        fig1 = make_subplots(
                            rows=1,
                            cols=2,
                            subplot_titles=(
                                f"Tumor (t={saved_times[selected_time_idx]:.1f}d)",
                                f"Drug (t={saved_times[selected_time_idx]:.1f}d)",
                            ),
                        )

                        fig1.add_trace(
                            go.Heatmap(
                                z=results["results"]["tumor_evolution"][
                                    selected_time_idx
                                ].T,
                                colorscale="Reds",
                            ),
                            row=1,
                            col=1,
                        )

                        fig1.add_trace(
                            go.Heatmap(
                                z=results["results"]["drug_evolution"][
                                    selected_time_idx
                                ].T,
                                colorscale="Blues",
                            ),
                            row=1,
                            col=2,
                        )

                        fig1.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)

                    with col2:
                        # Oxygen and Radiation at selected time
                        fig2 = make_subplots(
                            rows=1,
                            cols=2,
                            subplot_titles=(
                                f"Oxygen (t={saved_times[selected_time_idx]:.1f}d)",
                                f"Radiation (t={saved_times[selected_time_idx]:.1f}d)",
                            ),
                        )

                        fig2.add_trace(
                            go.Heatmap(
                                z=results["results"]["oxygen_evolution"][
                                    selected_time_idx
                                ].T,
                                colorscale="Greens",
                            ),
                            row=1,
                            col=1,
                        )

                        fig2.add_trace(
                            go.Heatmap(
                                z=results["results"]["radiation_evolution"][
                                    selected_time_idx
                                ].T,
                                colorscale="Purples",
                            ),
                            row=1,
                            col=2,
                        )

                        fig2.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)

                    # Generate matplotlib plot for complete evolution
                    if st.button("📥 Generate Full Evolution Plot"):
                        with st.spinner("Generating spatiotemporal evolution plot..."):
                            model = results["spatial_model"]
                            fig_evolution = model.plot_spatiotemporal_evolution(
                                results["results"]
                            )
                            st.pyplot(fig_evolution)

            else:
                st.info(
                    "💡 Spatiotemporal evolution data not available. Run simulation with spatiotemporal tracking enabled."
                )

        with tab4:
            # Cross-sectional analysis
            if "tumor_evolution" in results["results"]:
                st.write("**📍 Spatial and Temporal Cross-Sections**")

                col1, col2 = st.columns(2)

                with col1:
                    # Temporal cross-sections
                    if st.button("📈 Generate Temporal Cross-Sections"):
                        with st.spinner("Generating temporal analysis..."):
                            model = results["spatial_model"]
                            fig_temporal = model.plot_temporal_cross_sections(
                                results["results"]
                            )
                            st.pyplot(fig_temporal)

                with col2:
                    # Spatial profiles
                    direction = st.selectbox("Profile Direction:", ["x", "y"])
                    if st.button("📊 Generate Spatial Profiles"):
                        with st.spinner("Generating spatial profiles..."):
                            model = results["spatial_model"]
                            fig_profiles = model.plot_spatial_profiles(
                                results["results"], direction=direction
                            )
                            st.pyplot(fig_profiles)
            else:
                st.info(
                    "💡 Cross-sectional analysis requires spatiotemporal evolution data."
                )

        # Add summary metrics
        st.subheader("📊 PDE Simulation Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Final Tumor Volume",
                f"{results['results']['tumor_volume'][-1]:.3f} mm³",
            )

        with col2:
            st.metric(
                "Volume Reduction",
                f"{(1 - results['results']['tumor_volume'][-1] / results['results']['tumor_volume'][0]) * 100:.1f}%",
            )

        with col3:
            st.metric(
                "Min Oxygen Level", f"{results['results']['avg_oxygen'].min():.3f}"
            )

        with col4:
            st.metric(
                "Max Drug Concentration",
                f"{results['results']['total_drug'].max():.3f}",
            )

    def render_abm_results(self, results: Dict):
        """Render ABM simulation results."""
        # Handle different data structures (single patient vs cohort)
        if "results" in results:
            data = results["results"]
        else:
            data = results  # Direct simulation data from cohort
        
        # Handle case where data might be a list (from optimized ABM)
        if isinstance(data, list):
            # Convert list format to expected dictionary format
            if data:
                # Extract time series data from list of states
                time_data = [state.get('time', 0) for state in data]
                tumor_data = [state.get('tumor_volume', 0) for state in data]
                immune_data = [state.get('immune_count', 0) for state in data]
                dead_data = [state.get('dead_count', 0) for state in data]
                
                # Create compatible data structure
                data = {
                    'time': time_data,
                    'tumor_count': tumor_data,
                    'immune_count': immune_data,
                    'dead_count': dead_data,
                    'total_cells': [state.get('total_cells', 0) for state in data]
                }
                
                st.info("🔄 Converted optimized ABM results format for display")
            else:
                st.warning("No simulation data available")
                return
            
        col1, col2 = st.columns(2)

        with col1:
            # Population dynamics with interactive plot
            time_data = data.get("time", [])
            time_hours = time_data
            time_days = np.array(time_hours) / 24 if len(time_hours) > 0 else []

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Cell Population Dynamics", "Microenvironment"),
                vertical_spacing=0.12,
            )

            # Check if we have the expected data
            if len(time_days) > 0 and "tumor_count" in data:
                fig.add_trace(
                    go.Scatter(
                        x=time_days,
                        y=data["tumor_count"],
                        mode="lines",
                        name="Tumor Cells",
                        line=dict(color="red"),
                    ),
                    row=1,
                    col=1,
                )

            if len(time_days) > 0 and "immune_count" in data:
                fig.add_trace(
                    go.Scatter(
                        x=time_days,
                        y=data["immune_count"],
                        mode="lines",
                        name="Immune Cells",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=1,
                )

            if len(time_days) > 0 and "avg_oxygen" in data:
                fig.add_trace(
                    go.Scatter(
                        x=time_days,
                        y=data["avg_oxygen"],
                        mode="lines",
                        name="Avg Oxygen",
                        line=dict(color="blue"),
                    ),
                    row=2,
                    col=1,
                )

            fig.update_xaxes(title_text="Time (days)", row=2, col=1)
            fig.update_yaxes(title_text="Cell Count", row=1, col=1)
            fig.update_yaxes(title_text="Oxygen Level", row=2, col=1)
            fig.update_layout(height=500, showlegend=True)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Final grid state with enhanced visualization
            if "abm_model" in results:
                # Get cell counts for debugging
                model = results["abm_model"]
                initial_grid = results.get("initial_grid", None)
                
                # 🔧 DEBUG: Add extensive debugging to understand the issue
                st.write("**🔍 Debug Information:**")
                st.write(f"Model type: {type(model)}")
                st.write(f"Has cells attribute: {hasattr(model, 'cells')}")
                st.write(f"Has cell_counts attribute: {hasattr(model, 'cell_counts')}")
                
                if hasattr(model, 'cells'):
                    st.write(f"Total cells in model.cells: {len(model.cells)}")
                    if model.cells:
                        sample_cell = model.cells[0]
                        st.write(f"Sample cell type: {sample_cell.cell_type} (type: {type(sample_cell.cell_type)})")
                
                if hasattr(model, 'cell_counts'):
                    st.write(f"Model cell_counts: {model.cell_counts}")
                
                # Get actual cell counts from model
                if hasattr(model, 'cells') and model.cells:
                    total_cells = len(model.cells)
                    
                    # 🔧 FIX: Use the correct filtering logic based on our test
                    tumor_count = len([c for c in model.cells if c.cell_type == CellType.TUMOR])
                    immune_count = len([c for c in model.cells if c.cell_type == CellType.IMMUNE]) 
                    dead_count = len([c for c in model.cells if c.cell_type == CellType.DEAD])
                    
                    st.write(f"**Corrected cell counts:** Tumor: {tumor_count}, Immune: {immune_count}, Dead: {dead_count}")
                else:
                    total_cells = tumor_count = immune_count = dead_count = 0
                    st.write("**⚠️ No cells found in model!**")
                
                # Show cell count summary with more detail
                st.write(f"**Current Cell Counts:** Tumor: {tumor_count}, Immune: {immune_count}, Dead: {dead_count}, Total: {total_cells}")
                
                # Add cell density information
                grid_volume = model.width * model.height * (model.depth if hasattr(model, 'depth') else 1)
                cell_density = total_cells / grid_volume * 100 if grid_volume > 0 else 0
                st.write(f"**Grid Size:** {model.width} × {model.height} × {getattr(model, 'depth', 1)} = {grid_volume:,} positions")
                st.write(f"**Cell Density:** {cell_density:.2f}% of grid positions occupied")
                
                # Create interactive heatmap of final or initial state
                # Always prepare a 2D slice for default view; let user choose stage
                default_slice = model.depth // 2 if hasattr(model, 'depth') else 0
                stage_options = ["Final"]
                if initial_grid is not None:
                    stage_options.insert(0, "Initial")
                selected_stage = st.selectbox("Visualization Stage:", stage_options, key="abm_stage")

                if selected_stage == "Initial" and initial_grid is not None:
                    if hasattr(initial_grid, 'ndim') and initial_grid.ndim == 3:
                        slice_idx = min(max(default_slice, 0), initial_grid.shape[2]-1)
                        grid_viz = initial_grid[:, :, slice_idx]
                    else:
                        grid_viz = initial_grid
                else:
                    grid_viz = model.get_grid_visualization(slice_z=default_slice)
                
                # 🔧 FIX: Ensure we have a proper 2D array for visualization
                if len(grid_viz.shape) == 3 and grid_viz.shape[2] == 1:
                    grid_viz = grid_viz[:, :, 0]  # Extract 2D slice from 3D array
                
                # Debug: show unique values in grid (use the original untransposed grid for debug)
                unique_values, counts = np.unique(grid_viz, return_counts=True)
                value_counts = np.zeros(4, dtype=int)
                for val, count in zip(unique_values, counts):
                    if val < 4:
                        value_counts[val] = count
                
                st.write("**🔍 Grid Debug Info:**")
                st.write(f"• Grid shape: {grid_viz.shape}")
                st.write(f"• Unique values: {unique_values} (0=Empty, 1=Tumor, 2=Immune, 3=Dead)")
                st.write(f"• Value counts: {counts}")
                st.write(f"• Total grid positions: {grid_viz.size}")
                
                # Check if grid is empty and why
                if len(unique_values) == 1 and unique_values[0] == 0:
                    st.error("⚠️ **Grid is completely empty!** This suggests:")
                    st.write("• You might be viewing an empty slice (try a different Z-slice)")
                    st.write("• The simulation might not have run properly")
                    st.write("• There might be a coordinate system issue")
                else:
                    st.success(f"✅ **Grid contains cells:** {counts[1:].sum()} non-empty positions")
                
                # Add visualization options
                if hasattr(model, 'depth') and model.depth > 1:
                    depth = model.depth
                    st.write(f"**📐 Model Grid:** {model.width} × {model.height} × {depth} (Valid Z-slices: 0-{depth-1})")
                    
                    view_option = st.selectbox(
                        "View Option:",
                        ["3D Projection (All Slices)", "Specific Z-Slice", "3D Scatter Plot"],
                        key="abm_view_option"
                    )
                    
                    if view_option == "Specific Z-Slice":
                        # 🔧 FIX: Ensure slider bounds are correct and add validation
                        max_slice = depth - 1
                        default_slice = min(depth // 2, max_slice)  # Ensure default is in bounds
                        
                        z_slice = st.slider(
                            f"Select Z-Slice (0-{max_slice}):", 
                            0, max_slice, default_slice,
                            key="abm_z_slice"
                        )
                        
                        # Additional validation
                        if z_slice >= depth:
                            st.error(f"⚠️ Slice {z_slice} is out of bounds! Using slice {default_slice} instead.")
                            z_slice = default_slice
                        
                        st.write(f"**📍 Viewing Z-Slice:** {z_slice} of {max_slice}")
                        if selected_stage == "Initial" and initial_grid is not None and hasattr(initial_grid, 'ndim') and initial_grid.ndim == 3:
                            grid_viz = initial_grid[:, :, z_slice]
                        else:
                            grid_viz = model.get_grid_visualization(slice_z=z_slice)
                        # Ensure 2D for heatmap (optimized model may return (x,y,1))
                        if hasattr(grid_viz, 'shape') and len(grid_viz.shape) == 3 and grid_viz.shape[2] == 1:
                            grid_viz = grid_viz[:, :, 0]
                        title_suffix = f" (Z-Slice {z_slice})"
                    elif view_option == "3D Scatter Plot":
                        # Create 3D scatter plot
                        tumor_positions = []
                        immune_positions = []
                        dead_positions = []

                        # Sample cells for performance (show max 1000 of each type)
                        max_cells_per_type = 1000
                        tumor_cells = [c for c in model.cells if c.cell_type.name == "TUMOR" or c.cell_type.value == "tumor"]
                        immune_cells = [c for c in model.cells if c.cell_type.name == "IMMUNE" or c.cell_type.value == "immune"]
                        dead_cells = [c for c in model.cells if c.cell_type.name == "DEAD" or c.cell_type.value == "dead"]
                        
                        # Sample if too many cells
                        if len(tumor_cells) > max_cells_per_type:
                            import random
                            tumor_cells = random.sample(tumor_cells, max_cells_per_type)
                        if len(immune_cells) > max_cells_per_type:
                            import random
                            immune_cells = random.sample(immune_cells, max_cells_per_type)
                        if len(dead_cells) > max_cells_per_type:
                            import random
                            dead_cells = random.sample(dead_cells, max_cells_per_type)

                        for cell in tumor_cells:
                            tumor_positions.append([cell.x, cell.y, cell.z])
                        for cell in immune_cells:
                            immune_positions.append([cell.x, cell.y, cell.z])
                        for cell in dead_cells:
                            dead_positions.append([cell.x, cell.y, cell.z])

                        fig_3d = go.Figure()

                        if tumor_positions:
                            tumor_pos = np.array(tumor_positions)
                            fig_3d.add_trace(go.Scatter3d(
                                x=tumor_pos[:, 0], y=tumor_pos[:, 1], z=tumor_pos[:, 2],
                                mode='markers',
                                marker=dict(color='red', size=4, opacity=0.7),
                                name=f'Tumor Cells ({len(tumor_positions)})'
                            ))

                        if immune_positions:
                            immune_pos = np.array(immune_positions)
                            fig_3d.add_trace(go.Scatter3d(
                                x=immune_pos[:, 0], y=immune_pos[:, 1], z=immune_pos[:, 2],
                                mode='markers',
                                marker=dict(color='green', size=3, opacity=0.7),
                                name=f'Immune Cells ({len(immune_positions)})'
                            ))

                        if dead_positions:
                            dead_pos = np.array(dead_positions)
                            fig_3d.add_trace(go.Scatter3d(
                                x=dead_pos[:, 0], y=dead_pos[:, 1], z=dead_pos[:, 2],
                                mode='markers',
                                marker=dict(color='black', size=2, opacity=0.4),
                                name=f'Dead Cells ({len(dead_positions)})'
                            ))

                        fig_3d.update_layout(
                            title="3D Cell Distribution",
                            scene=dict(
                                xaxis_title="X Position",
                                yaxis_title="Y Position", 
                                zaxis_title="Z Position"
                            ),
                            height=600,
                            width=700
                        )

                        st.plotly_chart(fig_3d, use_container_width=True)
                        
                        # Add 3D analysis
                        st.write("**3D Spatial Analysis:**")
                        if tumor_positions:
                            tumor_center = np.mean(tumor_positions, axis=0)
                            st.write(f"• Tumor Center: ({tumor_center[0]:.1f}, {tumor_center[1]:.1f}, {tumor_center[2]:.1f})")
                            
                            # Calculate spread
                            if len(tumor_positions) > 1:
                                tumor_spread = np.std(tumor_positions, axis=0)
                                st.write(f"• Tumor Spread: X±{tumor_spread[0]:.1f}, Y±{tumor_spread[1]:.1f}, Z±{tumor_spread[2]:.1f}")
                                
                                # Calculate invasion depth (max distance from center)
                                distances_from_center = np.linalg.norm(np.array(tumor_positions) - tumor_center, axis=1)
                                max_invasion = np.max(distances_from_center)
                                st.write(f"• Max Invasion Distance: {max_invasion:.1f} grid units")
                        
                        if immune_positions and tumor_positions:
                            # Calculate immune-tumor interaction metrics
                            immune_pos_array = np.array(immune_positions)
                            tumor_pos_array = np.array(tumor_positions)
                            
                            # Find average distance between immune cells and nearest tumor
                            min_distances = []
                            for immune_pos in immune_pos_array:
                                distances = np.linalg.norm(tumor_pos_array - immune_pos, axis=1)
                                min_distances.append(np.min(distances))
                            
                            avg_immune_tumor_distance = np.mean(min_distances)
                            st.write(f"• Avg Immune-Tumor Distance: {avg_immune_tumor_distance:.1f} grid units")
                            
                            close_immune = sum(1 for d in min_distances if d <= 3.0)
                            st.write(f"• Immune Cells Near Tumors (<3 units): {close_immune} ({close_immune/len(immune_positions)*100:.1f}%)")
                        
                        # Skip the heatmap for 3D view
                        grid_viz = None
                        title_suffix = " (3D Scatter)"
                    else:
                        # True projection over Z with priority: Dead(3) > Tumor(1) > Immune(2) > Empty(0)
                        if selected_stage == "Initial" and initial_grid is not None:
                            full_grid = initial_grid
                        else:
                            full_grid = model.get_grid_visualization()
                        if hasattr(full_grid, 'ndim') and full_grid.ndim == 3:
                            has_dead = (full_grid == 3).any(axis=2)
                            has_tumor = (full_grid == 1).any(axis=2)
                            has_immune = (full_grid == 2).any(axis=2)
                            projection = np.zeros((full_grid.shape[0], full_grid.shape[1]), dtype=int)
                            projection = np.where(has_dead, 3, projection)
                            projection = np.where(~has_dead & has_tumor, 1, projection)
                            projection = np.where(~has_dead & ~has_tumor & has_immune, 2, projection)
                            grid_viz = projection
                        else:
                            # Non-optimized ABM returns 2D aggregated already
                            grid_viz = full_grid
                        title_suffix = " (3D Projection)"
                else:
                    title_suffix = " (2D View)"

                # Create a more distinctive colorscale that better shows differences
                # Use discrete colors for better cell type differentiation
                if grid_viz is not None:
                    # Ensure 2D input for Heatmap
                    if hasattr(grid_viz, 'shape') and len(grid_viz.shape) == 3 and grid_viz.shape[2] == 1:
                        grid_viz = grid_viz[:, :, 0]
                    # 🔧 FIX: Transpose the grid to fix X/Y coordinate system
                    # ABM model returns grid[x,y] but heatmap expects grid[y,x]
                    grid_viz_transposed = grid_viz.T
                    
                    colorscale_discrete = [
                        [0.0, "#FFFFFF"],      # White for empty
                        [0.25, "#FFFFFF"],     # Keep white
                        [0.25001, "#FF4136"],  # Red for tumor  
                        [0.5, "#FF4136"],      # Keep red
                        [0.50001, "#2ECC40"], # Green for immune
                        [0.75, "#2ECC40"],     # Keep green
                        [0.75001, "#111111"],  # Black for dead
                        [1.0, "#111111"]       # Keep black
                    ]

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=grid_viz_transposed,
                            colorscale=colorscale_discrete,
                            showscale=True,
                            zmin=0,
                            zmax=3,
                            colorbar=dict(
                                title="Cell Type",
                                tickmode="array",
                                tickvals=[0, 1, 2, 3],
                                ticktext=["Empty", "Tumor", "Immune", "Dead"],
                                len=0.8,
                            ),
                            hovertemplate="X: %{x}<br>Y: %{y}<br>Cell Type: %{customdata}<extra></extra>",
                            customdata=np.where(grid_viz_transposed == 0, "Empty",
                                       np.where(grid_viz_transposed == 1, "Tumor", 
                                       np.where(grid_viz_transposed == 2, "Immune", "Dead")))
                        )
                    )

                    fig.update_layout(
                        title=f"Final Cell Distribution{title_suffix}",
                        xaxis_title="Grid X",
                        yaxis_title="Grid Y",
                        height=500,
                        width=500,
                        xaxis=dict(constrain="domain"),
                        yaxis=dict(constrain="domain", scaleanchor="x", scaleratio=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add statistics table (using transposed grid for correct counts)
                    st.write("**Spatial Statistics:**")
                    unique_values_t, counts_t = np.unique(grid_viz_transposed, return_counts=True)
                    value_counts_t = np.zeros(4, dtype=int)
                    for val, count in zip(unique_values_t, counts_t):
                        if val < 4:
                            value_counts_t[val] = count
                    
                    stats_df = pd.DataFrame({
                        'Cell Type': ['Empty', 'Tumor', 'Immune', 'Dead'],
                        'Count': [value_counts_t[i] for i in range(4)],
                        'Percentage': [f"{value_counts_t[i] / grid_viz_transposed.size * 100:.1f}%" for i in range(4)]
                    })
                    st.dataframe(stats_df, use_container_width=True)

        # Add detailed metrics
        st.subheader("ABM Simulation Metrics")

        # Calculate additional metrics
        initial_tumor = (
            results["results"]["tumor_count"][0]
            if results["results"]["tumor_count"]
            else 0
        )
        final_tumor = (
            results["results"]["tumor_count"][-1]
            if results["results"]["tumor_count"]
            else 0
        )
        initial_immune = (
            results["results"]["immune_count"][0]
            if results["results"]["immune_count"]
            else 0
        )
        final_immune = (
            results["results"]["immune_count"][-1]
            if results["results"]["immune_count"]
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Final Tumor Cells",
                f"{final_tumor}",
                delta=f"{final_tumor - initial_tumor}",
            )

        with col2:
            st.metric(
                "Final Immune Cells",
                f"{final_immune}",
                delta=f"{final_immune - initial_immune}",
            )

        with col3:
            tumor_change = (
                ((final_tumor - initial_tumor) / initial_tumor * 100)
                if initial_tumor > 0
                else 0
            )
            st.metric("Tumor Change", f"{tumor_change:.1f}%")

        with col4:
            immune_efficiency = (
                (final_immune / (final_tumor + 1)) if final_tumor > 0 else final_immune
            )
            st.metric("Immune Efficiency", f"{immune_efficiency:.2f}")

    def render_comparison_tab(self):
        """Render comparison tools."""
        st.markdown(
            '<h2 class="sub-header">Treatment Comparison</h2>', unsafe_allow_html=True
        )

        # Protocol comparison
        st.subheader("Compare Treatment Protocols")

        protocols = self.data_handler.generate_treatment_protocols()
        selected_protocols = st.multiselect(
            "Select protocols to compare",
            options=list(protocols.keys()),
            default=list(protocols.keys())[:2],
        )

        if st.button("Compare Protocols") and st.session_state.patient_data is not None:
            with st.spinner("Comparing treatment protocols..."):
                comparison_results = self.data_handler.compare_treatment_protocols(
                    st.session_state.patient_data,
                    {name: protocols[name] for name in selected_protocols},
                )

                # Display comparison results
                self.display_protocol_comparison(comparison_results)

    def render_export_tab(self):
        """Render export tools."""
        st.markdown(
            '<h2 class="sub-header">Export Results</h2>', unsafe_allow_html=True
        )

        if not st.session_state.simulation_results:
            st.info("Run simulations to enable export.")
            return

        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "PDF Report"])

        if st.button("Generate Export"):
            self.generate_export(export_format)

    def run_sensitivity_analysis(self):
        """Run parameter sensitivity analysis."""
        if st.session_state.patient_data is None:
            st.error("Please load patient data first!")
            return

        with st.spinner("Running sensitivity analysis..."):
            # Parameters to analyze
            parameters = {
                "r_tumor": [0.05, 0.1, 0.15, 0.2],
                "k_kill": [1e-10, 1e-9, 1e-8, 1e-7],
                "alpha_rt": [0.1, 0.3, 0.5, 0.7],
            }

            results = {}
            base_model = TumorImmuneODE()

            for param_name, param_values in parameters.items():
                param_results = []

                for value in param_values:
                    # Create model with modified parameter
                    model_params = base_model.params.copy()
                    model_params[param_name] = value
                    model = TumorImmuneODE(model_params)

                    # Run simulation
                    initial_conditions = np.array(
                        [
                            st.session_state.patient_data["tumor_volume"].iloc[0] * 1e6,
                            1e5,
                            0,
                            100,
                            0,
                        ]
                    )
                    time_points = np.linspace(0, 120, 121)

                    try:
                        simulation = model.simulate(
                            initial_conditions,
                            time_points,
                            st.session_state.therapy_schedule or {},
                        )
                        metrics = model.calculate_metrics(simulation)
                        param_results.append(
                            {
                                "value": value,
                                "tumor_control": metrics["tumor_control_probability"],
                                "final_burden": metrics["final_tumor_burden"],
                            }
                        )
                    except Exception as e:
                        st.warning(f"Simulation failed for {param_name}={value}: {e}")

                results[param_name] = param_results

            # Display results
            st.subheader("Parameter Sensitivity Results")

            # Create sensitivity plots
            fig = make_subplots(
                rows=1, cols=len(parameters), subplot_titles=list(parameters.keys())
            )

            colors = ["blue", "red", "green"]
            for i, (param_name, param_data) in enumerate(results.items()):
                if param_data:
                    values = [d["value"] for d in param_data]
                    tcp_values = [d["tumor_control"] for d in param_data]

                    fig.add_trace(
                        go.Scatter(
                            x=values,
                            y=tcp_values,
                            mode="lines+markers",
                            name=f"TCP vs {param_name}",
                            line=dict(color=colors[i]),
                        ),
                        row=1,
                        col=i + 1,
                    )

            fig.update_layout(height=400, showlegend=True)
            fig.update_yaxes(title_text="Tumor Control Probability")
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.subheader("Sensitivity Summary")
            summary_data = []
            for param_name, param_data in results.items():
                if param_data and len(param_data) > 1:
                    tcp_values = [d["tumor_control"] for d in param_data]
                    sensitivity = max(tcp_values) - min(tcp_values)
                    summary_data.append(
                        {
                            "Parameter": param_name,
                            "Min TCP": f"{min(tcp_values):.3f}",
                            "Max TCP": f"{max(tcp_values):.3f}",
                            "Sensitivity": f"{sensitivity:.3f}",
                        }
                    )

            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)

    def run_treatment_optimization(self):
        """Run treatment protocol optimization."""
        if st.session_state.patient_data is None:
            st.error("Please load patient data first!")
            return

        with st.spinner("Optimizing treatment protocol..."):
            from scipy.optimize import minimize

            def objective_function(params):
                """Objective function for optimization."""
                rt_dose, rt_duration, immuno_dose = params

                # Create therapy schedule
                therapy_schedule = {
                    "radiotherapy": {
                        "start_time": 30,
                        "end_time": 30 + rt_duration,
                        "dose": rt_dose,
                        "frequency": 1,
                        "duration": 1,
                    },
                    "immunotherapy": {
                        "start_time": 35,
                        "end_time": 95,
                        "dose": immuno_dose,
                        "frequency": 7,
                        "duration": 1,
                    },
                }

                try:
                    # Run simulation
                    model = TumorImmuneODE()
                    initial_conditions = np.array(
                        [
                            st.session_state.patient_data["tumor_volume"].iloc[0] * 1e6,
                            1e5,
                            0,
                            100,
                            0,
                        ]
                    )
                    time_points = np.linspace(0, 180, 181)

                    simulation = model.simulate(
                        initial_conditions, time_points, therapy_schedule
                    )
                    metrics = model.calculate_metrics(simulation)

                    # Objective: maximize tumor control while minimizing toxicity
                    tumor_control = metrics["tumor_control_probability"]
                    toxicity_penalty = (rt_dose * rt_duration / 100) + (
                        immuno_dose / 10
                    )

                    # Return negative because we want to maximize
                    return -(tumor_control - 0.1 * toxicity_penalty)

                except Exception:
                    return 1000  # Large penalty for failed simulations

            # Optimization bounds: [rt_dose, rt_duration, immuno_dose]
            bounds = [(1.0, 5.0), (10, 60), (0.5, 3.0)]
            initial_guess = [2.0, 35, 1.0]

            # Run optimization
            result = minimize(
                objective_function, initial_guess, method="L-BFGS-B", bounds=bounds
            )

            if result.success:
                optimal_rt_dose, optimal_rt_duration, optimal_immuno_dose = result.x

                st.success("Optimization completed successfully!")

                # Display optimal parameters
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Optimal RT Dose (Gy)", f"{optimal_rt_dose:.2f}")

                with col2:
                    st.metric(
                        "Optimal RT Duration (days)", f"{optimal_rt_duration:.0f}"
                    )

                with col3:
                    st.metric("Optimal Immuno Dose", f"{optimal_immuno_dose:.2f}")

                # Show comparison with standard protocol
                st.subheader("Optimization Results")

                # Standard protocol
                standard_schedule = {
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
                }

                # Optimal protocol
                optimal_schedule = {
                    "radiotherapy": {
                        "start_time": 30,
                        "end_time": 30 + optimal_rt_duration,
                        "dose": optimal_rt_dose,
                        "frequency": 1,
                        "duration": 1,
                    },
                    "immunotherapy": {
                        "start_time": 35,
                        "end_time": 95,
                        "dose": optimal_immuno_dose,
                        "frequency": 7,
                        "duration": 1,
                    },
                }

                # Compare protocols
                protocols = {
                    "Standard Protocol": standard_schedule,
                    "Optimized Protocol": optimal_schedule,
                }

                comparison_results = self.data_handler.compare_treatment_protocols(
                    st.session_state.patient_data, protocols
                )

                self.display_protocol_comparison(comparison_results)

            else:
                st.error(f"Optimization failed: {result.message}")

    def display_protocol_comparison(self, comparison_results: Dict):
        """Display protocol comparison results."""
        # Create comparison metrics table
        metrics_data = []
        for protocol, data in comparison_results.items():
            metrics_data.append(
                {
                    "Protocol": protocol,
                    "TCP": f"{data['metrics']['tumor_control_probability']:.3f}",
                    "ICV": f"{data['metrics']['immune_cell_viability']:.3f}",
                    "Reduction": f"{data['metrics']['tumor_reduction']:.1%}",
                    "Fit Quality": f"{data['fit_quality']:.3f}",
                }
            )

        df_metrics = pd.DataFrame(metrics_data)
        st.subheader("Protocol Comparison Metrics")
        st.dataframe(df_metrics, use_container_width=True)

        # Plot comparison
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Tumor Dynamics", "Treatment Metrics")
        )

        colors = px.colors.qualitative.Set1
        for i, (protocol, data) in enumerate(comparison_results.items()):
            simulation = data["simulation"]
            fig.add_trace(
                go.Scatter(
                    x=simulation["time"],
                    y=simulation["tumor_cells"] / 1e6,
                    name=protocol,
                    line=dict(color=colors[i % len(colors)]),
                ),
                row=1,
                col=1,
            )

        # Metrics bar chart
        metrics_names = ["TCP", "ICV", "Tumor Reduction"]
        for i, protocol in enumerate(comparison_results.keys()):
            metrics = comparison_results[protocol]["metrics"]
            values = [
                metrics["tumor_control_probability"],
                metrics["immune_cell_viability"],
                metrics["tumor_reduction"],
            ]
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=values,
                    name=protocol,
                    marker_color=colors[i % len(colors)],
                ),
                row=1,
                col=2,
            )

        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="Time (days)", row=1, col=1)
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Tumor Volume (cm³)", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    def generate_export(self, export_format: str):
        """Generate export file."""
        if export_format == "CSV":
            # Export simulation data as CSV
            for model_type, results in st.session_state.simulation_results.items():
                if "results" in results:
                    csv = results["results"].to_csv(index=False)
                    st.download_button(
                        label=f"Download {model_type} Results (CSV)",
                        data=csv,
                        file_name=f"{model_type}_simulation_results.csv",
                        mime="text/csv",
                    )

        elif export_format == "JSON":
            # Export all simulation data as JSON
            import json

            def convert_for_json(obj):
                """Convert numpy arrays and pandas objects for JSON serialization."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict("records")
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            # Prepare export data
            export_data = {
                "simulation_results": convert_for_json(
                    st.session_state.simulation_results
                ),
                "patient_data": convert_for_json(
                    st.session_state.patient_data.to_dict("records")
                )
                if st.session_state.patient_data is not None
                else None,
                "therapy_schedule": convert_for_json(st.session_state.therapy_schedule),
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "simulator_version": "1.0",
            }

            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download Simulation Data (JSON)",
                data=json_str,
                file_name="cancer_simulator_results.json",
                mime="application/json",
            )

        elif export_format == "PDF Report":
            # Generate comprehensive PDF report
            from matplotlib.backends.backend_pdf import PdfPages
            import io

            # Create a PDF in memory
            pdf_buffer = io.BytesIO()

            with PdfPages(pdf_buffer) as pdf:
                # Page 1: Title and summary
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")

                # Title
                ax.text(
                    0.5,
                    0.9,
                    "Cancer Treatment Simulator Report",
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                )

                # Report info
                report_info = f"""
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Patient ID: {st.session_state.patient_data["patient_id"].iloc[0] if st.session_state.patient_data is not None else "N/A"}
Scenario: {st.session_state.patient_data["scenario"].iloc[0] if st.session_state.patient_data is not None else "N/A"}
Models Run: {", ".join(st.session_state.simulation_results.keys()) if st.session_state.simulation_results else "None"}
"""
                ax.text(0.1, 0.7, report_info, ha="left", va="top", fontsize=12)

                # Summary metrics
                if st.session_state.simulation_results:
                    ax.text(
                        0.1,
                        0.5,
                        "Summary Results:",
                        ha="left",
                        va="top",
                        fontsize=14,
                        fontweight="bold",
                    )

                    y_pos = 0.45
                    for (
                        model_type,
                        results,
                    ) in st.session_state.simulation_results.items():
                        if "metrics" in results:
                            metrics = results["metrics"]
                            summary_text = f"""
{model_type} Model:
  • Tumor Control Probability: {metrics.get("tumor_control_probability", "N/A"):.3f}
  • Tumor Reduction: {metrics.get("tumor_reduction", "N/A"):.1%}
  • Final Tumor Burden: {metrics.get("final_tumor_burden", "N/A"):.2e}
"""
                            ax.text(
                                0.1,
                                y_pos,
                                summary_text,
                                ha="left",
                                va="top",
                                fontsize=10,
                            )
                            y_pos -= 0.15

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Page 2+: Results plots
                for model_type, results in st.session_state.simulation_results.items():
                    if model_type == "ODE" and "results" in results:
                        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
                        data = results["results"]

                        # Tumor dynamics
                        axes[0, 0].plot(data["time"], data["tumor_cells"] / 1e6)
                        axes[0, 0].set_title("Tumor Volume")
                        axes[0, 0].set_ylabel("Volume (cm³)")
                        axes[0, 0].grid(True)

                        # Immune response
                        axes[0, 1].plot(data["time"], data["immune_cells"] / 1e3)
                        axes[0, 1].set_title("Immune Cells")
                        axes[0, 1].set_ylabel("Count (×10³)")
                        axes[0, 1].grid(True)

                        # Drug concentration
                        axes[1, 0].plot(data["time"], data["drug_concentration"])
                        axes[1, 0].set_title("Drug Concentration")
                        axes[1, 0].set_ylabel("Concentration")
                        axes[1, 0].set_xlabel("Time (days)")
                        axes[1, 0].grid(True)

                        # Treatment schedule
                        axes[1, 1].plot(
                            data["time"], data["rt_dose"], label="Radiation"
                        )
                        axes[1, 1].plot(
                            data["time"], data["immuno_dose"], label="Immunotherapy"
                        )
                        axes[1, 1].set_title("Treatment Schedule")
                        axes[1, 1].set_ylabel("Dose")
                        axes[1, 1].set_xlabel("Time (days)")
                        axes[1, 1].legend()
                        axes[1, 1].grid(True)

                        plt.suptitle(f"{model_type} Model Results", fontsize=16)
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

            pdf_buffer.seek(0)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name="cancer_simulator_report.pdf",
                mime="application/pdf",
            )

    def run(self):
        """Run the main application."""
        self.render_header()

        # Create main layout
        self.render_sidebar()
        self.render_main_content()

        # Footer
        st.divider()
        st.markdown(
            """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        Cancer Treatment Dynamics Simulator v1.0 | 
        Built with Streamlit | 
        Based on mechanistic models from cancer radiobiology research
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_dashboard_tab(self):
        """Render comprehensive dashboard."""
        st.markdown(
            '<h2 class="sub-header">Simulation Dashboard</h2>', unsafe_allow_html=True
        )

        # Quick status overview
        st.subheader("📋 Current Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            patient_status = (
                "✅ Loaded"
                if st.session_state.patient_data is not None
                else "❌ Not loaded"
            )
            st.metric("Patient Data", patient_status)

        with col2:
            therapy_status = (
                "✅ Configured"
                if st.session_state.therapy_schedule
                else "❌ Not configured"
            )
            st.metric("Therapy Schedule", therapy_status)

        with col3:
            simulation_count = len(st.session_state.simulation_results)
            st.metric("Simulations Run", f"{simulation_count}")

        with col4:
            current_model = st.session_state.current_model
            st.metric("Active Model", current_model)

        st.divider()

        # Quick actions
        st.subheader("🚀 Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate Sample Patient", type="secondary"):
                patient_data = self.data_handler.generate_synthetic_patient(
                    "DEMO_001", "standard"
                )
                st.session_state.patient_data = patient_data
                st.success("Demo patient generated!")
                st.rerun()

        with col2:
            if st.button("🔬 Run Quick Simulation", type="secondary"):
                if st.session_state.patient_data is not None:
                    self.run_simulation(90)  # Quick 90-day simulation
                    st.success("Quick simulation completed!")
                    st.rerun()
                else:
                    st.error("Please load patient data first!")

        with col3:
            if st.button("📈 Compare All Models", type="secondary"):
                if st.session_state.patient_data is not None:
                    with st.spinner("Running all models..."):
                        # Run all three models
                        original_model = st.session_state.current_model

                        for model in ["ODE", "PDE", "ABM"]:
                            st.session_state.current_model = model
                            self.run_simulation(120)

                        st.session_state.current_model = original_model
                        st.success("All models completed!")
                        st.rerun()
                else:
                    st.error("Please load patient data first!")

        # Recent activity
        if st.session_state.simulation_results:
            st.subheader("📊 Recent Simulation Results")

            for model_type, results in st.session_state.simulation_results.items():
                with st.expander(f"{model_type} Model Summary", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        if "metrics" in results:
                            metrics = results["metrics"]
                            st.write("**Key Metrics:**")
                            st.write(
                                f"• Tumor Control Probability: {metrics.get('tumor_control_probability', 'N/A'):.3f}"
                            )
                            st.write(
                                f"• Tumor Reduction: {metrics.get('tumor_reduction', 'N/A'):.1%}"
                            )
                            st.write(
                                f"• Final Burden: {metrics.get('final_tumor_burden', 'N/A'):.2e}"
                            )

                    with col2:
                        # Mini visualization
                        if model_type == "ODE" and "results" in results:
                            data = results["results"]
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=data["time"],
                                    y=data["tumor_cells"] / 1e6,
                                    mode="lines",
                                    name="Tumor Volume",
                                )
                            )
                            fig.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=20, b=0),
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)

        # System information
        st.subheader("ℹ️ System Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write("**Model Components:**")
            st.write("• ODE: Systemic tumor-immune dynamics")
            st.write("• PDE: Spatial diffusion processes")
            st.write("• ABM: Cell-level interactions")

        with info_col2:
            st.write("**Available Features:**")
            st.write("• Parameter sensitivity analysis")
            st.write("• Treatment protocol optimization")
            st.write("• Multi-format data export")
            st.write("• Patient cohort generation")


def main():
    """Main entry point."""
    app = CancerSimulatorGUI()
    app.run()


if __name__ == "__main__":
    main()
