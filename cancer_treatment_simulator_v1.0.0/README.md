# Cancer Treatment Simulator - Interactive Mathematical Modeling Platform v1.0.0

## 🎯 Overview
An advanced interactive platform for modeling cancer treatment dynamics using multiscale mathematical approaches. This simulator integrates Ordinary Differential Equations (ODE), Partial Differential Equations (PDE), and Agent-Based Models (ABM) to provide comprehensive treatment analysis and optimization capabilities for cancer research.

## 🧬 Core Mathematical Models
- **� ODE Models**: Systemic tumor-immune-drug interactions with treatment scheduling
- **🌊 PDE Models**: Spatial diffusion processes and transport phenomena  
- **🤖 ABM Models**: Cell-level behaviors and interactions with 3D visualization
- **� Optimized ABM**: High-performance agent-based modeling with vectorized operations

## 💻 System Requirements
- **Python**: 3.8 or higher (3.10+ recommended for optimal performance)
- **OS**: Windows, macOS, or Linux
- **Network**: Internet connection for initial dependency installation
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)
- **Storage**: ~1GB for complete environment and dependencies
- **RAM**: 8GB+ recommended (4GB minimum) for large ABM simulations

## 🚀 Quick Start Guide

### Windows Users (Recommended)
1. Extract this package to your preferred folder
2. **Double-click `START_SIMULATOR.bat`** for automatic port detection
3. First run will install dependencies (one-time setup)
4. The simulator opens automatically in your browser at `http://localhost:8502+`

### Alternative Launch Methods
- **Python 3.10**: Run `py -3.10 Install_Dependencies.py` then `START_SIMULATOR.bat`
- **Manual Launch**: `python -m streamlit run main.py --server.port 8502`
- **Development**: Use `python main.py` after installing requirements

### First-Time Setup
- The launcher automatically finds an available port (8502-8510)
- Dependencies are installed to `.cancer_simulator_env/` virtual environment
- No system-wide changes or administrator privileges required

## 📁 Package Contents
```
cancer_treatment_simulator_v1.0.0/
├── 🚀 START_SIMULATOR.bat        # Windows auto-port launcher
├── 📦 Install_Dependencies.py    # Python 3.10 dependency installer
├── 🧬 main.py                   # Main Streamlit application
├── 🤖 treatment_predictor.py    # AI treatment optimization engine
├── 📋 requirements.txt          # Python dependencies
├── 🔧 check_ports.bat           # Port availability checker
├── 🧹 cleanup_*.bat             # Environment cleanup utilities
├── 📊 models/                   # Mathematical modeling modules
│   ├── ode_model.py            # Ordinary Differential Equations
│   ├── pde_model.py            # Partial Differential Equations
│   ├── abm_model.py            # Agent-Based Model (standard)
│   └── abm_model_optimized.py  # High-performance ABM with vectorization
├── 💾 data/                     # Patient data handling
│   └── patient_data.py         # Synthetic patient generation
└── 📚 README.md                 # This documentation
```

## 🎯 Key Features

### 🔬 Mathematical Modeling Capabilities
- **📈 ODE Simulations**: Systemic tumor-immune dynamics with treatment scheduling
  - Continuous drug concentration modeling
  - Radiation dose-response relationships  
  - Immune system activation and exhaustion
  - Multi-parameter sensitivity analysis

- **🌊 PDE Simulations**: Spatial tumor growth and drug distribution
  - 2D/3D spatial diffusion processes
  - Drug transport and uptake modeling
  - Spatial radiation dose planning
  - Cross-sectional analysis tools

- **🤖 ABM Simulations**: Individual cell-level interactions
  - 3D grid-based cell positioning (up to 150×150×80)
  - Cell division, death, and immune cytotoxicity
  - High-performance optimized version with vectorized operations
  - Real-time 3D visualization and spatial analysis
  - Initial vs. final state comparison

### 🧠 AI-Powered Treatment Optimization
- **Smart Treatment Optimizer**: Multi-objective optimization balancing efficacy and toxicity
- **Treatment Protocol Comparison**: Side-by-side analysis of different approaches
- **Parameter Sensitivity Analysis**: Systematic exploration of parameter space
- **Personalized Recommendations**: Patient-specific treatment suggestions

### � Interactive Visualization & Analysis
- **Real-time Plotting**: Dynamic charts using Plotly for all model types
- **3D Spatial Views**: ABM cell distribution with scatter plots and heatmaps
- **Treatment Schedule Visualization**: Timeline-based therapy planning
- **Export Capabilities**: CSV, JSON, and PDF report generation

## 🔧 Usage Instructions

### Application Interface
The simulator provides a tabbed interface with the following main sections:

1. **📊 Simulation Tab**
   - **Single Patient Mode**: Individual patient simulation with custom parameters
   - **Patient Cohort Mode**: Batch analysis of multiple patient scenarios
   - **Model Configuration**: Real-time parameter adjustment for ODE/PDE/ABM models
   - **Treatment Parameters**: Radiotherapy, chemotherapy, and immunotherapy scheduling

2. **📈 Analysis Tab**
   - **Smart Treatment Optimizer**: AI-powered multi-objective optimization
   - **Treatment Comparison**: Side-by-side protocol analysis
   - **Sensitivity Analysis**: Parameter importance assessment
   - **Model Comparison**: Performance evaluation across ODE/PDE/ABM approaches

3. **👥 Cohort Tab**
   - **Patient Generation**: Create synthetic patient populations
   - **Cohort Analysis**: Statistical analysis across patient groups
   - **Treatment Response**: Population-level treatment effectiveness

4. **📁 Export Tab**
   - **Data Export**: CSV/JSON export for external analysis
   - **Report Generation**: PDF reports with visualizations
   - **Result Archiving**: Save simulation states for reproducibility

5. **🏠 Dashboard Tab**
   - **Quick Actions**: Rapid simulation setup and execution
   - **System Status**: Current simulation and data status
   - **Recent Results**: Access to previous simulation runs

### Model-Specific Features

**ODE Model Configuration:**
- Tumor growth rates, immune kill rates (log scale)
- Radiation sensitivity, drug uptake parameters
- Treatment timing and dosing schedules

**PDE Model Configuration:**
- Grid resolution and spatial discretization
- Diffusion coefficients for tumor and drug transport
- Spatial treatment planning and dose distribution

**ABM Model Configuration:**
- 3D grid dimensions (X/Y/Z sizing)
- Initial tumor radius and immune cell populations
- Cell division rates and cytotoxicity parameters
- Option to preview dynamics without therapy interference

## 🛠️ Troubleshooting

### Common Issues

## 🛠️ Troubleshooting

### Common Issues

**"Python Not Found" Error**
```bash
# Install Python 3.8+ from https://python.org
# Ensure "Add Python to PATH" is checked during installation
# Verify installation:
python --version
# or
py --version
```

**Port Already in Use**
- `START_SIMULATOR.bat` automatically finds available ports (8502-8510)
- Manually specify port: `python -m streamlit run main.py --server.port 8503`
- Close other Streamlit applications or restart system

**Installation/Dependency Issues**
```bash
# Clean install dependencies:
python Install_Dependencies.py

# Manual virtual environment setup:
python -m venv .cancer_simulator_env
.cancer_simulator_env\Scripts\activate  # Windows
source .cancer_simulator_env/bin/activate  # Unix/Mac
pip install -r requirements.txt
```

**Performance Issues (ABM Models)**
- Enable "High-Performance ABM" in sidebar for large simulations
- Reduce grid size for faster computation (e.g., 80×80×40 instead of 150×150×80)
- Use "Ignore therapy during ABM run" for parameter preview mode
- Ensure 8GB+ RAM for large-scale ABM simulations

**Browser/Display Issues**
- Use Chrome, Firefox, Safari, or Edge (latest versions)
- Enable JavaScript and disable ad blockers for localhost
- Clear browser cache if visualizations don't load
- Try different localhost port if connection fails

**Model-Specific Issues**
- **ODE**: Ensure immune kill rate is on log scale (-9 to -6 typical range)
- **PDE**: Verify grid size doesn't exceed memory limits
- **ABM**: Check that initial tumor radius fits within grid dimensions
- **ABM Visualization**: Use "3D Projection (All Slices)" if individual slices appear empty

### Advanced Configuration

**Environment Variables**
```bash
# Force specific Python version
set PYTHON_CMD=py -3.10  # Windows
export PYTHON_CMD=python3.10  # Unix/Mac

# Custom port range
set STREAMLIT_PORT_RANGE=8500-8520
```

**Development Mode**
```bash
# Run with debug logging
python -m streamlit run main.py --logger.level=debug

# Run with custom configuration
python -m streamlit run main.py --server.maxUploadSize=200 --server.port=8502
```

**Virtual Environment Management**
```bash
# Check environment status
ls .cancer_simulator_env/Scripts/  # Windows
ls .cancer_simulator_env/bin/      # Unix/Mac

# Manual environment activation
.cancer_simulator_env\Scripts\activate     # Windows
source .cancer_simulator_env/bin/activate  # Unix/Mac

# Verify installed packages
pip list | grep -E "(streamlit|pandas|numpy|plotly)"
```

## 📧 Support & Documentation

### Getting Help
1. **Built-in Help**: Each model and parameter includes tooltips and help text
2. **README Troubleshooting**: Check the troubleshooting section above
3. **Log Analysis**: Review terminal output for error messages
4. **Parameter Guidance**: Use suggested ranges provided in the interface

### Research Applications
- **Publication-Ready Output**: Export high-resolution plots and comprehensive reports
- **Reproducible Research**: Save simulation parameters and random seeds
- **Parameter Documentation**: Built-in help explains biological significance
- **Multi-Model Comparison**: Validate results across ODE/PDE/ABM approaches

### Technical Specifications
- **Mathematical Models**: Based on established cancer biology literature
- **Numerical Methods**: Scipy ODE solvers, finite difference PDE schemes, discrete ABM
- **Performance**: Optimized ABM with vectorized operations and spatial indexing
- **Validation**: Cross-model consistency checks and parameter sensitivity analysis

## 🔄 Updates and Maintenance

### Version Management
- **Current Version**: v1.0.0 (Mathematical modeling platform)
- **Update Process**: Download new distribution and replace folder
- **Environment**: Virtual environment recreated automatically on major updates
- **Backward Compatibility**: Results from v1.0.0 remain compatible

### Maintenance Tasks
- **No Regular Maintenance**: Self-contained virtual environment
- **Storage Cleanup**: Delete `.cancer_simulator_env/` to force clean reinstall
- **Cache Clearing**: Remove `__pycache__/` folders if needed
- **Log Rotation**: Terminal logs are not persistent between sessions

## 📜 Technical Information

### Dependencies
```
Core Framework:
- streamlit >= 1.28.0    # Web application framework
- plotly >= 5.17.0       # Interactive visualizations
- pandas >= 2.0.0        # Data manipulation
- numpy >= 1.24.0        # Numerical computing

Scientific Computing:
- scipy >= 1.11.0        # ODE/PDE solvers
- scikit-learn >= 1.3.0  # Machine learning for AI optimization
- numba >= 0.58.0        # JIT compilation for ABM performance

Visualization:
- matplotlib >= 3.7.0    # Static plotting
- seaborn >= 0.12.0     # Statistical visualization
```

### Architecture
- **Frontend**: Streamlit web interface with Plotly visualizations
- **Backend**: Python-based mathematical modeling engines
- **Models**: Modular design allowing easy switching between ODE/PDE/ABM
- **Data**: Synthetic patient generation and real data import capabilities
- **AI**: Scikit-learn based treatment optimization with multi-objective functions

### Performance Characteristics
- **ODE Models**: ~1-10 seconds for 180-day simulations
- **PDE Models**: ~10-60 seconds depending on grid resolution
- **Standard ABM**: ~30-300 seconds for medium grids (80×80×40)
- **Optimized ABM**: ~5-60 seconds with vectorized operations (2-5x speedup)
- **Memory Usage**: 100MB-2GB depending on model complexity and grid size

### Security & Privacy
- **Local Processing**: All computations performed locally, no data transmission
- **Isolated Environment**: Virtual environment prevents system conflicts
- **No External Dependencies**: Runs offline after initial setup
- **Data Privacy**: Patient data never leaves local machine