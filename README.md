# 🎢 AI Roller Coaster Designer

An interactive web application for designing roller coasters and predicting rider experience ratings using machine learning. Build custom coasters with modular blocks, visualize G-forces in real-time, and get instant AI-powered ratings based on real-world accelerometer data.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-9CDCFE?style=for-the-badge&logo=lightgbm&logoColor=white)

## ✨ Features

- **🎨 Interactive Track Builder**: Design coasters using 10 modular building blocks (lifts, drops, loops, hills, etc.)
- **🤖 AI Rating Prediction**: LightGBM model trained on 1,299 real coasters predicts fun ratings (1-5 stars)
- **⚡ Real-Time Physics**: Energy conservation, G-force calculations, and safety analysis
- **📊 Rich Visualizations**: Track profiles, G-force plots, airtime timelines, and interactive 3D views
- **🏆 Leaderboard**: Submit designs and compare with other users and real-world coasters
- **📈 Pareto Front Analysis**: Visualize the trade-off between fun and safety

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rollercoaster.git
cd rollercoaster

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 🎮 How to Use

1. **Design Your Coaster**: Use the sidebar to add building blocks (lift hills, drops, loops, etc.)
2. **Adjust Parameters**: Modify height, length, radius, and other properties for each block
3. **Generate Track**: Click "Generate Track" to compute physics and get AI rating
4. **View Results**: See G-force plots, airtime analysis, and predicted fun rating
5. **Submit to Leaderboard**: Save your design and compare with others

### Building Blocks

- **⛰️ Lift Hill**: Chain lift to initial height
- **⬇️ Vertical Drop**: Steep initial drop
- **🔄 Loop**: Clothoid loop element
- **🎈 Airtime Hill**: Floater airtime moment
- **🌀 Spiral**: Helix/corkscrew element
- **🐰 Bunny Hop**: Quick airtime bump
- **↪️ Banked Turn**: High-speed turn
- **🚀 Launch**: Magnetic acceleration boost
- **➡️ Flat Section**: Straight section
- **🛑 Brake Run**: Final braking section

## 🤖 Machine Learning Model

The rating prediction uses a **LightGBM gradient boosting model** trained on:
- **1,299 roller coasters** with paired accelerometer data and user ratings
- **26 engineered features** including G-forces, airtime, vibration, and metadata
- **Test R² = 0.73**, MAE = 0.24 stars

The model processes rider-frame accelerometer data (vertical, lateral, longitudinal G-forces) and predicts ratings on a 1-5 star scale that correlates with actual rider experiences.

## 📊 Data Sources

- **RideForcesDB**: 2,088 accelerometer recordings from 794 coasters across 152 parks
- **Captain Coaster**: 1,768 coasters with detailed rating distributions
- **Mapped Dataset**: 1,299 coasters with both accelerometer data and ratings (73.5% coverage)

## 🏗️ Project Structure

```
rollercoaster/
├── app.py                    # Main Streamlit app
├── app_builder.py            # Builder page implementation
├── pages/                    # Streamlit pages
│   ├── 01_Builder.py        # Track builder interface
│   ├── 02_RFDB_Data.py      # Real coaster data viewer
│   └── 03_Leaderboard.py    # Submissions leaderboard
├── utils/                    # Core utilities
│   ├── lgbm_predictor.py    # LightGBM model integration
│   ├── acceleration.py      # Physics calculations
│   ├── track_blocks.py      # Building block definitions
│   └── submission_manager.py # Leaderboard management
├── models/                   # Trained ML models
│   └── lightgbm/            # LightGBM model files
├── scripts/                  # Utility scripts
├── submissions/              # User submissions (JSON)
└── docs/                     # Documentation
```

## 🔧 Technical Details

### Physics Simulation

- **Energy Conservation**: Velocity computed from potential energy (95% efficiency)
- **Frenet-Serret Frame**: Coordinate transformation to rider reference frame
- **G-Force Calculation**: Tangential, centripetal, and gravitational accelerations
- **Realistic Parameters**: Mass, drag, friction, and launch sections

### Safety Analysis

Continuous safety scoring with penalty functions for:
- Vertical G-forces > 2.0g (increasing penalties above 3.0g, 4.0g, 5.0g, 7.0g)
- Lateral G-forces > ±2.0g (severe penalties above ±5.0g)
- Longitudinal G-forces > ±3.0g

## 📋 Requirements

Key dependencies:
- `streamlit>=1.28.0`
- `lightgbm>=4.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `plotly>=5.17.0`
- `scipy>=1.10.0`

See `requirements.txt` for the complete list.

## 🎯 Features in Detail

### Track Design
- Modular block-based construction
- Real-time parameter adjustment
- Automatic smoothing at transitions
- Random template generation
- 3D track visualization

### Physics & Analysis
- Energy conservation calculations
- G-force analysis (vertical, lateral, longitudinal)
- Airtime detection (floater, flojector, ejector)
- Safety score computation
- Vibration and jerk metrics

### Visualization
- Interactive track profile plots
- G-force time series with safety zones
- Airtime timeline visualization
- 3D track preview
- Pareto front plots (leaderboard)

## 📚 Documentation

- **Demo Description**: See `docs/DEMO_DESCRIPTION.md` for a detailed technical overview
- **Deployment Guide**: See `docs/DEPLOYMENT_GUIDE.md` for deployment instructions
- **RFDB Data**: See `docs/RFDB_MAPPING_COMPLETE.md` for data source information

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- New building block types (corkscrews, dive loops, etc.)
- Enhanced physics (vehicle dynamics, wind effects)
- 3D track design with full banking control
- Personalized rating predictions
- Multi-objective optimization

## 📝 License

This project is for educational and personal use.

## 🎢 Have Fun!

Design your dream roller coaster and see how thrilling it would be!

---

**Quick Commands:**
```bash
# Start the app
streamlit run app.py

# View real coaster data
# Navigate to "RFDB Data" page in the app

# Check leaderboard
# Navigate to "Leaderboard" page in the app
```
