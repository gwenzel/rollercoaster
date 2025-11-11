# 🎢 AI Roller Coaster Designer

An advanced ML-powered roller coaster analysis system that predicts ride ratings from accelerometer data using a BiGRU neural network. Features include real-time physics simulation, rating prediction, and comprehensive data crawlers for training.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## 📁 Project Structure

```
rollercoaster/
├── app.py                          # Main Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── utils/                          # Core utilities
│   ├── model.py                    # Original thrill prediction model
│   ├── track.py                    # Track generation and physics
│   ├── visualize.py                # Plotting functions
│   ├── bigru_predictor.py          # BiGRU score prediction
│   └── accelerometer_transform.py  # Coordinate transformation (Frenet-Serret)
│
├── models/                         # Trained ML models
│   └── bigru_score_model.pth       # BiGRU model weights
│
├── scripts/                        # Standalone scripts
│   ├── bigru_score_predictor.py    # BiGRU model definition & training
│   ├── train_bigru_model.py        # Training pipeline
│   ├── create_dummy_model.py       # Generate test model
│   ├── test_*.py                   # Various test scripts
│   ├── create_*.py                 # Data processing scripts
│   └── show_*.py                   # Visualization scripts
│
├── ratings_data/                   # Rating & mapping data
│   ├── complete_coaster_mapping.csv        # Master mapping file (1,299 coasters)
│   ├── rating_to_rfdb_mapping_enhanced.csv # Rating-to-RFDB mapping
│   ├── coaster_name_mapping_rfdb.py        # Name mapping functions
│   ├── coaster_id_to_url_mapping.csv       # Coaster URLs
│   ├── all_reviews/                        # Review data
│   ├── star_ratings_per_rc/                # Rating distributions
│   ├── tests/                              # Test files
│   └── *.py, *.md                          # Scripts & documentation
│
├── rfdb_csvs/                      # RFDB accelerometer data (2,088 files)
│   └── [park]/[coaster]/*.csv      # 3-axis accelerometer recordings
│
├── crawler/                        # Web scraping tools
│   ├── captaincoaster/             # Captain Coaster rating crawler
│   ├── rideforcesdb/               # RFDB accelerometer crawler
│   └── shared/                     # Shared utilities
│
├── docs/                           # Documentation
│   ├── BIGRU_INTEGRATION.md        # BiGRU integration guide
│   ├── BIGRU_README.md             # BiGRU model documentation
│   ├── INTEGRATION_SUMMARY.md      # System integration overview
│   ├── RFDB_MAPPING_COMPLETE.md    # RFDB mapping documentation
│   └── COMPLETE.md                 # Complete feature documentation
│
└── accel_data/                     # Sample acceleration data
```

## 🌟 Features

### 🤖 BiGRU Score Prediction
- **Deep Learning Model**: BiGRU (Bidirectional Gated Recurrent Unit) neural network
- **Input**: 3-axis accelerometer data (Lateral, Vertical, Longitudinal)
- **Output**: Predicted star rating (0.5-5.0 stars) based on ride forces
- **Training Data**: 1,299 coasters with both accelerometer data and ratings
- **Total Samples**: 3,700 accelerometer recordings from real rides

### 🎢 Interactive Track Designer
- **Modular Track Elements**: Climb, drop, loop, curve, helix, hill sections
- **Real-Time Physics**: Energy conservation, velocity, acceleration calculations
- **3D Visualization**: Interactive Plotly plots with G-force coloring
- **Live Rating Prediction**: BiGRU model predicts rating as you design

### 📊 Comprehensive Dataset
- **Captain Coaster Ratings**: 1,768 coasters with detailed rating distributions
  - Average ratings, total review counts
  - Rating breakdown by stars (0.5★ to 5★)
- **RFDB Accelerometer Data**: 794 unique coasters across 152 parks
  - 2,088 CSV files with 3-axis acceleration recordings
  - Lateral, Vertical, Longitudinal g-forces
- **Complete Mapping**: 1,299 coasters with both ratings AND accelerometer data (73.5% coverage)

### 🕷️ Data Crawlers
- **Captain Coaster Crawler**: Extracts rating distributions from JavaScript
- **RFDB Crawler**: Downloads accelerometer data from RideForcesDB
- **Automated Mapping**: Fuzzy string matching with park hierarchy understanding

### 🔬 Physics Transformation
- **Frenet-Serret Frame**: Differential geometry for coordinate transformation
- **Track → Accelerometer**: Converts absolute coordinates to relative g-forces
- **Rider Reference Frame**: Tangent (forward), Normal (centripetal), Binormal (lateral) vectors

## � Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gwenzel/rollercoaster.git
cd rollercoaster

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Train BiGRU Model

```bash
# Option 1: Use the training script
cd scripts
python train_bigru_model.py

# Option 2: Create dummy model for testing
python create_dummy_model.py
```

## 📋 Dependencies

Key requirements (see `requirements.txt` for full list):

```txt
# Core
streamlit>=1.51.0
torch>=2.9.0
pandas>=2.0.0
numpy>=1.24.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0

# ML & Processing
scikit-learn>=1.3.0
scipy>=1.16.3
tqdm>=4.67.1

# Web Scraping
beautifulsoup4>=4.12.0
requests>=2.31.0
```

## � Using the Complete Dataset

The master dataset `ratings_data/complete_coaster_mapping.csv` contains everything you need:

```python
import pandas as pd

# Load the complete mapping
df = pd.read_csv('ratings_data/complete_coaster_mapping.csv')

# Example: Get coasters with high ratings and multiple recordings
top_coasters = df[
    (df['avg_rating'] >= 4.5) & 
    (df['csv_count'] >= 3)
].sort_values('avg_rating', ascending=False)

print(f"Found {len(top_coasters)} top-rated coasters with multiple recordings")

# Example: Load accelerometer data for a specific coaster
coaster = df.iloc[0]  # Highest rated
accel_path = f"../{coaster['full_path']}"
print(f"Coaster: {coaster['coaster_name']}")
print(f"Rating: {coaster['avg_rating']} stars ({coaster['total_ratings']} ratings)")
print(f"Accelerometer files: {coaster['csv_count']}")
```

### Available Columns

**Identifiers**: `coaster_id`, `coaster_name`, `ratings_coaster`, `ratings_park`

**Ratings**: `avg_rating`, `total_ratings`, `pct_0.5_stars` through `pct_5.0_stars`, `count_0.5_stars` through `count_5.0_stars`

**Accelerometer Data**: `rfdb_park_folder`, `rfdb_coaster_folder`, `csv_count`, `full_path`

**Quality Metrics**: `coaster_similarity`, `park_similarity`, `combined_similarity`, `match_type`

## 🎮 Web Application Usage

1. **Design Track**: Use modular track builder to add climb, drop, loop, curve, helix, and hill elements
2. **Adjust Parameters**: Modify length, height, angle, radius for each element
3. **View Physics**: Real-time velocity and acceleration calculations
4. **Get Rating**: BiGRU model predicts rating from simulated accelerometer data
5. **Visualize**: Interactive 3D plot with G-force coloring
- Hill Amplitude: 5m
- Expected Thrill: ~3-4/10

**Extreme Thrill Ride:**
- Peak Height: 90m
- Drop Angle: 80°
- Loop Radius: 8m
- Number of Hills: 5
- Hill Amplitude: 10m
- Expected Thrill: ~7-9/10

**Classic Looper:**
- Peak Height: 50m
- Drop Angle: 70°
- Loop Radius: 10m
- Number of Hills: 3
- Hill Amplitude: 8m
- Expected Thrill: ~5-6/10

## 📁 Project Structure

```
rollercoaster/
├── app.py                          # Main Streamlit application
├── utils/
│   ├── track.py                    # Track generation & physics
│   ├── model.py                    # Thrill prediction model
│   ├── visualize.py                # Plotly visualization
│   └── __pycache__/
├── requirements.txt                # App dependencies
├── requirements_crawler.txt        # Crawler dependencies
├── crawler.py                      # Web crawler (separate feature)
└── README.md                       # This file
```

## 🔧 Technical Details

### Track Generation (`utils/track.py`)

The track consists of 5 sections:
1. **Climb**: Linear ascent to peak height
2. **Drop**: Steep descent at specified angle
3. **Loop**: Circular vertical loop
4. **Hills**: Sinusoidal undulations
5. **Flat**: Final horizontal section

### Physics Calculations

**Velocity**: Calculated using conservation of energy
```python
v = sqrt(2 * g * Δh)
```

**Acceleration**: Derived from velocity gradient
```python
a = dv/dx
```

**Curvature**: Second derivative of track profile
```python
κ = d²y/dx²
```

### Thrill Prediction Model (`utils/model.py`)

Current model uses a weighted heuristic:
```python
thrill = 0.02 * max_height 
       + 0.5 * max_slope 
       + 0.1 * num_hills 
       + 0.3 * (1/loop_radius)
```

**Future Enhancement**: Replace with ML model trained on real coaster data (use `crawler.py` to collect training data!)

## 🎨 Visualization

The app uses Plotly for interactive 3D-like track visualization:
- **Color Mapping**: Turbo colormap (blue → green → yellow → red)
- **Hover Info**: Distance and height at any point
- **Zoom/Pan**: Interactive controls for detailed inspection
- **Responsive**: Adapts to window size

## 🔮 Future Enhancements

### Planned Features
- [ ] **Real ML Model**: Train on actual coaster review data
- [ ] **3D Visualization**: True 3D track with banking
- [ ] **G-Force Display**: Real-time G-force indicator
- [ ] **Safety Checks**: Validate design constraints
- [ ] **Export Options**: Save designs as images/data
- [ ] **Comparison Mode**: Compare multiple designs
- [ ] **User Ratings**: Collect user feedback
- [ ] **Physics Presets**: Quick templates (wooden, steel, launch, etc.)

### Integration with Crawler
Use the `crawler.py` script to:
1. Collect ~74,000 real roller coaster reviews
2. Extract features from highly-rated coasters
3. Train ML model on real-world data
4. Improve thrill prediction accuracy

```bash
# Collect training data
python quick_scrape.py medium

# Use the data for ML training
python train_model.py captaincoaster_medium.csv
```

## 🐛 Troubleshooting

**Issue**: Streamlit not found
```bash
# Solution: Install streamlit
pip install streamlit
```

**Issue**: Module import errors
```bash
# Solution: Ensure you're in the correct directory
cd rollercoaster
python -m streamlit run app.py
```

**Issue**: Blank visualization
```bash
# Solution: Check plotly version
pip install --upgrade plotly
```

**Issue**: ScriptRunContext warning
```bash
# Solution: This is harmless and can be ignored
# Or suppress with: warnings.filterwarnings('ignore')
```

## 🎯 Performance Tips

- **Faster Updates**: Reduce track point density in `generate_track()`
- **Smoother Animation**: Adjust number of points per section
- **Memory Usage**: Clear cache with `st.cache_data.clear()`

## 📊 Data Export

To save your designs programmatically:

```python
import pandas as pd

# Generate track
track_df = generate_track(50, 70, 10, 3, 8)

# Save to CSV
track_df.to_csv('my_coaster_design.csv', index=False)

# Save features
features = compute_features(track_df, 10, 3)
pd.DataFrame([features]).to_json('my_coaster_features.json')
```

## 🤝 Contributing

Want to improve the app?
1. Add new track elements (corkscrews, inversions)
2. Implement better physics (banking, friction)
3. Create ML model with real data
4. Design new visualizations
5. Add user authentication and save features

## 📝 License

This project is for educational and personal use.

## 🎢 Have Fun Designing!

Create your dream roller coaster and see how thrilling it would be!

---

**Quick Commands:**
```bash
# Start the app
streamlit run app.py

# Collect real coaster data (for ML training)
python quick_scrape.py small

# Run tests
python test_crawler.py
```

**Need Help?** Check the inline code documentation or adjust parameters gradually to understand their effects!
