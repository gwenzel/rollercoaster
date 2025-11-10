# 🎢 AI Roller Coaster Designer

An interactive Streamlit web application that lets you design custom roller coasters with real-time physics simulation and AI-powered thrill prediction.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🌟 Features

### Interactive Design Parameters
- **Peak Height**: Adjust the initial climb height (20-100m)
- **Drop Angle**: Control the steepness of the main drop (30-85°)
- **Loop Radius**: Customize the loop size (5-20m)
- **Number of Hills**: Add hills to the track (1-5)
- **Hill Amplitude**: Control hill height variation (3-10m)

### Real-Time Visualization
- **3D Track Profile**: Interactive Plotly visualization
- **Acceleration Coloring**: Track colored by G-force intensity
  - Blue: Lower acceleration
  - Red/Pink: Higher acceleration (more intense!)
- **Dynamic Updates**: Changes reflect instantly as you adjust parameters

### Physics Simulation
- **Energy Conservation**: Realistic velocity calculations based on height
- **Acceleration Analysis**: G-force computation along the track
- **Track Geometry**: Accurate slope and curvature calculations

### AI Thrill Prediction
- **Automated Rating**: ML model predicts thrill level (0-10 scale)
- **Feature Engineering**: Analyzes track characteristics:
  - Total track length
  - Maximum height
  - Maximum slope
  - Mean curvature
  - Loop properties
  - Hill count

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.17.0
matplotlib>=3.7.0
```

## 🚀 Installation

1. **Clone or download the repository**
   ```bash
   cd rollercoaster
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or using conda:
   ```bash
   conda install streamlit numpy pandas plotly matplotlib
   ```

3. **Verify installation**
   ```bash
   streamlit --version
   ```

## 🎮 Usage

### Starting the App

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Adjust Parameters** - Use the sidebar sliders to modify track design
2. **View Track** - See the roller coaster profile with acceleration coloring
3. **Check Thrill Score** - View the AI-predicted thrill rating
4. **Inspect Features** - Examine the computed track characteristics

### Example Configurations

**Gentle Family Coaster:**
- Peak Height: 30m
- Drop Angle: 40°
- Loop Radius: 15m
- Number of Hills: 2
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
