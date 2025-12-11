# 🎢 Interactive Roller Coaster Drawing App

## Overview

A user-friendly Streamlit application that lets you **draw your own roller coaster** and instantly get **AI-powered rating predictions**!

## ✨ Features

### 🎨 Easy Design Methods

1. **Quick Templates**
   - Classic Loop: Traditional loop coaster
   - Mega Drop: Extreme drops and high speeds
   - Twister: Multiple curves and banks
   - Family Friendly: Gentle hills
   - Custom: Fully customizable

2. **Manual Point Entry**
   - Enter control points manually
   - Fine-tune each height and distance
   - Perfect for precise designs

3. **Canvas Drawing** (Coming Soon)
   - Draw with your mouse directly
   - Free-form track creation

### 🤖 AI Rating Prediction

- **Instant Analysis**: Get ratings as soon as you generate
- **BiGRU Neural Network**: Trained on 1,299 real coasters
- **Real G-Force Analysis**: See actual rider accelerations
- **Design Tips**: Get suggestions to improve your rating

### 📊 Real-Time Feedback

- Track profile visualization
- G-force graphs (Lateral, Vertical, Longitudinal)
- Ride statistics (length, height, duration)
- Rating interpretation (from world-class to needs work)

## 🚀 How to Use

### Launch the App

```bash
# Make sure you're in the rollercoaster directory
cd c:\Users\Lenovo\rollercoaster

# Run the drawing app (on a different port than main app)
streamlit run app_drawing.py --server.port 8502
```

The app will open at: `http://localhost:8502`

### Design Your Coaster

1. **Choose a template** or enter manual points in the sidebar
2. **Adjust parameters**:
   - Track length multiplier (stretch/compress)
   - Height multiplier (scale vertical)
   - Initial speed (starting velocity)
   - Smoothness (curve interpolation)
3. **Click "Generate Ride!"** button
4. **View instant AI rating** and g-force analysis

### Improve Your Design

The app provides real-time tips:
- 💡 "Add steeper drops for more vertical g-forces"
- 💡 "Consider adding curves for lateral excitement"
- 💡 "Add more height variation for better ratings"

## 🎯 Template Guide

### Classic Loop (4.0-4.5 ⭐)
- Traditional vertical loop design
- Moderate intensity
- Good for beginners

### Mega Drop (4.5-4.9 ⭐)
- Extreme vertical drop
- High speed sections
- Maximum thrill factor

### Twister (3.5-4.2 ⭐)
- Multiple curves and banking
- Lateral g-forces
- Great for technique practice

### Family Friendly (2.5-3.5 ⭐)
- Gentle hills
- Smooth transitions
- Lower intensity

## 📊 Understanding Your Results

### Rating Scale

| Rating | Meaning | Examples |
|--------|---------|----------|
| 4.5-5.0 ⭐⭐⭐⭐⭐ | Outstanding! World-class | Zadra, VelociCoaster |
| 4.0-4.5 ⭐⭐⭐⭐⭐ | Excellent! | Steel Vengeance, Fury 325 |
| 3.5-4.0 ⭐⭐⭐⭐ | Great design | Most major park coasters |
| 3.0-3.5 ⭐⭐⭐ | Solid ride | Regional attractions |
| 2.5-3.0 ⭐⭐ | Decent | Small park rides |
| < 2.5 ⭐ | Needs improvement | Kiddie coasters |

### G-Force Analysis

The app shows three types of accelerations:

1. **Lateral** (Blue) - Side-to-side forces
   - Felt in curves and banking
   - High values = intense turns

2. **Vertical** (Red) - Up/down forces
   - Felt in drops and hills
   - Positive = pushed into seat
   - Negative = lifted from seat

3. **Longitudinal** (Green) - Forward/backward forces
   - Felt in acceleration/braking
   - Changes in speed

### Typical G-Force Ranges

- **Comfortable**: 0.5-2.0g vertical, < 1.0g lateral
- **Thrilling**: 2.0-4.0g vertical, 1.0-2.0g lateral
- **Extreme**: 4.0-5.0g vertical, 2.0-3.0g lateral
- **Dangerous**: > 5.0g (would need restraints/safety review)

## 🔬 How It Works

### The Pipeline

```
Your Drawing
    ↓
2D Profile (X, Y coordinates)
    ↓
3D Track Generation (interpolation, smoothing)
    ↓
Physics Simulation (velocity, acceleration)
    ↓
Frenet-Serret Transformation (world → rider frame)
    ↓
3-Axis Accelerometer Data (Lateral, Vertical, Longitudinal)
    ↓
BiGRU Neural Network
    ↓
Rating Prediction (0.5-5.0 stars)
```

### Technical Details

1. **Spline Interpolation**: Smooth curves using cubic splines
2. **Physics Engine**: Energy conservation, velocity calculations
3. **Coordinate Transform**: Differential geometry (Frenet-Serret frame)
4. **ML Model**: Bidirectional GRU with 128 hidden units
5. **Training Data**: 1,299 coasters, 3,700 recordings

## 🎨 Advanced Options

### Track Length Multiplier (0.5-2.0)
- **< 1.0**: Compress horizontally (faster ride)
- **1.0**: Normal length
- **> 1.0**: Stretch horizontally (longer ride)

### Height Multiplier (0.5-2.0)
- **< 1.0**: Shorter hills/drops
- **1.0**: Normal height
- **> 1.0**: Taller elements (more intense)

### Initial Speed (0-20 m/s)
- **0-5 m/s**: Slow start (like lift hill)
- **5-10 m/s**: Moderate launch
- **10-20 m/s**: Fast launch coaster

### Smoothness (0.5-5.0)
- **Low (0.5-1.5)**: Sharp transitions, jerky
- **Medium (1.5-3.0)**: Balanced
- **High (3.0-5.0)**: Very smooth, flowing

## 💡 Design Tips

### For High Ratings (4.5+)

1. **Start with height**: 60-80m initial drop
2. **Vary the elements**: Mix drops, hills, curves
3. **Control g-forces**: 
   - Peak vertical: 3-4g
   - Peak lateral: 1.5-2g
4. **Smooth transitions**: Use smoothness 2.0-3.0
5. **Moderate length**: 300-400m works well

### Common Mistakes

❌ **Too flat**: No height variation = boring  
✅ Add dramatic drops and hills

❌ **Too jerky**: Low smoothness = uncomfortable  
✅ Use smoothness 2.0+ for comfort

❌ **No variety**: Straight drops only  
✅ Mix different element types

❌ **Extreme g-forces**: > 6g = dangerous  
✅ Keep vertical under 5g, lateral under 3g

## 🆚 Comparison with Main App

### `app.py` (Original)
- Modular track builder with specific elements
- Precise control over each section
- More technical/engineering focused

### `app_drawing.py` (Drawing)
- Free-form design with templates
- Quick sketching and iteration
- More creative/artistic focused

**Both apps share the same BiGRU prediction model!**

## 🐛 Troubleshooting

### "Could not generate acceleration data"
- **Solution**: Increase smoothness parameter
- **Solution**: Add more control points
- **Solution**: Ensure track has height variation

### Rating seems too low/high
- **Remember**: Model trained on real coasters
- **Check**: G-force values (extreme = not realistic)
- **Verify**: Track has realistic proportions

### App is slow
- **Solution**: Reduce number of interpolation points
- **Solution**: Use simpler templates first
- **Solution**: Close other Streamlit apps

## 🚧 Roadmap / Future Features

- [ ] True canvas drawing with mouse/touch
- [ ] 3D track visualization (not just profile)
- [ ] Banking/tilting angles
- [ ] Multiple inversions (loops, corkscrews)
- [ ] Export track data to CSV
- [ ] Share designs with unique URLs
- [ ] Leaderboard of highest-rated community designs
- [ ] Mobile-responsive drawing
- [ ] VR preview of your coaster

## 📚 Related Documentation

- **Main App**: See `README.md`
- **BiGRU Model**: See `docs/BIGRU_README.md`
- **Dataset**: See `ratings_data/README.md`
- **Project Structure**: See `docs/PROJECT_STRUCTURE.md`

## 🎉 Try These Designs!

### World-Class Replica (4.8+ ⭐)
```
Template: Mega Drop
Height Multiplier: 1.8
Length Multiplier: 1.2
Smoothness: 2.5
Initial Speed: 8 m/s
```

### Family Fun (3.2-3.5 ⭐)
```
Template: Family Friendly
Height Multiplier: 0.8
Length Multiplier: 1.0
Smoothness: 4.0
Initial Speed: 5 m/s
```

### Extreme Twister (4.5+ ⭐)
```
Template: Twister
Height Multiplier: 1.5
Length Multiplier: 1.3
Smoothness: 2.0
Initial Speed: 10 m/s
```

---

**Ready to design?** Run `streamlit run app_drawing.py --server.port 8502` and create your masterpiece! 🎢✨
