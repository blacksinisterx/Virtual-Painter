# 🎨 CV Art Experiments

> **Transform your hands into digital brushes with cutting-edge computer vision**

*An innovative Python application that merges hand gesture recognition with artistic expression, creating an immersive drawing experience powered by MediaPipe and OpenCV.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-orange.svg)](https://mediapipe.dev)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ What Makes This Special

CV Art Experiments transforms your camera into a canvas, your hands into brushes, and your gestures into art. This isn't just another drawing app—it's a **computer vision playground** where technology meets creativity.

### 🌟 **Core Philosophy**
- **Gesture-Driven**: Natural hand movements control everything
- **Intelligent**: Smart color management and brush dynamics
- **Professional**: Grid tools and pixel art capabilities
- **Intuitive**: Glass morphism UI with seamless interactions

---

## 🚀 Features & Capabilities

### 🖐️ **Advanced Hand Tracking**
- **Real-time gesture recognition** using MediaPipe
- **Multi-finger detection** with smooth tracking
- **Gesture-based controls** for natural interaction
- **Confidence-based filtering** for precision

### 🎨 **Professional Drawing Engine**
- **Dynamic brush system** with pressure simulation
- **Intelligent color management** with persistence
- **Multiple drawing modes** (Freestyle, Pixel Art, Grid-Assisted)
- **Glass morphism UI** with modern aesthetics

### 📐 **Grid & Pixel Art Tools**
- **Dual grid systems**: Normal rectangular + Slanted diamond grids
- **Snap-to-grid functionality** for precise positioning
- **Pixel art mode** with customizable shapes
- **Optional black borders** for pixel definition
- **Grid overlay toggle** for clean drawing

### 🎯 **Smart Color Features**
- **Intelligent color wheel** with HSV control
- **Color persistence** across sessions
- **Popular colors display** with usage tracking
- **Automatic color harmonies** and suggestions

### ⚡ **Professional Controls**
- **Keyboard shortcuts** for all major functions
- **Canvas management** (clear, save, export)
- **Brush size controls** with visual feedback
- **Real-time info panel** with drawing statistics

---

## 🛠️ Technical Architecture

```
📁 Virtual-Painter/
├── 🐍 src/
│   ├── 🎯 main.py                 # Application orchestrator & entry point
│   ├── ⚙️ config/
│   │   ├── __init__.py
│   │   └── settings.py            # Configuration management
│   ├── 🧠 core/
│   │   ├── 🖥️ camera.py           # Camera interface & management
│   │   ├── 👋 hand_tracker.py     # MediaPipe gesture recognition
│   │   ├── 🎨 canvas.py           # Drawing engine & rendering
│   │   ├── 🖌️ brush_engine.py     # Dynamic brush system
│   │   ├── 📐 canvas_tools.py     # Grid tools & pixel art
│   │   ├── 🌈 color_manager.py    # Color intelligence & persistence
│   │   └── ⚠️ exceptions.py       # Error handling
│   ├── 🎭 gestures/
│   │   └── handlers.py            # Gesture interpretation
│   └── 🖼️ ui/
│       ├── 🎡 color_wheel.py      # Interactive color selection
│       ├── 📊 info_panel.py       # Real-time statistics
│       ├── 🎨 drawing_utils.py    # Visual utilities
│       └── 🖥️ ui_renderer.py      # Interface orchestration
├── 💾 assets/
│   ├── 🎨 color_preferences.json  # User color history
│   └── 🖼️ saved_drawings/         # Artwork gallery
└── 📋 requirements.txt            # Dependencies
```

### 🧮 **Core Components**

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **Camera System** | Video capture & processing | Real-time feed, frame management |
| **Hand Tracker** | Gesture recognition | MediaPipe integration, confidence filtering |
| **Canvas Engine** | Drawing & rendering | Multi-mode drawing, layer management |
| **Brush System** | Dynamic painting | Pressure simulation, smoothing |
| **Grid Tools** | Precision drawing | Snap-to-grid, pixel art support |
| **Color Manager** | Intelligent colors | Persistence, harmonies, popularity |
| **UI Renderer** | Interface management | Glass morphism, real-time updates |

---

## 🎮 Controls & Gestures

### ⌨️ **Keyboard Shortcuts**

| Key | Function | Description |
|-----|----------|-------------|
| `C` | Clear Canvas | Erase all artwork |
| `S` | Save Drawing | Export to PNG with timestamp |
| `G` | Toggle Grid | Show/hide grid overlay |
| `F` | Toggle Grid Type | Switch between normal/slanted grids |
| `P` | Pixel Art Mode | Switch to grid-based pixel drawing |
| `X` | Pixel Borders | Toggle black borders in pixel mode |
| `Z` | Undo | Undo last drawing action |
| `Y` | Redo | Redo previously undone action |
| `T` | Smart Colors | Toggle intelligent color features |
| `B` | Brush Mode | Toggle normal/watercolor mode |
| `+/-` | Brush Size | Increase/decrease brush size |
| `O/L` | Opacity | Increase/decrease brush opacity |
| `K/J` | Hardness | Increase/decrease brush hardness |
| `Q` | Exit | Graceful application shutdown |

### 👋 **Hand Gestures**

| Gesture | Action | Visual Feedback |
|---------|--------|----------------|
| **Index Finger Extended** | Draw | Brush trail with current color |
| **Peace Sign (Index + Middle)** | Navigate UI | Cursor movement indicator |
| **Fist** | Pause Drawing | Hand tracking continues |
| **Thumb + Index** | Color Selection | Hover over color wheel |

---

## 🚀 Quick Start Guide

### 📋 **Prerequisites**
- Python 3.8 or higher
- Webcam or external camera
- Good lighting conditions

### ⚡ **Installation**

```bash
# Clone the repository
git clone https://github.com/blacksinisterx/Virtual-Painter.git
cd Virtual-Painter

# Install dependencies
pip install -r requirements.txt

# Launch the application
python src/main.py
```

### 🎯 **First Use**
1. **Position yourself** 2-3 feet from the camera
2. **Ensure good lighting** for hand detection
3. **Extend your index finger** to start drawing
4. **Use two fingers** to navigate the color wheel
5. **Press `G`** to enable grid for precision work

---

## 🎨 Drawing Modes Explained

### 🖌️ **Freestyle Mode** (Default)
- **Natural drawing** with smooth brush strokes
- **Dynamic brush sizing** based on movement
- **Color blending** and opacity effects
- **Perfect for**: Artistic expression, sketching, organic shapes

### 📐 **Pixel Art Mode** (`P` key)
- **Grid-constrained drawing** for pixel-perfect art
- **Rectangle or diamond shapes** based on grid type
- **Snap-to-grid positioning** for precision
- **Optional black borders** (`X` key) for definition
- **Perfect for**: Pixel art, icons, retro graphics

### 🔷 **Grid-Assisted Mode** (`G` key)
- **Visual grid overlay** without constraints
- **Choose between normal or slanted grids** (`R` key)
- **Helpful guidelines** for proportion and alignment
- **Perfect for**: Technical drawings, architecture, planning

---

## 🌈 Color Intelligence System

### 🎡 **Interactive Color Wheel**
- **HSV-based selection** for natural color picking
- **Real-time preview** with gesture navigation
- **Smooth transitions** between color spaces
- **Visual feedback** for selected colors

### 🧠 **Smart Color Management**
- **Automatic persistence** - colors saved between sessions
- **Usage tracking** - popular colors bubble to the top
- **Color harmony suggestions** - complementary color hints
- **JSON-based storage** in `assets/color_preferences.json`

### 📊 **Color Analytics**
```json
{
  "history": [
    {
      "color": [255, 0, 0],
      "timestamp": 1753836052.5314398,
      "usage_count": 2,
      "context": "initial"
    }
  ],
  "favorites": [...],
  "popular_colors": [...],
  "recent_sessions": [...]
}
```

---

## 🔧 Advanced Configuration

### ⚙️ **Settings Customization**

Edit `src/config/settings.py` to customize:

```python
# Canvas Configuration
CANVAS_WIDTH = 1200          # Drawing area width
CANVAS_HEIGHT = 800          # Drawing area height
BACKGROUND_COLOR = (20, 20, 20)  # Dark theme background

# Hand Tracking
MIN_DETECTION_CONFIDENCE = 0.7   # Hand detection threshold
MIN_TRACKING_CONFIDENCE = 0.5    # Tracking stability

# Drawing Engine
MAX_BRUSH_SIZE = 50             # Maximum brush diameter
SMOOTH_FACTOR = 0.3             # Movement smoothing
GRID_SIZE = 20                  # Default grid cell size
```

### 🎨 **Custom Color Schemes**

Create custom color palettes in `assets/color_preferences.json`:

```json
{
  "themes": {
    "neon": ["#FF00FF", "#00FFFF", "#FFFF00"],
    "earth": ["#8B4513", "#228B22", "#87CEEB"],
    "sunset": ["#FF6347", "#FFD700", "#FF69B4"]
  }
}
```

---

## 🎯 Use Cases & Applications

### 🎓 **Educational**
- **Teaching geometry** with grid tools
- **Color theory exploration** with the color wheel
- **Computer vision concepts** demonstration
- **Art therapy** and creative expression

### 💼 **Professional**
- **UI/UX prototyping** with pixel art mode
- **Concept sketching** for designers
- **Interactive presentations** and demos
- **Art installations** with gesture control

### 🎮 **Creative Projects**
- **Digital art creation** with natural gestures
- **Pixel art game assets** with grid precision
- **Interactive art exhibits** using computer vision
- **Educational tools** for STEM learning

---

## 🛠️ Development & Contributing

### 🏗️ **Project Structure Philosophy**
- **Modular architecture** - each component has a single responsibility
- **Clean interfaces** - minimal coupling between modules
- **Extensible design** - easy to add new features
- **Professional patterns** - following Python best practices

### 🔧 **Adding New Features**

1. **New Gesture**: Add handler in `gestures/handlers.py`
2. **Drawing Mode**: Extend `canvas.py` with new rendering logic
3. **UI Element**: Create component in `ui/` directory
4. **Tool**: Add functionality to `canvas_tools.py`

### 🧪 **Testing Your Changes**

```bash
# Run with debug mode
python src/main.py --debug

# Test specific components
python -m src.core.hand_tracker
python -m src.ui.color_wheel
```

### 📝 **Code Style**
- **Type hints** for all function parameters
- **Docstrings** for public methods
- **Error handling** with custom exceptions
- **Configuration-driven** behavior

---

## 🎨 Gallery & Examples

### 🖼️ **Sample Creations**

Your drawings are automatically saved in `assets/saved_drawings/` with timestamps:

```
assets/saved_drawings/
├── Drawing_1754021024.png  # Freestyle sketch
├── Drawing_1754021139.png  # Pixel art creation
└── Drawing_1754021242.png  # Grid-assisted design
```

### 🎯 **Pro Tips**

| Tip | Description |
|-----|-------------|
| **Steady Hands** | Move slowly for smooth lines |
| **Good Lighting** | Bright, even lighting improves tracking |
| **Camera Height** | Position camera at chest level |
| **Background** | Plain backgrounds work best |
| **Practice** | Start with simple shapes, build complexity |

---

## 🔍 Troubleshooting

### 🚨 **Common Issues**

| Problem | Solution |
|---------|----------|
| **Hand not detected** | Check lighting, ensure hand is visible |
| **Erratic tracking** | Reduce background clutter, improve lighting |
| **Performance issues** | Close other applications, check CPU usage |
| **Colors not saving** | Ensure write permissions to `assets/` folder |
| **Grid misalignment** | Restart app to reset grid calculations |

### 🩺 **Debug Mode**

Enable debug output for troubleshooting:
```bash
python src/main.py --verbose
```

### 📊 **Performance Monitoring**

The info panel shows real-time statistics:
- **FPS**: Camera frame rate
- **Hand confidence**: Detection quality
- **Drawing points**: Canvas complexity
- **Memory usage**: System resources

---

## 🗺️ Roadmap & Future Features

### 🚀 **Next Release (v2.0)**
- [ ] **Multi-hand support** - Two-handed drawing
- [ ] **Gesture customization** - User-defined gestures
- [ ] **3D brush effects** - Depth-based brush changes
- [ ] **Recording mode** - Save drawing sessions as videos

### 🌟 **Future Vision**
- [ ] **Machine learning brushes** - AI-assisted drawing
- [ ] **Collaborative canvas** - Multi-user drawing
- [ ] **VR integration** - Immersive drawing experience
- [ ] **Mobile version** - iOS/Android compatibility

### 💡 **Community Ideas**
- [ ] **Custom brush shapes** - Import/create brush patterns
- [ ] **Layer system** - Multiple drawing layers
- [ ] **Animation tools** - Frame-by-frame animation
- [ ] **Export formats** - SVG, PDF, multiple formats

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🔧 **Development Setup**
```bash
# Fork and clone
git clone https://github.com/blacksinisterx/Virtual-Painter.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if available
```

### 📝 **Contribution Guidelines**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 🎯 **What We're Looking For**
- **Bug fixes** and performance improvements
- **New gesture types** and recognition patterns
- **UI/UX enhancements** and accessibility features
- **Documentation** improvements and tutorials
- **Test coverage** and quality assurance

---

### 🙏 **Acknowledgments**
- **MediaPipe** team for excellent hand tracking
- **OpenCV** community for computer vision tools
- **Python** ecosystem for making this possible
- **Contributors** who make this project better

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=blacksinisterx/Virtual-Painter&type=Date)](https://star-history.com/#blacksinisterx/Virtual-Painter&Date)

---

<div align="center">

### ✨ **Made with ❤️ and Computer Vision** ✨

*Transform your gestures into art • Push the boundaries of creative technology • Join the CV Art revolution*

</div>

---

**Ready to turn your hands into brushes?** 🎨✋

```bash
git clone https://github.com/blacksinisterx/Virtual-Painter.git
cd Virtual-Painter
pip install -r requirements.txt
python src/main.py
```

*Let your creativity flow through computer vision!* 🚀
