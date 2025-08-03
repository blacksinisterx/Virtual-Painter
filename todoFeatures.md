# 🎨 Virtual Painter - Future Features TODO

## Overview
This document contains planned features and enhancements for future development sessions. Each feature includes implementation complexity, user impact, and technical considerations.

---

## 🖱️ **Feature 1: Mouse Drawing Support**
**Priority**: High | **Complexity**: Low | **Impact**: High

### Description
Enable traditional mouse-based drawing for accessibility and precision work.

### Implementation Details
- **Left Mouse Button**: Drawing mode (equivalent to DRAWING gesture)
- **Right Mouse Button**: Erasing mode (equivalent to ERASING gesture)
- **Mouse Wheel**: Brush/eraser size adjustment
- **Modifier Keys**: 
  - Ctrl + Mouse: Color selection mode
  - Shift + Mouse: Straight line drawing
  - Alt + Mouse: Eyedropper tool

### Technical Considerations
- Integrate with existing gesture processing pipeline
- Maintain current brush size and color systems
- Add mouse event handlers to main loop
- Ensure smooth cursor tracking at high DPI

### User Benefits
- Accessibility for users who prefer mouse input
- Precision drawing for detailed work
- Backup input method when camera tracking fails
- Familiar interface for traditional digital artists

---

## 🪞 **Feature 2: Mirror Drawing Modes**
**Priority**: Medium | **Complexity**: Medium | **Impact**: High

### Description
Symmetrical drawing modes for creating balanced artwork and mandala-style designs.

### Mirror Types
1. **Vertical Mirror (`|`)**: Left-right symmetry
2. **Horizontal Mirror (`-`)**: Up-down symmetry  
3. **Quadrant Mirror (`+`)**: Four-way symmetry (both vertical and horizontal)

### Implementation Details
- **Activation**: Keyboard shortcuts (M for vertical, H for horizontal, Q for quadrant)
- **Visual Feedback**: Semi-transparent guide lines showing mirror axes
- **Real-time Mirroring**: Each stroke duplicated across mirror axes instantly
- **Toggle System**: Cycle through modes: Off → Vertical → Horizontal → Quadrant → Off

### Technical Considerations
- Transform drawing coordinates based on active mirror mode
- Apply transformations to brush strokes in real-time
- Handle canvas boundaries when mirroring (clipping vs scaling)
- Ensure consistent brush properties across mirrored strokes
- Memory optimization for multiple simultaneous stroke rendering

### Advanced Features
- **Adjustable Mirror Lines**: Drag to reposition mirror axes
- **Kaleidoscope Mode**: 6-way or 8-way radial symmetry
- **Mirror Preview**: Show ghost preview of mirrored strokes before committing

---

## ✋ **Feature 3: Grab & Move Gesture System**
**Priority**: High | **Complexity**: Very High | **Impact**: Very High

### Description
Revolutionary gesture-based object manipulation system allowing users to grab and move connected drawing elements.

### Gesture Definition
- **Trigger**: All five fingers closed (fist gesture)
- **Grab Point**: Center of palm/hand
- **Activation**: Hover over drawing element for 3+ seconds
- **Control**: Move hand to translate grabbed object
- **Release**: Change to any other gesture type

### Drawing Element Detection
- **Connected Component Analysis**: Identify continuous drawing regions
- **Flood Fill Algorithm**: Determine boundaries of connected drawing matter
- **Bounding Box Calculation**: Efficient collision detection for grab targeting
- **Multi-Object Support**: Handle overlapping or adjacent drawing groups

### Implementation Phases

#### Phase 3.1: Foundation
- Implement fist gesture recognition in gesture detector
- Add connected component analysis to drawing engine
- Create object boundary detection system
- Build basic grab point collision detection

#### Phase 3.2: Interaction System
- Develop 3-second hover timer mechanism
- Implement object selection feedback (highlight, glow effect)
- Create smooth object translation system
- Add release gesture detection

#### Phase 3.3: Advanced Features
- **Object Rotation**: Twist hand to rotate grabbed objects
- **Object Scaling**: Pinch/expand gesture while grabbing
- **Multi-Object Selection**: Grab multiple objects simultaneously
- **Object Layering**: Z-order management for overlapping objects

### Technical Challenges
- **Real-time Object Segmentation**: Fast connected component analysis on large canvases
- **Smooth Hand Tracking**: Sub-pixel precision for natural object movement
- **Memory Management**: Efficient storage and manipulation of object data
- **Collision Detection**: Fast spatial queries for grab point intersection
- **Gesture State Management**: Complex state transitions between grab/move/release

### Visual Feedback Systems
- **Hover Indicator**: Progressive fill animation during 3-second hover
- **Grab Confirmation**: Object outline glow when successfully grabbed
- **Movement Trail**: Ghost preview showing where object will be placed
- **Gesture Guide**: Visual cues showing available gestures while grabbing

### Performance Optimizations
- **Spatial Indexing**: Quadtree or R-tree for fast object lookup
- **Dirty Region Tracking**: Only reprocess modified canvas areas
- **Background Processing**: Asynchronous object analysis
- **Gesture Prediction**: Anticipate grab attempts for smoother interaction

---

## 🎯 **Implementation Priority Order**

### Phase A: Quick Wins (Next Session)
1. **Mouse Drawing Support** - Low complexity, high immediate value
2. **Basic Mirror Modes** - Vertical and horizontal mirror functionality

### Phase B: Core Features (Future Sessions)
1. **Quadrant Mirror Mode** - Complete mirror system
2. **Grab Gesture Foundation** - Basic fist detection and object segmentation
3. **Basic Grab & Move** - Simple object translation

### Phase C: Advanced Systems (Long-term)
1. **Advanced Grab Features** - Rotation, scaling, multi-object support
2. **Performance Optimizations** - Spatial indexing, async processing
3. **Polish & Refinement** - Visual feedback, gesture prediction

---

## 🔧 **Technical Architecture Notes**

### Integration Points
- **Gesture System**: Extend existing GestureType enum with GRAB, MOUSE_DRAW, MOUSE_ERASE
- **Drawing Engine**: Add object segmentation and manipulation capabilities
- **Canvas System**: Implement object layer management
- **Input Handler**: Merge mouse and gesture input streams

### Data Structures
```python
class DrawingObject:
    """Represents a connected drawing element that can be manipulated"""
    boundaries: List[Point]
    pixels: Set[Point] 
    bounding_box: Rectangle
    center: Point
    is_grabbed: bool
    grab_offset: Point

class MirrorState:
    """Manages mirror drawing configuration"""
    mode: MirrorMode  # OFF, VERTICAL, HORIZONTAL, QUADRANT
    vertical_axis: int
    horizontal_axis: int
    show_guides: bool
```

### Performance Targets
- **Object Detection**: < 50ms for canvas analysis
- **Grab Response**: < 16ms from gesture to visual feedback
- **Movement Latency**: < 8ms from hand movement to object translation
- **Memory Usage**: < 100MB additional for object management system

---

## 📝 **Development Notes**

### Testing Strategy
- **Unit Tests**: Individual gesture recognition, object detection algorithms
- **Integration Tests**: End-to-end grab-and-move workflows
- **Performance Tests**: Large canvas stress testing, multi-object scenarios
- **User Experience Tests**: Gesture comfort, learning curve assessment

### Documentation Requirements
- **User Guide**: Gesture tutorials, mirror mode explanations
- **Developer Docs**: API documentation for object manipulation system
- **Performance Guide**: Optimization recommendations for complex scenes

---

*Last Updated: August 3, 2025*
*Next Review: Next development session*
