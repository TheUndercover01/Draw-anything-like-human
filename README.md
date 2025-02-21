# Sketch Animation GUI

A professional GUI application for converting photos to animated sketches using the Photo-Sketching model from Li et al. This application provides an intuitive interface for processing images into contour drawings and animating their creation.

<p align="center">

</p>

<p align="center">
  <img src="imgs/Screenshot 2025-02-21 at 8.38.01 PM.png" alt="Photo to Sketch Examples" width="800"/>
  <br>
  <em>The Sketch Animation GUI interface showing the image selector (left) and animation canvas (right)</em>
  
</p>
<p align="center">
  <img src="imgs/Screenshot 2025-02-21 at 8.39.36 PM.png" alt="GUI Interface" width="800"/>
  <br>
  <em>Example results showing original photos (top) and their corresponding sketches (bottom)</em>
  
</p>

## Overview

This application builds on the research from the paper "Photo-Sketching: Inferring Contour Drawings from Images" (WACV 2019) by providing a user-friendly interface for:

- Image selection and preview
- Photo-to-sketch conversion
- Animated sketch rendering
- Interactive stroke visualization

## Technical Foundation

This implementation uses the pre-trained model from the Photo-Sketching paper, which employs a sophisticated pipeline for converting photos to contour drawings. The original research:

```
@article{LIPS2019,
  title={Photo-Sketching: Inferring Contour Drawings from Images},
  author={Li, Mengtian and Lin, Zhe and M\v ech, Radom\'ir and and Yumer, Ersin and Ramanan, Deva},
  journal={WACV},
  year={2019}
}
```

Visit the [original repository](https://github.com/mtli/PhotoSketch) for more details about the underlying model.

## Installation

### Prerequisites

```bash
# Create conda environment
conda create -n sketch-gui python=3.8

# Activate environment
conda activate sketch-gui

# Install required packages
pip install torch torchvision opencv-python pillow tkinter numpy
```

### Model Setup

1. Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1TQf-LyS8rRDDapdcTnEgWzYJllPgiXdj/view)
2. Place the model file in the `output/pretrained/` directory
3. Ensure your directory structure matches:
```
sketch-gui/
├── output/
│   └── pretrained/
│       └── latest_net_G.pth
├── examples/
│   └── [your input images]
├── checkpoint/
└── main.py
```

## Usage

1. Run the application:
```bash
python main.py --examples
```

2. Interface Components:
   - Left Panel: Image selector showing available images
   - Right Panel: Animation canvas and controls
   - Bottom: Status bar for feedback

3. Controls:
   - "← Previous" / "Next →": Navigate through available images
   - "Select": Choose current image for processing
   - "Process Image": Convert photo to sketch
   - "Fast Draw": Quick rendering of the sketch
   - "Animated Draw": Stroke-by-stroke animation
   - "Clear": Reset the canvas

## Features

### Image Selection
- Permanent left panel for easy image browsing
- Image preview with maintained aspect ratio
- Image counter showing position in gallery
- Quick navigation between images

### Sketch Processing
- Integration with Photo-Sketching model
- Automatic image preprocessing
- Progress feedback during conversion

### Animation
- Two drawing modes:
  - Fast Draw: Immediate sketch rendering
  - Animated Draw: Progressive stroke animation
- Adjustable animation speed
- Interactive canvas clearing
- Professional stroke rendering

### User Interface
- Split panel design for efficient workflow
- Professional styling and layout
- Clear status feedback
- Responsive design
- Cross-platform compatibility

## Technical Details

### Model Architecture
The application uses the Photo-Sketching model which employs:
- A modified pix2pix architecture
- Specialized preprocessing for contour enhancement
- Custom post-processing for stroke extraction

### Animation Pipeline
The stroke animation system:
1. Processes the model output to identify connected components
2. Analyzes stroke thickness and continuity
3. Orders strokes for natural drawing progression
4. Renders strokes with smooth animation

## Contributing

Feel free to:
- Open issues for bugs or enhancement requests
- Submit pull requests with improvements
- Share examples of interesting use cases

## License

This project's GUI implementation is released under the MIT License. The underlying Photo-Sketching model follows its original license - please refer to their repository for details.

## Acknowledgments

- Original Photo-Sketching paper authors for their groundbreaking research
- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/) project which the original model was based on
- The PyTorch team for their excellent framework

## Citation

If you use this GUI or the underlying model in your work, please cite the original paper:

```bibtex
@article{LIPS2019,
  title={Photo-Sketching: Inferring Contour Drawings from Images},
  author={Li, Mengtian and Lin, Zhe and M\v ech, Radom\'ir and and Yumer, Ersin and Ramanan, Deva},
  journal={WACV},
  year={2019}
}
```
