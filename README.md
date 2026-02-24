# MNIST Live Draw

A lightweight, single-page MNIST live drawing demo with real-time classification, transformation controls, and previews of the transformed input and the 28×28 model input.

## How to run
1. Serve locally:

```bash
python -m http.server 8000
```

2. Open `http://localhost:8000` in a browser.

You can also upload these files to GitHub Pages and the app will run as a static site.

## Model source
- TensorFlow.js MNIST model (10-class) hosted via GitHub Pages:
  - `https://iwatake2222.github.io/tfjs_study/mnist/conv_mnist_tfjs/model.json`

## Preprocessing pipeline
1. Draw in white on a black canvas (or inverted if the toggle is enabled).
2. Apply transformations: rotation, scale, translation, blur, and optional inversion.
3. Find the digit’s bounding box in the transformed image.
4. Scale the digit to fit a 20×20 box and center it in a 28×28 frame.
5. Normalize pixel values to `[0, 1]` and feed a `[1, 28, 28, 1]` tensor into the model.

## Notes on transformations
- Rotations, scale changes, and translations can move strokes out of the canonical MNIST alignment, lowering confidence.
- Blur or inversion often reduces confidence because the model expects sharp white strokes on a dark background.
