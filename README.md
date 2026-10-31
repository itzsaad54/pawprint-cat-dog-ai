# PAWPRINT — Cat or Dog, decided by AI

A black-and-white web app that classifies a photo as **cat** or **dog** and shows how confident the model is. The model runs **entirely in your browser** using ONNX Runtime Web, so no images are uploaded anywhere.

**Live demo:** _add your Vercel link here_

## Features
- Drag & drop, file picker, paste (Ctrl+V) or camera capture
- Verdict with confidence bar and per-class probabilities
- Prediction history (saved in the browser)
- Light / dark (monochrome) theme toggle
- No backend, no build step — static files only

## Model
| | |
|---|---|
| Architecture | MobileNetV2 (ImageNet pre-trained), transfer learning |
| Input | 224 × 224 RGB |
| Output | 1 sigmoid neuron (below 0.5 = cat, above 0.5 = dog) |
| Training | 1,000 images (800 train / 200 validation), 500 per class |
| Test set | 200 images (100 cats, 100 dogs) |
| **Test accuracy** | **98.5%** (measured on the original Keras model) |

Training had two stages: the base was frozen first (lr 1e-3), then the last 30 layers were fine-tuned (lr 1e-5). Data augmentation: horizontal flip, rotation and zoom. The full notebook is in this repo.

## Project structure
```
index.html        UI and styles
script.js         Image handling and ONNX inference
model/model.onnx  Trained model
```

## Run locally
The model must be served over HTTP (opening `index.html` by double-click will not load it).

```bash
python -m http.server 8000
```
Then open <http://localhost:8000>. The status badge should read **"Model ready · browser"**.

## Deploy on Vercel
1. Push this repo to GitHub.
2. In Vercel: **Add New → Project** → import the repo.
3. Framework Preset: **Other**. Leave Root Directory as `./` and leave build/output settings empty.
4. Click **Deploy**.

## Configuration
At the top of `script.js`:
```js
MODEL_URL: 'model/model.onnx',
PREPROCESS: 'raw',   // 'raw' (0-255), 'mobilenet' (x/127.5-1) or 'unit' (x/255)
DOG_INDEX: 1,
```
If predictions look wrong, try a different `PREPROCESS` value.

## Tech stack
HTML, CSS, vanilla JavaScript, [ONNX Runtime Web](https://onnxruntime.ai/), TensorFlow / Keras (training).

## Limitations
- Only distinguishes cats from dogs; any other image will still be forced into one of the two classes.
- Trained on a small dataset (1,000 images), so unusual breeds, poses or lighting may be misclassified.

## References
- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381)
- [Keras MobileNet docs](https://keras.io/api/applications/mobilenet/)
- [TensorFlow transfer learning guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
