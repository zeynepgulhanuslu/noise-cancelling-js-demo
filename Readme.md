
## Noise Cancellation JavaScript Demo

In this project, a demo has been prepared for the usage of a sample noise cancellation model converted to the ONNX format using JavaScript.

The model used in this demo is called [Denoiser](https://github.com/facebookresearch/denoiser)

You can place the model under the data directory.

[Model](https://drive.google.com/file/d/1gSMqfu5jQ2tajOckVP5a08EB334Ro95g/view?usp=sharing)

You can perform the installation with the following commands:
    
## Installation
```bash
git clone https://github.com/zeynepgulhanuslu/noise-cancelling-js-demo.git
git lfs pull 
```

## Required Packages
You can install the required packages as follows:

```bash
npm install fs ndarray node-wav onnxruntime-web @types/node-wav
```

## Usage Example

You can clean an audio file by running the enhanceAudio.js file.

You can run it as an example with the following command:

```bash 
node enhanceAudio.js data/dns64.onnx sample-small.wav

```
You can also find the TypeScript version in the enhanceAudio.ts file.





