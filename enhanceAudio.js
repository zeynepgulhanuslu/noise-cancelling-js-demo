const fs = require("fs");
const ndarray = require("ndarray");
const wav = require("node-wav");
const ort = require('onnxruntime-web');
const { off } = require("process");
ort.env.wasm.numThreads = 1; 
const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

function getAudioBlock(data, sampleRate, ms, overlapMs) {
  const blockSize = Math.floor((sampleRate / 1000) * ms);
  const overlapSize = Math.floor((sampleRate / 1000) * overlapMs);
  const hopSize = blockSize - overlapSize;
  
  console.log("blockSize: " + blockSize);
  console.log("overlapSize: " + overlapSize);
  console.log("hopSize: " + hopSize);

  const chunks = [];
  let i = 0;
  while (i < data.length) {
    const start = i;
    const end = (i + blockSize < data.length) ? i + blockSize : data.length;
    const chunk = data.subarray(start, end);
    chunks.push(chunk);
    i += hopSize;
    
  }
  return chunks;
}

async function runInferenceWithChunks(modelPath, audioFilePath, chunkSizeInMs, overlapMs) {
  try {
    const buffer = fs.readFileSync(audioFilePath);
    const result = wav.decode(buffer);
    const noisy = result.channelData[0];
    const sr = result.sampleRate;
    console.log('audio file: ' + audioFilePath);
    console.log('input length: ' + noisy.length);
    console.log('sample rate: ' + sr);


    const batches = getAudioBlock(noisy, sr, chunkSizeInMs, overlapMs);
    const blockSize = Math.floor((sr / 1000) * chunkSizeInMs);
    const overlapSize = Math.min(100, Math.floor((sr / 1000) * overlapMs));

    const outputLength = noisy.length;
    const outputAudio = new Float32Array(outputLength);
    console.log("output length: " + outputLength);

    const session = await InferenceSession.create(modelPath);

    if (session) {
      console.log("Model loaded successfully");
    } else {
      console.log("Failed to load model");
    }
    const inputTensor = new Tensor("float32", new Float32Array(blockSize), [1, 1, blockSize]);
    const feeds = {
      'input': inputTensor,
    };

    let count = 0;
    for (const batch of batches) {
      const noisyBatch = ndarray(batch, [1, batch.length]);
      const noisyBatchFloat = convertFloatArray(noisyBatch.data);
      inputTensor.data.set(noisyBatchFloat);

      const option = createRunOptions();
      const outputMap = await session.run(feeds, option);

      const output = Array.from(outputMap.output.data);
      const offset = count * blockSize - count * Math.floor((sr / 1000) * overlapMs); // calculate the offset of the current block, taking overlap into account
      const end = Math.min(offset + output.length, outputAudio.length);
      outputAudio.set(output.slice(0, end - offset), offset);

      count += 1;
    }

    const outputBuffer = wav.encode([outputAudio], {
      sampleRate: sr,
      float: true,
    });

    const outputFilePath = audioFilePath.replace('.wav', '-enhanced-chunks-quantized.wav');
    fs.writeFileSync(outputFilePath, outputBuffer);

  } catch (e) {
    console.error(`Failed to inference ONNX model: ${e}.`);
  }
}

function convertFloatArray(x) {
  return Float32Array.from(x);
}

function createRunOptions() {
  return { logSeverityLevel: 0 };
}

const modelPath = process.argv[2] || 'data/dns64_quantized.onnx';
const audioFilePath = process.argv[3] || 'data/sample-small.wav';
const chunkSizeInMs = process.argv[4] || 300;
const overlapMs = process.argv[5] || 50;
runInferenceWithChunks(modelPath, audioFilePath, chunkSizeInMs, overlapMs);
