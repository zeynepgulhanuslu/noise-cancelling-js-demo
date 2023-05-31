import fs from "fs";
import * as wav from "node-wav";
import ort, { InferenceSession, Tensor,env } from "onnxruntime-web";

env.wasm.numThreads = 1;

/**
 * This function splits an audio array into smaller chunks with overlap. This way we can simulate it like real-time conversion.
 * @param data Audio array 
 * @param sampleRate Sampling rate of an audio file
 * @param ms Block size in milliseconds
 * @param overlapMs Overlap size in milliseconds
 */

function getAudioBlock(data, sampleRate, ms, overlapMs) {
  const blockSize = Math.floor((sampleRate / 1000) * ms);
  const overlapSize = Math.floor((sampleRate / 1000) * overlapMs);
  const hopSize = blockSize - overlapSize;
  
  console.log("blockSize: " + blockSize);
  console.log("overlapSize: " + overlapSize);
  console.log("hopSize: " + hopSize);

  const chunks: Float32Array[] = [];
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

function convertFloatArray(x: Float32Array): Float32Array {
    return x;
}
  
function createRunOptions(): any {
    // run options: please refer to the other example for details usage for run options
  
    // specify log verbose to this inference run
    return { backendHint:'wasm'};
}
/**
 * Runs inference with audio chunks using an ONNX model.
 * @param modelPath Path of the ONNX model file
 * @param audioFilePath Path of the audio file to process
 * @param chunkInMs Block size in milliseconds
 * @param overlapMs Overlap size in milliseconds
 */

async function runInferenceWithChunks(modelPath: string, audioFilePath: string, chunkInMs: number, overlapMs: number) {

    const buffer = fs.readFileSync(audioFilePath);
    const result = wav.decode(buffer);
    const noisy = result.channelData[0];
    const sr = result.sampleRate;

    console.log(`Audio file: ${audioFilePath}`);
    console.log(`Input length: ${noisy.length}`);

    // Inference session for model.
    const session = await InferenceSession.create(modelPath);

    if (session) {
      console.log("Model loaded successfully");
    } else {
      console.log("Failed to load model");
    }

    const batches = getAudioBlock(noisy, sr, chunkInMs, overlapMs);
    const outputLength = noisy.length;
    const outputAudio = new Float32Array(outputLength);
    let count = 0;
    const blockSize = Math.floor((sr / 1000) * chunkInMs);

    // input tensor
    const inputTensor = new Tensor("float32", new Float32Array(blockSize), [1, 1, blockSize]);
    const feeds = {
      input: inputTensor,
    };

    for (const batch of batches) {
      const noisyBatch = new Float32Array(batch.length);
      noisyBatch.set(batch);
      const noisyBatchFloat = convertFloatArray(noisyBatch);
      console.log("batch data shape: " + `[1, ${batch.length}]`);

      inputTensor.data.set(noisyBatchFloat);

      const option = createRunOptions();
      const outputMap = await session.run(feeds, option);
      console.log(`Inference is over for batch ${count}`);

      const output = outputMap.output.data as Float32Array;

      const offset = count * blockSize - count * Math.floor((sr / 1000) * overlapMs); // calculate the offset of the current block, taking overlap into account
      const end = Math.min(offset + output.length, outputAudio.length);
      outputAudio.set(output.slice(0, end - offset), offset);

      count += 1;
    }
    
    const outputBuffer = wav.encode([outputAudio], {
        sampleRate: sr,
        float: true,
      });
    const outputFile = audioFilePath.replace('.wav', '_output.wav');
    fs.writeFileSync(outputFile, outputBuffer);
    
    console.log(`Output file saved: ${outputFile}`);
}

  // get the command line arguments
const modelPath = 'data/dns64.onnx';
const audioFilePath = 'data/sample-small.wav';
const chunkSizeInMs =  300; // chunk size in ms
const overlapMs = 120; // overlap size in ms
runInferenceWithChunks(modelPath, audioFilePath, chunkSizeInMs, overlapMs);