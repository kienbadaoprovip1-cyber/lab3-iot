import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';

const roundEven = value => Number.isInteger(value - 0.5) ? 2 * Math.round(value / 2) : Math.round(value);
export async function loadClassifier(folder = resolve('model_mobilenetv3')) {
  const metadataBytes = await readFile(resolve(folder, 'metadata.json'));
  const metadata = JSON.parse(metadataBytes);
  const bytes = await readFile(resolve(folder, 'model.onnx'));
  const modelId = createHash('sha256').update(bytes).update(metadataBytes).digest('hex');
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['cpu'], intraOpNumThreads: 2 });
  const { mean, std, crop: size, resize_shorter: shorter } = metadata.input;
  const inputName = session.inputNames[0], outputName = session.outputNames[0];
  const warmInput = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]);
  const warm = await session.run({ [inputName]: warmInput });
  warmInput.dispose(); for (const value of Object.values(warm)) value.dispose();
  return {
    labels: metadata.labels, modelId, backend: 'onnxruntime-cpu', architecture: metadata.architecture,
    async predict(jpeg) {
      const start = performance.now();
      const image = sharp(jpeg, { limitInputPixels: 16000000, failOn: 'error' });
      const meta = await image.metadata();
      if (meta.format !== 'jpeg') throw new Error('Only JPEG is supported');
      // Auto-orient first, matching ImageOps.exif_transpose in training.
      const oriented = await image.autoOrient().removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
      const scale = shorter / Math.min(oriented.info.width, oriented.info.height);
      const width = Math.floor(oriented.info.width * scale), height = Math.floor(oriented.info.height * scale);
      const rgb = await sharp(oriented.data, { raw: oriented.info }).resize(width, height, { fit: 'fill', kernel: 'linear' })
        .extract({ left: roundEven((width - size) / 2), top: roundEven((height - size) / 2), width: size, height: size }).raw().toBuffer();
      const plane = size * size, values = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++)
        values[c * plane + i] = (rgb[i * 3 + c] / 255 - mean[c]) / std[c];
      const input = new ort.Tensor('float32', values, [1, 3, size, size]);
      const preprocessMs = performance.now() - start;
      let outputs;
      try {
        const before = performance.now();
        outputs = await session.run({ [inputName]: input });
        const scores = Array.from(outputs[outputName].data);
        const inferenceMs = performance.now() - before;
        if (scores.length !== metadata.labels.length || scores.some(p => !Number.isFinite(p))) throw new Error('Invalid ONNX output');
        const predictions = metadata.labels.map((className, i) => ({ className, probability: scores[i] }))
          .sort((a,b) => b.probability - a.probability);
        return { label: predictions[0].className, probability: predictions[0].probability, predictions,
          preprocess_ms: preprocessMs, inference_ms: inferenceMs, model_id: modelId, architecture: metadata.architecture };
      } finally {
        input.dispose(); if (outputs) for (const value of Object.values(outputs)) value.dispose();
      }
    },
    async dispose() { await session.release(); }
  };
}
