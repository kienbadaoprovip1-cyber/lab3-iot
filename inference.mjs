// Default detects and classifies each face. Classification-only backends are explicit.
export async function loadClassifier(folder) {
  const backend = process.env.MODEL_BACKEND || 'yolo';
  if (!['yolo', 'multiface', 'mobilenetv3', 'teachable'].includes(backend)) throw new Error('Unknown MODEL_BACKEND: ' + backend);
  const module = await import(backend === 'yolo' ? './inference-yolo.mjs' : backend === 'multiface' ? './inference-multiface.mjs' : backend === 'teachable' ? './inference-tm.mjs' : './inference-onnx.mjs');
  return module.loadClassifier(folder);
}
