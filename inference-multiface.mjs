import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { loadClassifier as loadMaskClassifier } from './inference-onnx.mjs';

export function iou(a, b) {
  const overlap = Math.max(0, Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)) *
    Math.max(0, Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y));
  return overlap / (a.width*a.height+b.width*b.height-overlap || 1);
}
export function suppress(candidates, threshold = .3) {
  const keep = [];
  for (const box of [...candidates].sort((a,b) => b.score-a.score))
    if (!keep.some(other => iou(box,other)>threshold)) keep.push(box);
  return keep;
}
export function sceneSummary(detections) {
  const counts = { Mask: 0, 'No mask': 0, 'Incorrect mask': 0, Unknown: 0 };
  for (const d of detections) counts[d.label]++;
  const label = !detections.length ? 'No one' : ['No mask','Incorrect mask','Mask'].find(l => counts[l]) || 'Unknown';
  const selected = detections.filter(d => d.label === label);
  return { counts, label, probability: label === 'No one' ? 1 : Math.max(0, ...selected.map(d => d.probability)) };
}
export async function loadClassifier() {
  const classifier = await loadMaskClassifier('model_multiface_mask');
  const bytes = await readFile('model_yunet/face_detection_yunet_2023mar.onnx');
  const faceThreshold = Number(process.env.FACE_SCORE_THRESHOLD || .6);
  const classThreshold = Number(process.env.CLASS_CONFIDENCE_THRESHOLD || .85);
  if (!(faceThreshold > 0 && faceThreshold <= 1 && classThreshold > 0 && classThreshold <= 1)) throw new Error('Invalid confidence threshold');
  const modelId = createHash('sha256').update(bytes).update(classifier.modelId)
    .update(JSON.stringify({ faceThreshold, classThreshold, nms: .3, input: 640, maxFaces: 100, squareCropScale: 1.2 })).digest('hex');
  const detector = await ort.InferenceSession.create(bytes, { executionProviders: ['cpu'], intraOpNumThreads: 2 });
  const size = 640, plane = size*size;
  const warmInput = new ort.Tensor('float32', new Float32Array(3*plane), [1,3,size,size]);
  const warm = await detector.run({ input: warmInput });
  warmInput.dispose(); for (const value of Object.values(warm)) value.dispose();
  return {
    labels: classifier.labels, modelId, backend: 'onnxruntime-cpu', task: 'multi-face-mask-detection',
    architecture: 'YuNet + MobileNetV3-Small',
    async predict(jpeg) {
      const start = performance.now();
      const image = sharp(jpeg, { limitInputPixels: 16000000, failOn: 'error' });
      if ((await image.metadata()).format !== 'jpeg') throw new Error('Only JPEG is supported');
      const source = await image.autoOrient().removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
      const width = source.info.width, height = source.info.height;
      const scale = size / Math.max(width,height), resizedWidth = Math.max(1,Math.round(width*scale)), resizedHeight = Math.max(1,Math.round(height*scale));
      const pixels = await sharp(source.data, { raw: source.info }).resize(resizedWidth,resizedHeight,{fit:'fill',kernel:'linear'})
        .extend({top:0,left:0,right:size-resizedWidth,bottom:size-resizedHeight,background:{r:0,g:0,b:0}}).raw().toBuffer();
      // YuNet input is BGR float32, raw 0..255; no ImageNet normalization.
      const values = new Float32Array(3*plane);
      for (let i=0;i<plane;i++) for(let c=0;c<3;c++) values[c*plane+i] = pixels[3*i+2-c];
      const input = new ort.Tensor('float32',values,[1,3,size,size]);
      let preprocessMs = performance.now()-start, detectorMs, outputs;
      const candidates = [];
      try {
        const before = performance.now(); outputs = await detector.run({input}); detectorMs = performance.now()-before;
        for (const stride of [8,16,32]) {
          const cols=size/stride, cls=outputs['cls_'+stride].data, obj=outputs['obj_'+stride].data, boxes=outputs['bbox_'+stride].data;
          for (let i=0;i<cls.length;i++) {
            const score=Math.sqrt(Math.max(0,Math.min(1,cls[i]))*Math.max(0,Math.min(1,obj[i])));
            if(score<faceThreshold) continue;
            const cx=(i%cols+boxes[4*i])*stride, cy=(Math.floor(i/cols)+boxes[4*i+1])*stride;
            const w=Math.exp(boxes[4*i+2])*stride,h=Math.exp(boxes[4*i+3])*stride;
            if (![cx,cy,w,h].every(Number.isFinite) || cx>=resizedWidth || cy>=resizedHeight) continue;
            const x1=Math.max(0,(cx-w/2)*width/resizedWidth), y1=Math.max(0,(cy-h/2)*height/resizedHeight);
            const x2=Math.min(width,(cx+w/2)*width/resizedWidth), y2=Math.min(height,(cy+h/2)*height/resizedHeight);
            if(x2-x1>=3 && y2-y1>=3) candidates.push({x:x1,y:y1,width:x2-x1,height:y2-y1,score});
          }
        }
      } finally { input.dispose(); if(outputs) for(const value of Object.values(outputs)) value.dispose(); }
      const faces=suppress(candidates), detections=[];
      let classificationMs=0;
      for(const face of faces.slice(0,100).sort((a,b)=>a.x-b.x||a.y-b.y)) {
        const before=performance.now();
        // Square crop retains nose and chin; a center crop of a tall tight box would discard them.
        const side=Math.min(width,height,Math.ceil(Math.max(face.width,face.height)*1.2));
        const left=Math.max(0,Math.min(width-side,Math.floor(face.x+face.width/2-side/2)));
        const top=Math.max(0,Math.min(height-side,Math.floor(face.y+face.height/2-side/2)));
        const crop=await sharp(source.data,{raw:source.info}).extract({left,top,width:side,height:side}).jpeg({quality:95}).toBuffer();
        preprocessMs+=performance.now()-before;
        const result=await classifier.predict(crop);
        classificationMs+=result.inference_ms;preprocessMs+=result.preprocess_ms;
        const reliable=result.probability>=classThreshold;
        detections.push({id:detections.length+1,box:{x:face.x,y:face.y,width:face.width,height:face.height},
          face_score:face.score,label:reliable?result.label:'Unknown',predicted_label:result.label,
          probability:result.probability,predictions:result.predictions,reliable});
      }
      return {...sceneSummary(detections),detections,image:{width,height},face_count:detections.length,
        truncated:faces.length>100,face_score_threshold:faceThreshold,class_confidence_threshold:classThreshold,
        detector_ms:detectorMs,classification_ms:classificationMs,inference_ms:detectorMs+classificationMs,
        preprocess_ms:preprocessMs,pipeline_ms:performance.now()-start,model_id:modelId,
        architecture:'YuNet + MobileNetV3-Small',task:'multi-face-mask-detection'};
    },
    async dispose() { await detector.release(); await classifier.dispose(); }
  };
}
