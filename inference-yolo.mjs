import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { suppress, sceneSummary } from './inference-multiface.mjs';

export async function loadClassifier(folder='model_yolo') {
  const bytes=await readFile(folder+'/model.onnx'),metadataBytes=await readFile(folder+'/metadata.json');
  const metadata=JSON.parse(metadataBytes),labels=metadata.labels,size=metadata.image_size;
  const confidence=Number(metadata.confidence),nms=Number(metadata.nms_iou);
  if(labels.length!==3||!(confidence>0&&confidence<1&&size>=32))throw new Error('Invalid YOLO metadata');
  const modelId=createHash('sha256').update(bytes).update(metadataBytes).digest('hex');
  const threads=Number(process.env.ORT_NUM_THREADS || 2);
  if(!Number.isInteger(threads)||threads<1||threads>16)throw new Error('ORT_NUM_THREADS must be an integer from 1 to 16');
  const session=await ort.InferenceSession.create(bytes,{executionProviders:['cpu'],intraOpNumThreads:threads});
  const inputName=session.inputNames[0],outputName=session.outputNames[0],plane=size*size;
  const input=new ort.Tensor('float32',new Float32Array(3*plane),[1,3,size,size]);
  const warm=await session.run({[inputName]:input});input.dispose();for(const value of Object.values(warm))value.dispose();
  return {
    labels,modelId,backend:'onnxruntime-cpu',architecture:metadata.architecture,task:'multi-face-mask-detection',
    async predict(jpeg) {
      const start=performance.now(),image=sharp(jpeg,{limitInputPixels:16000000,failOn:'error'});
      if((await image.metadata()).format!=='jpeg')throw new Error('Only JPEG is supported');
      const source=await image.autoOrient().removeAlpha().toColourspace('srgb').raw().toBuffer({resolveWithObject:true});
      const width=source.info.width,height=source.info.height,gain=Math.min(size/width,size/height);
      const rw=Math.max(1,Math.round(width*gain)),rh=Math.max(1,Math.round(height*gain));
      const left=Math.round((size-rw)/2-.1),top=Math.round((size-rh)/2-.1);
      const pixels=await sharp(source.data,{raw:source.info}).resize(rw,rh,{fit:'fill',kernel:'linear'})
        .extend({top,left,right:size-rw-left,bottom:size-rh-top,background:{r:114,g:114,b:114}}).raw().toBuffer();
      const values=new Float32Array(3*plane);
      for(let i=0;i<plane;i++)for(let c=0;c<3;c++)values[c*plane+i]=pixels[3*i+c]/255;
      const input=new ort.Tensor('float32',values,[1,3,size,size]),preprocessMs=performance.now()-start;
      let outputs,inferenceMs;const candidates=[];
      try {
        const before=performance.now();outputs=await session.run({[inputName]:input});inferenceMs=performance.now()-before;
        const output=outputs[outputName],data=output.data;
        if(output.dims.length!==3||output.dims[1]!==4+labels.length)throw new Error('Unexpected YOLO ONNX output layout');
        const count=output.dims[2];
        for(let i=0;i<count;i++){
          const scores=labels.map((_,c)=>data[(4+c)*count+i]),score=Math.max(...scores);
          if(!Number.isFinite(score)||score<confidence)continue;
          const cx=data[i],cy=data[count+i],w=data[2*count+i],h=data[3*count+i];
          if(![cx,cy,w,h].every(Number.isFinite))continue;
          const x1=Math.max(0,Math.min(width,(cx-w/2-left)/gain)),y1=Math.max(0,Math.min(height,(cy-h/2-top)/gain));
          const x2=Math.max(0,Math.min(width,(cx+w/2-left)/gain)),y2=Math.max(0,Math.min(height,(cy+h/2-top)/gain));
          if(x2-x1<2||y2-y1<2)continue;
          candidates.push({x:x1,y:y1,width:x2-x1,height:y2-y1,score,
            label:labels[scores.indexOf(score)],predictions:labels.map((className,c)=>({className,probability:scores[c]})).sort((a,b)=>b.probability-a.probability)});
        }
      }finally{input.dispose();if(outputs)for(const value of Object.values(outputs))value.dispose();}
      // Class-agnostic NMS: a physical face should not get competing class boxes.
      const boxes=suppress(candidates,nms),detections=boxes.slice(0,100).sort((a,b)=>a.x-b.x||a.y-b.y).map((d,i)=>({
        id:i+1,box:{x:d.x,y:d.y,width:d.width,height:d.height},label:d.label,predicted_label:d.label,
        probability:d.score,face_score:d.score,predictions:d.predictions,reliable:true}));
      return {...sceneSummary(detections),detections,image:{width,height},face_count:detections.length,truncated:boxes.length>100,
        face_score_threshold:confidence,class_confidence_threshold:confidence,score_semantics:'joint detection/class confidence, sigmoid per class; not a calibrated face probability',
        inference_ms:inferenceMs,detector_ms:inferenceMs,classification_ms:0,preprocess_ms:preprocessMs,
        pipeline_ms:performance.now()-start,model_id:modelId,architecture:metadata.architecture,task:'multi-face-mask-detection'};
    },
    async dispose(){await session.release();}
  };
}
