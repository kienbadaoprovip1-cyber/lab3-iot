import sharp from 'sharp';
import {createApp} from './server.mjs';
import {loadClassifier} from './inference.mjs';
sharp.cache({memory:16,files:0,items:16});
sharp.concurrency(1);
const classifier=await loadClassifier(process.env.MODEL_FOLDER || 'model_yolo');
const port=Number(process.env.PORT || 8080);
createApp(classifier).listen(port,'0.0.0.0',()=>console.log('Lab3 ready on port '+port));
