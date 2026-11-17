import sharp from 'sharp';
import {createApp} from './server.mjs';
import {createModelSwitcher} from './model-switcher.mjs';
import {initDatabase} from './database.mjs';

sharp.cache({memory:16,files:0,items:16});
sharp.concurrency(1);

initDatabase();

const classifier=await createModelSwitcher();
const port=Number(process.env.PORT || 8080);
createApp(classifier).listen(port,'0.0.0.0',()=>console.log('Lab3 ready on port '+port));
