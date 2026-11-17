import express from 'express';
import { performance } from 'node:perf_hooks';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { loadClassifier } from './inference.mjs';
import { initDatabase, saveDetection, getViolations, getStats, clearLogs } from './database.mjs';

export function createApp(classifier, options = {}) {
  initDatabase(options.dbPath);
  const app = express();
  app.disable('x-powered-by');
  // Lab API has no cookies or credentials. CORS permits pages served from ESP32 LAN IPs.
  const origins = (process.env.ALLOWED_ORIGINS || '*').split(',').map(s => s.trim());
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origins.includes('*')) res.set('Access-Control-Allow-Origin', '*');
    else if (origins.includes(origin)) res.set('Access-Control-Allow-Origin', origin).vary('Origin');
    res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.set('Access-Control-Allow-Headers', 'Content-Type, X-Device-ID');
    res.set('Cache-Control', 'no-store');
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
  });
  app.use(express.static(fileURLToPath(new URL('./public/', import.meta.url))));
  app.get('/health', (_req, res) => res.json({ ready: true, labels: classifier.labels,
    model_id: classifier.modelId, backend: classifier.backend, task: classifier.task || 'image-classification', architecture: classifier.architecture || 'Teachable Machine legacy', execution_location: process.env.EXECUTION_LOCATION || 'local' }));
  
  // Database API endpoints
  app.get('/api/violations', async (req, res) => {
    const limit = Math.min(100, Math.max(1, Number(req.query.limit) || 30));
    const offset = Math.max(0, Number(req.query.offset) || 0);
    const violationOnly = req.query.all !== 'true';
    res.json(await getViolations({ limit, offset, violationOnly }));
  });

  app.get('/api/stats', async (_req, res) => {
    const stats = await getStats();
    res.json(stats || { total_scans: 0, total_faces: 0, total_violations: 0, compliance_rate: 100 });
  });

  app.post('/api/violations/clear', async (_req, res) => {
    const ok = await clearLogs();
    res.json({ success: ok });
  });

  let busy = false;
  app.post('/predict', (req, res, next) => {
    if (!req.is('image/jpeg')) return res.status(415).json({ error: 'Send raw JPEG with Content-Type: image/jpeg' });
    req.receivedAt = performance.now(); next();
  }, express.raw({ type: 'image/jpeg', limit: '300kb' }), async (req, res) => {
    if (!req.body?.length) return res.status(400).json({ error: 'Empty image' });
    if (busy) return res.status(429).json({ error: 'Inference busy; retry later' });
    busy = true;
    try {
      const result = await classifier.predict(req.body);
      const serverMs = performance.now() - req.receivedAt;
      const requestId = randomUUID();
      const executionLocation = process.env.EXECUTION_LOCATION || 'local';

      saveDetection({
        id: requestId,
        counts: result.counts || {},
        inference_ms: result.inference_ms || 0,
        server_ms: serverMs,
        execution_location: executionLocation,
        detections: result.detections || [],
        imageBuffer: req.body,
        device_id: (req.headers['x-device-id'] || 'webcam').toString()
      });

      res.json({ request_id: requestId, ...result, server_ms: serverMs,
        execution_location: executionLocation });
    } catch (error) {
      console.error('Prediction failed:', error.message);
      res.status(422).json({ error: 'Cannot decode or infer this JPEG' });
    } finally { busy = false; }
  });
  app.use((err, _req, res, _next) => res.status(err.status || 500).json({ error: err.status === 413 ? 'Image exceeds 300 KB' : 'Request failed' }));
  return app;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const classifier = await loadClassifier(process.env.MODEL_FOLDER || undefined);
  const port = Number(process.env.PORT || 8080);
  createApp(classifier).listen(port, '0.0.0.0', () => console.log(`Lab3 API ready on :${port}; labels=${classifier.labels.join(', ')}`));
}
