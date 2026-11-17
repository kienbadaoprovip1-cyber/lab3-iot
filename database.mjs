import { DatabaseSync } from 'node:sqlite';
import { mkdirSync, writeFileSync, readdirSync, unlinkSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';

let db = null;
let violationsDir = null;

function getSupabaseConfig() {
  const url = (process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || '').replace(/\/+$/, '').replace(/\/rest\/v1\/?$/, '');
  const key = process.env.SUPABASE_KEY || process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';
  return { url, key, enabled: Boolean(url && key) };
}

export function initDatabase(dbPath = process.env.DB_PATH || 'data/lab3.db') {
  const sb = getSupabaseConfig();
  if (sb.enabled) {
    console.log(`[Database] Connected to Supabase Cloud: ${sb.url}`);
    return { type: 'supabase', url: sb.url };
  }

  try {
    const dataDir = resolve('data');
    violationsDir = resolve('public/violations');
    mkdirSync(dataDir, { recursive: true });
    mkdirSync(violationsDir, { recursive: true });

    db = new DatabaseSync(resolve(dbPath));
    db.exec(`
      CREATE TABLE IF NOT EXISTS detection_logs (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        device_id TEXT DEFAULT 'webcam',
        total_faces INTEGER DEFAULT 0,
        mask_count INTEGER DEFAULT 0,
        no_mask_count INTEGER DEFAULT 0,
        incorrect_count INTEGER DEFAULT 0,
        unknown_count INTEGER DEFAULT 0,
        has_violation INTEGER DEFAULT 0,
        image_url TEXT,
        inference_ms REAL DEFAULT 0,
        server_ms REAL DEFAULT 0,
        execution_location TEXT DEFAULT 'local',
        detections_json TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_detection_logs_time ON detection_logs (created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_detection_logs_violation ON detection_logs (has_violation, created_at DESC);
    `);
    console.log('[Database] Connected to Local SQLite: ' + dbPath);
    return db;
  } catch (err) {
    console.error('Failed to initialize SQLite database:', err.message);
    return null;
  }
}

async function uploadToSupabaseStorage(id, imageBuffer) {
  const sb = getSupabaseConfig();
  if (!sb.enabled) return null;
  try {
    const uploadUrl = `${sb.url}/storage/v1/object/violations/${id}.jpg`;
    const res = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        'apikey': sb.key,
        'Authorization': `Bearer ${sb.key}`,
        'Content-Type': 'image/jpeg',
        'x-upsert': 'true'
      },
      body: imageBuffer
    });
    if (res.ok) {
      return `${sb.url}/storage/v1/object/public/violations/${id}.jpg`;
    }
    console.error('Supabase storage upload error:', await res.text());
  } catch (e) {
    console.error('Supabase upload failed:', e.message);
  }
  return null;
}

export async function saveDetection({
  id,
  counts = {},
  inference_ms = 0,
  server_ms = 0,
  execution_location = 'local',
  detections = [],
  imageBuffer = null,
  device_id = 'webcam'
}) {
  const maskCount = counts['Mask'] || counts['mask'] || 0;
  const noMaskCount = counts['No mask'] || counts['no_mask'] || 0;
  const incorrectCount = counts['Incorrect mask'] || counts['incorrect'] || 0;
  const unknownCount = counts['Unknown'] || counts['unknown'] || 0;
  const totalFaces = maskCount + noMaskCount + incorrectCount + unknownCount;
  const hasViolation = (noMaskCount > 0 || incorrectCount > 0) ? 1 : 0;

  const sb = getSupabaseConfig();
  if (sb.enabled) {
    try {
      let imageUrl = null;
      if (hasViolation && imageBuffer && imageBuffer.length) {
        imageUrl = await uploadToSupabaseStorage(id, imageBuffer);
      }

      const row = {
        id,
        created_at: new Date().toISOString(),
        device_id,
        total_faces: totalFaces,
        mask_count: maskCount,
        no_mask_count: noMaskCount,
        incorrect_count: incorrectCount,
        unknown_count: unknownCount,
        has_violation: Boolean(hasViolation),
        image_url: imageUrl,
        inference_ms: Number(inference_ms.toFixed(2)),
        server_ms: Number(server_ms.toFixed(2)),
        execution_location,
        detections_json: JSON.stringify(detections)
      };

      const resp = await fetch(`${sb.url}/rest/v1/detection_logs`, {
        method: 'POST',
        headers: {
          'apikey': sb.key,
          'Authorization': `Bearer ${sb.key}`,
          'Content-Type': 'application/json',
          'Prefer': 'return=minimal'
        },
        body: JSON.stringify(row)
      });
      if (!resp.ok) {
        console.error('Supabase insert failed:', resp.status, await resp.text());
      } else {
        console.log(`[Supabase] Recorded log ${id} (violation: ${hasViolation})`);
      }

      return { id, hasViolation, imageUrl };
    } catch (err) {
      console.error('Supabase save error:', err.message);
      return null;
    }
  }

  // Fallback to SQLite
  if (!db) return null;
  try {
    let imageUrl = null;
    if (hasViolation && imageBuffer && imageBuffer.length) {
      const fileName = `${id}.jpg`;
      const filePath = join(violationsDir, fileName);
      writeFileSync(filePath, imageBuffer);
      imageUrl = `/violations/${fileName}`;
    }

    const insert = db.prepare(`
      INSERT INTO detection_logs (
        id, created_at, device_id, total_faces, mask_count, no_mask_count,
        incorrect_count, unknown_count, has_violation, image_url,
        inference_ms, server_ms, execution_location, detections_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    insert.run(
      id,
      new Date().toISOString(),
      device_id,
      totalFaces,
      maskCount,
      noMaskCount,
      incorrectCount,
      unknownCount,
      hasViolation,
      imageUrl,
      Number(inference_ms.toFixed(2)),
      Number(server_ms.toFixed(2)),
      execution_location,
      JSON.stringify(detections)
    );

    return { id, hasViolation, imageUrl };
  } catch (err) {
    console.error('Error saving detection to SQLite:', err.message);
    return null;
  }
}

export async function getViolations({ limit = 30, offset = 0, violationOnly = true } = {}) {
  const sb = getSupabaseConfig();
  if (sb.enabled) {
    try {
      let url = `${sb.url}/rest/v1/detection_logs?select=*&order=created_at.desc&limit=${limit}&offset=${offset}`;
      if (violationOnly) url += '&has_violation=eq.true';

      const res = await fetch(url, {
        headers: {
          'apikey': sb.key,
          'Authorization': `Bearer ${sb.key}`,
          'Range-Unit': 'items',
          'Prefer': 'count=exact'
        }
      });
      const items = await res.json();
      const contentRange = res.headers.get('content-range');
      const total = contentRange ? Number(contentRange.split('/')[1]) || items.length : items.length;

      return {
        total,
        items: Array.isArray(items) ? items.map(r => ({
          ...r,
          has_violation: Boolean(r.has_violation),
          detections: r.detections_json ? JSON.parse(r.detections_json) : []
        })) : []
      };
    } catch (err) {
      console.error('Supabase fetch error:', err.message);
      return { total: 0, items: [] };
    }
  }

  if (!db) return { total: 0, items: [] };
  try {
    const countStmt = db.prepare(
      violationOnly
        ? 'SELECT COUNT(*) as count FROM detection_logs WHERE has_violation = 1'
        : 'SELECT COUNT(*) as count FROM detection_logs'
    );
    const total = Number(countStmt.get().count || 0);

    const queryStmt = db.prepare(
      violationOnly
        ? `SELECT * FROM detection_logs WHERE has_violation = 1 ORDER BY created_at DESC LIMIT ? OFFSET ?`
        : `SELECT * FROM detection_logs ORDER BY created_at DESC LIMIT ? OFFSET ?`
    );
    const rows = queryStmt.all(limit, offset);
    const items = rows.map(r => ({
      ...r,
      has_violation: Boolean(r.has_violation),
      detections: r.detections_json ? JSON.parse(r.detections_json) : []
    }));

    return { total, items };
  } catch (err) {
    console.error('Error fetching violations:', err.message);
    return { total: 0, items: [] };
  }
}

export async function getStats() {
  const sb = getSupabaseConfig();
  if (sb.enabled) {
    try {
      const res = await fetch(`${sb.url}/rest/v1/detection_logs?select=total_faces,mask_count,no_mask_count,incorrect_count,has_violation,inference_ms,server_ms`, {
        headers: {
          'apikey': sb.key,
          'Authorization': `Bearer ${sb.key}`
        }
      });
      const rows = await res.json();
      if (!Array.isArray(rows)) return null;

      const totalScans = rows.length;
      let totalFaces = 0, totalViolations = 0, totalMask = 0, totalNoMask = 0, totalIncorrect = 0;
      let sumInference = 0, sumServer = 0;

      for (const r of rows) {
        totalFaces += (r.total_faces || 0);
        if (r.has_violation) totalViolations++;
        totalMask += (r.mask_count || 0);
        totalNoMask += (r.no_mask_count || 0);
        totalIncorrect += (r.incorrect_count || 0);
        sumInference += (r.inference_ms || 0);
        sumServer += (r.server_ms || 0);
      }

      const complianceRate = totalFaces > 0 ? Number(((totalMask / totalFaces) * 100).toFixed(1)) : 100;
      return {
        total_scans: totalScans,
        total_faces: totalFaces,
        total_violations: totalViolations,
        total_mask: totalMask,
        total_no_mask: totalNoMask,
        total_incorrect: totalIncorrect,
        compliance_rate: complianceRate,
        avg_inference_ms: totalScans ? Number((sumInference / totalScans).toFixed(1)) : 0,
        avg_server_ms: totalScans ? Number((sumServer / totalScans).toFixed(1)) : 0
      };
    } catch (e) {
      console.error('Supabase stats error:', e.message);
      return null;
    }
  }

  if (!db) return null;
  try {
    const stmt = db.prepare(`
      SELECT 
        COUNT(*) as total_scans,
        COALESCE(SUM(total_faces), 0) as total_faces,
        COALESCE(SUM(has_violation), 0) as total_violations,
        COALESCE(SUM(mask_count), 0) as total_mask,
        COALESCE(SUM(no_mask_count), 0) as total_no_mask,
        COALESCE(SUM(incorrect_count), 0) as total_incorrect,
        COALESCE(AVG(inference_ms), 0) as avg_inference_ms,
        COALESCE(AVG(server_ms), 0) as avg_server_ms
      FROM detection_logs
    `);
    const row = stmt.get();
    const totalFaces = Number(row.total_faces);
    const totalMask = Number(row.total_mask);
    const complianceRate = totalFaces > 0 ? Number(((totalMask / totalFaces) * 100).toFixed(1)) : 100;

    return {
      total_scans: Number(row.total_scans),
      total_faces: totalFaces,
      total_violations: Number(row.total_violations),
      total_mask: totalMask,
      total_no_mask: Number(row.total_no_mask),
      total_incorrect: Number(row.total_incorrect),
      compliance_rate: complianceRate,
      avg_inference_ms: Number(Number(row.avg_inference_ms).toFixed(1)),
      avg_server_ms: Number(Number(row.avg_server_ms).toFixed(1))
    };
  } catch (err) {
    console.error('Error calculating stats:', err.message);
    return null;
  }
}

export async function clearLogs() {
  const sb = getSupabaseConfig();
  if (sb.enabled) {
    try {
      await fetch(`${sb.url}/rest/v1/detection_logs`, {
        method: 'DELETE',
        headers: {
          'apikey': sb.key,
          'Authorization': `Bearer ${sb.key}`,
          'Prefer': 'return=minimal'
        }
      });
      return true;
    } catch (e) {
      return false;
    }
  }

  if (!db) return false;
  try {
    db.exec('DELETE FROM detection_logs;');
    if (violationsDir && existsSync(violationsDir)) {
      const files = readdirSync(violationsDir);
      for (const file of files) {
        if (file.endsWith('.jpg') || file.endsWith('.jpeg')) {
          unlinkSync(join(violationsDir, file));
        }
      }
    }
    return true;
  } catch (err) {
    console.error('Error clearing logs:', err.message);
    return false;
  }
}
