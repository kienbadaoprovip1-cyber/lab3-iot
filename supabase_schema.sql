-- ========================================================
-- SUPABASE SCHEMA CHO LAB 3: FACE MASK DETECTION
-- ========================================================

-- 1. Tạo bảng lưu nhật ký nhận diện & vi phạm
CREATE TABLE IF NOT EXISTS detection_logs (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    device_id TEXT DEFAULT 'webcam',
    total_faces INT DEFAULT 0,
    mask_count INT DEFAULT 0,
    no_mask_count INT DEFAULT 0,
    incorrect_count INT DEFAULT 0,
    unknown_count INT DEFAULT 0,
    has_violation BOOLEAN DEFAULT FALSE,
    image_url TEXT,
    inference_ms REAL DEFAULT 0,
    server_ms REAL DEFAULT 0,
    execution_location TEXT DEFAULT 'cloud',
    detections_json TEXT
);

-- 2. Tạo Index để truy vấn nhanh
CREATE INDEX IF NOT EXISTS idx_detection_logs_time ON detection_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_logs_violation ON detection_logs (has_violation, created_at DESC);

-- 3. Cấp quyền truy cập cho Anon Key (Public / Client API)
ALTER TABLE detection_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Cho phép đọc dữ liệu công khai" ON detection_logs;
CREATE POLICY "Cho phép đọc dữ liệu công khai" 
ON detection_logs FOR SELECT 
USING (true);

DROP POLICY IF EXISTS "Cho phép API chèn dữ liệu" ON detection_logs;
CREATE POLICY "Cho phép API chèn dữ liệu" 
ON detection_logs FOR INSERT 
WITH CHECK (true);

DROP POLICY IF EXISTS "Cho phép xóa nhật ký" ON detection_logs;
CREATE POLICY "Cho phép xóa nhật ký" 
ON detection_logs FOR DELETE 
USING (true);

-- 4. Tạo Storage Bucket để lưu ảnh vi phạm (Public bucket)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('violations', 'violations', true) 
ON CONFLICT (id) DO NOTHING;

-- Cho phép upload và xem ảnh công khai trong bucket violations
DROP POLICY IF EXISTS "Public Access Storage" ON storage.objects;
CREATE POLICY "Public Access Storage" 
ON storage.objects FOR SELECT 
USING (bucket_id = 'violations');

DROP POLICY IF EXISTS "Allow Upload to Storage" ON storage.objects;
CREATE POLICY "Allow Upload to Storage" 
ON storage.objects FOR INSERT 
WITH CHECK (bucket_id = 'violations');
