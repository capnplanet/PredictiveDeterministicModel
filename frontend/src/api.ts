export interface IngestionReport {
  total_rows: number;
  success_rows: number;
  failed_rows: number;
  errors: string[];
}

export interface DemoPreloadResponse {
  profile: string;
  output_dir: string;
  entities: IngestionReport;
  events: IngestionReport;
  interactions: IngestionReport;
  artifacts_manifest: IngestionReport;
  single_artifact: { artifact_id: string; sha256: string; artifact_type: string };
  features: { updated_artifacts: number };
  training?: { run_id: string; metrics: Record<string, number> };
  sample_entity_ids?: string[];
}

export interface ArtifactUploadResponse {
  artifact_id: string;
  sha256: string;
  artifact_type: string;
}

export interface SingleArtifactUploadRequest {
  file: File;
  artifactType: string;
  entityId?: string;
  timestamp?: string;
  metadata?: string;
}

export async function uploadCsv(path: string, file: File): Promise<IngestionReport> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(path, { method: 'POST', body: form });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  return (await res.json()) as IngestionReport;
}

export async function uploadArtifactsManifest(file: File): Promise<IngestionReport> {
  return uploadCsv('/ingest/artifacts', file);
}

export async function preloadDemoData(
  profile: 'small' | 'medium',
  resetExisting = true,
  trainModel = true,
): Promise<DemoPreloadResponse> {
  const res = await fetch('/demo/preload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile, reset_existing: resetExisting, extract_features: true, train_model: trainModel }),
  });
  if (!res.ok) {
    throw new Error(`Demo preload failed: ${res.status}`);
  }
  return (await res.json()) as DemoPreloadResponse;
}

export async function uploadSingleArtifact(payload: SingleArtifactUploadRequest): Promise<ArtifactUploadResponse> {
  const form = new FormData();
  form.append('file', payload.file);
  form.append('artifact_type', payload.artifactType);
  if (payload.entityId) form.append('entity_id', payload.entityId);
  if (payload.timestamp) form.append('timestamp', payload.timestamp);
  if (payload.metadata) form.append('metadata', payload.metadata);

  const res = await fetch('/ingest/artifact', { method: 'POST', body: form });
  if (!res.ok) {
    throw new Error(`Artifact upload failed: ${res.status}`);
  }
  return (await res.json()) as ArtifactUploadResponse;
}

export interface TrainResponse {
  run_id: string;
  metrics: Record<string, number>;
}

export async function triggerTrain(): Promise<TrainResponse> {
  const res = await fetch('/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
  if (!res.ok) {
    throw new Error(`Train failed: ${res.status}`);
  }
  return (await res.json()) as TrainResponse;
}

export interface RunInfo {
  run_id: string;
  created_at: string;
  metrics: Record<string, number>;
}

export async function listRuns(): Promise<RunInfo[]> {
  const res = await fetch('/runs');
  if (!res.ok) {
    throw new Error(`List runs failed: ${res.status}`);
  }
  return (await res.json()) as RunInfo[];
}

export interface EntityPrediction {
  entity_id: string;
  regression: number;
  probability: number;
  ranking_score: number;
  narrative?: string;
  narrative_template?: string;
  narrative_long?: string;
  narrative_source?: string;
}

export interface PredictResponse {
  run_id: string;
  predictions: EntityPrediction[];
}

export interface QueryRequest {
  query: string;
  run_id?: string;
  limit?: number;
}

export interface QueryResult {
  entity_id: string;
  regression: number;
  probability: number;
  ranking_score: number;
  narrative?: string;
}

export interface QueryResponse {
  run_id: string;
  query: string;
  interpreted_as: string;
  llm_used: boolean;
  results: QueryResult[];
}

export async function predict(entityIds: string[], narrativeMode: 'template' | 'llm' | 'both' = 'both'): Promise<PredictResponse> {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entity_ids: entityIds, explanations: true, narrative_mode: narrativeMode }),
  });
  if (!res.ok) {
    throw new Error(`Predict failed: ${res.status}`);
  }
  return (await res.json()) as PredictResponse;
}

export async function queryPredictions(payload: QueryRequest): Promise<QueryResponse> {
  const res = await fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') {
        detail = ` ${body.detail}`;
      }
    } catch {
      detail = '';
    }
    throw new Error(`Query failed: ${res.status}.${detail}`.trim());
  }
  return (await res.json()) as QueryResponse;
}
