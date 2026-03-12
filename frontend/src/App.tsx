import React, { useMemo, useState } from 'react';
import {
  uploadCsv,
  uploadArtifactsManifest,
  uploadSingleArtifact,
  preloadDemoData,
  triggerTrain,
  listRuns,
  predict,
  queryPredictions,
  RunInfo,
  EntityPrediction,
  QueryResponse,
} from './api';
import './App.css';

type Tab = 'dataset' | 'train' | 'runs' | 'predict' | 'query';
type StatusTone = 'neutral' | 'success' | 'warning' | 'error';

type IconName = 'upload' | 'train' | 'ledger' | 'predict' | 'shield' | 'pulse' | 'chip';

const Icon: React.FC<{ name: IconName; className?: string }> = ({ name, className }) => {
  const common = { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  if (name === 'upload') {
    return (
      <svg className={className} {...common}>
        <path d="M12 16V5" />
        <path d="m7 10 5-5 5 5" />
        <path d="M4 19h16" />
      </svg>
    );
  }
  if (name === 'train') {
    return (
      <svg className={className} {...common}>
        <rect x="4" y="4" width="16" height="16" rx="2" />
        <path d="M9 4v4" />
        <path d="M15 4v4" />
        <path d="M9 16v4" />
        <path d="M15 16v4" />
        <path d="M4 9h4" />
        <path d="M4 15h4" />
        <path d="M16 9h4" />
        <path d="M16 15h4" />
      </svg>
    );
  }
  if (name === 'ledger') {
    return (
      <svg className={className} {...common}>
        <path d="M6 3h9l3 3v15H6z" />
        <path d="M15 3v3h3" />
        <path d="M9 11h6" />
        <path d="M9 15h6" />
      </svg>
    );
  }
  if (name === 'predict') {
    return (
      <svg className={className} {...common}>
        <circle cx="11" cy="11" r="7" />
        <path d="M21 21l-4.3-4.3" />
        <path d="M11 8v6" />
        <path d="M8 11h6" />
      </svg>
    );
  }
  if (name === 'pulse') {
    return (
      <svg className={className} {...common}>
        <path d="M3 12h4l2-4 4 8 2-4h6" />
      </svg>
    );
  }
  if (name === 'chip') {
    return (
      <svg className={className} {...common}>
        <rect x="5" y="5" width="14" height="14" rx="2" />
        <path d="M9 9h6v6H9z" />
      </svg>
    );
  }
  return (
    <svg className={className} {...common}>
      <path d="M12 3 4 7v6c0 4.5 3.1 7.9 8 9 4.9-1.1 8-4.5 8-9V7l-8-4z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
};

const formatMetric = (value: number | undefined): string => {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
  return value.toFixed(3);
};

const QUERY_PROMPT_PRESETS: Array<{ label: string; prompt: string }> = [
  {
    label: 'Strongest Relationships',
    prompt:
      'Show entities with the strongest relationships and highest ranking scores. For each result, provide a detailed narrative grounded in observed interactions, event patterns, and artifact signals from the latest trained run.',
  },
  {
    label: 'Risk Investigation',
    prompt:
      'Identify top entities by relationship strength that also show elevated probability. Explain likely risk drivers using only available model outputs and ingested evidence, and describe confidence limits.',
  },
  {
    label: 'Operational Prioritization',
    prompt:
      'Rank entities for immediate operational follow-up based on relationship intensity and ranking score. Return comprehensive, data-grounded narratives that reference temporal events and interaction context.',
  },
  {
    label: 'Anomaly Storyline',
    prompt:
      'Find entities with unusual relationship structures and strong ranking scores. Generate complex but accurate narratives that summarize what changed, why it matters, and what corroborating signals exist in ingested data.',
  },
];

export const App: React.FC = () => {
  const [tab, setTab] = useState<Tab>('dataset');
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [predictions, setPredictions] = useState<EntityPrediction[]>([]);
  const [artifactType, setArtifactType] = useState('image');
  const [artifactEntityId, setArtifactEntityId] = useState('');
  const [artifactTimestamp, setArtifactTimestamp] = useState('');
  const [artifactMetadata, setArtifactMetadata] = useState('');
  const [artifactFile, setArtifactFile] = useState<File | null>(null);
  const [demoProfile, setDemoProfile] = useState<'small' | 'medium'>('small');
  const [queryText, setQueryText] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [status, setStatus] = useState<{ message: string; tone: StatusTone }>({
    message: 'Operational console ready.',
    tone: 'neutral',
  });

  const setStatusMessage = (message: string, tone: StatusTone = 'neutral') => {
    setStatus({ message, tone });
  };

  const handleUpload = async (kind: 'entities' | 'events' | 'interactions', file: File | null) => {
    if (!file) return;
    setStatusMessage(`Uploading ${kind} dataset...`, 'warning');
    try {
      const path = `/ingest/${kind}`;
      const res = await uploadCsv(path, file);
      setStatusMessage(`Upload complete: ${kind} (${res.success_rows}/${res.total_rows} rows).`, 'success');
    } catch (error) {
      setStatusMessage(`Upload failed for ${kind}. ${(error as Error).message}`, 'error');
    }
  };

  const handleTrain = async () => {
    setStatusMessage('Model training operation started...', 'warning');
    try {
      const res = await triggerTrain();
      setStatusMessage(`Training complete. Run ID: ${res.run_id}`, 'success');
    } catch (error) {
      setStatusMessage(`Training failed. ${(error as Error).message}`, 'error');
    }
  };

  const handlePreloadDemo = async () => {
    setStatusMessage(`Preloading ${demoProfile} synthetic demo dataset and training a starter model...`, 'warning');
    try {
      const res = await preloadDemoData(demoProfile, true, true);
      const trainingSuffix = res.training?.run_id ? ` trained run ${res.training.run_id}.` : '';
      const sampleHint =
        res.sample_entity_ids && res.sample_entity_ids.length > 0
          ? ` Try inference IDs: ${res.sample_entity_ids.join(', ')}.`
          : '';
      setStatusMessage(
        `Demo preload complete: entities ${res.entities.success_rows}/${res.entities.total_rows}, events ${res.events.success_rows}/${res.events.total_rows}, interactions ${res.interactions.success_rows}/${res.interactions.total_rows}, artifacts ${res.artifacts_manifest.success_rows}/${res.artifacts_manifest.total_rows}, features updated ${res.features.updated_artifacts}.${trainingSuffix}${sampleHint}`,
        'success',
      );
    } catch (error) {
      setStatusMessage(`Demo preload failed. ${(error as Error).message}`, 'error');
    }
  };

  const handleArtifactsManifestUpload = async (file: File | null) => {
    if (!file) return;
    setStatusMessage('Uploading artifacts manifest...', 'warning');
    try {
      const res = await uploadArtifactsManifest(file);
      setStatusMessage(`Artifact manifest upload complete (${res.success_rows}/${res.total_rows} rows).`, 'success');
    } catch (error) {
      setStatusMessage(`Artifact manifest upload failed. ${(error as Error).message}`, 'error');
    }
  };

  const handleSingleArtifactUpload = async () => {
    if (!artifactFile) {
      setStatusMessage('Select an artifact file before upload.', 'warning');
      return;
    }

    const metadata = artifactMetadata.trim();
    if (metadata) {
      try {
        JSON.parse(metadata);
      } catch {
        setStatusMessage('Artifact metadata must be valid JSON.', 'warning');
        return;
      }
    }

    setStatusMessage('Uploading single artifact...', 'warning');
    try {
      const res = await uploadSingleArtifact({
        file: artifactFile,
        artifactType,
        entityId: artifactEntityId.trim() || undefined,
        timestamp: artifactTimestamp.trim() || undefined,
        metadata: metadata || undefined,
      });
      setStatusMessage(`Single artifact upload complete: ${res.sha256.slice(0, 12)}...`, 'success');
    } catch (error) {
      setStatusMessage(`Single artifact upload failed. ${(error as Error).message}`, 'error');
    }
  };

  const handleRefreshRuns = async () => {
    setStatusMessage('Syncing run ledger...', 'warning');
    try {
      const data = await listRuns();
      setRuns(data);
      setStatusMessage(`Run ledger refreshed. ${data.length} records available.`, 'success');
    } catch (error) {
      setStatusMessage(`Run refresh failed. ${(error as Error).message}`, 'error');
    }
  };

  const handlePredict = async (entityIds: string) => {
    const ids = entityIds.split(',').map((s) => s.trim()).filter(Boolean);
    if (ids.length === 0) {
      setStatusMessage('Enter at least one entity ID before prediction.', 'warning');
      return;
    }
    setStatusMessage(`Generating predictions for ${ids.length} entities...`, 'warning');
    try {
      const res = await predict(ids, 'both');
      setPredictions(res.predictions);
      setStatusMessage(`Prediction complete from run ${res.run_id}.`, 'success');
    } catch (error) {
      setStatusMessage(`Prediction failed. ${(error as Error).message}`, 'error');
    }
  };

  const handleQuery = async () => {
    const query = queryText.trim();
    if (!query) {
      setStatusMessage('Enter a natural language query before searching.', 'warning');
      return;
    }
    setStatusMessage('Running natural language query...', 'warning');
    try {
      const res = await queryPredictions({ query, limit: 5 });
      setQueryResult(res);
      setStatusMessage(`Query complete. ${res.results.length} results returned.`, 'success');
    } catch (error) {
      setStatusMessage(`Query failed. ${(error as Error).message}`, 'error');
    }
  };

  const handleUseQueryPrompt = (prompt: string) => {
    setQueryText(prompt);
    setStatusMessage('Example query loaded. Review and run when ready.', 'neutral');
  };

  const activeTitle: Record<Tab, string> = {
    dataset: 'Data Intake Zone',
    train: 'Model Operation',
    runs: 'Run Ledger',
    predict: 'Inference Console',
    query: 'Natural Language Query',
  };

  const latestRun = useMemo(() => {
    if (runs.length === 0) return null;
    return [...runs].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))[0];
  }, [runs]);

  const runCards = useMemo(() => {
    if (!latestRun) {
      return [
        { label: 'Latest Run', value: 'n/a', icon: 'ledger' as IconName },
        { label: 'Run Count', value: String(runs.length), icon: 'pulse' as IconName },
        { label: 'Model Health', value: 'n/a', icon: 'shield' as IconName },
      ];
    }
    const regR2 = latestRun.metrics?.reg_r2;
    const clsF1 = latestRun.metrics?.cls_f1;
    const ndcg = latestRun.metrics?.['rank_ndcg@10'];
    const health = [regR2, clsF1, ndcg].filter((v): v is number => typeof v === 'number');
    const score = health.length > 0 ? health.reduce((a, b) => a + b, 0) / health.length : NaN;

    return [
      { label: 'Latest Run', value: latestRun.run_id.slice(0, 12), icon: 'ledger' as IconName },
      { label: 'Run Count', value: String(runs.length), icon: 'pulse' as IconName },
      { label: 'Model Health', value: formatMetric(score), icon: 'shield' as IconName },
    ];
  }, [latestRun, runs.length]);

  return (
    <div className="app-shell">
      <div className="bg-grid" aria-hidden="true" />
      <header className="mission-header">
        <div>
          <p className="kicker">Deterministic Analytics Platform</p>
          <h1>Defense-Grade Decision Console</h1>
          <p className="subtitle">Controlled workflows for data intake, model training, and explainable prediction operations.</p>
        </div>
        <div className="header-badges">
          <div className="badge">
            <span className="badge-label">Mode</span>
            <strong>Development</strong>
          </div>
          <div className="badge">
            <span className="badge-label">Active View</span>
            <strong>{activeTitle[tab]}</strong>
          </div>
        </div>
      </header>

      <nav className="tab-rail" aria-label="Navigation">
        <button data-testid="tab-dataset" className={tab === 'dataset' ? 'tab active' : 'tab'} onClick={() => setTab('dataset')}>
          <Icon className="tab-icon" name="upload" />
          Data Intake
        </button>
        <button data-testid="tab-train" className={tab === 'train' ? 'tab active' : 'tab'} onClick={() => setTab('train')}>
          <Icon className="tab-icon" name="train" />
          Model Ops
        </button>
        <button data-testid="tab-runs" className={tab === 'runs' ? 'tab active' : 'tab'} onClick={() => setTab('runs')}>
          <Icon className="tab-icon" name="ledger" />
          Run Ledger
        </button>
        <button data-testid="tab-predict" className={tab === 'predict' ? 'tab active' : 'tab'} onClick={() => setTab('predict')}>
          <Icon className="tab-icon" name="predict" />
          Inference
        </button>
        <button data-testid="tab-query" className={tab === 'query' ? 'tab active' : 'tab'} onClick={() => setTab('query')}>
          <Icon className="tab-icon" name="pulse" />
          Query
        </button>
      </nav>

      <div data-testid="status-banner" className={`status-banner ${status.tone}`}>
        <span className="status-dot" aria-hidden="true" />
        <span>{status.message}</span>
      </div>

      {tab === 'dataset' && (
        <section className="panel fade-in">
          <h2>Data Intake Zone</h2>
          <p className="panel-intro">Load mission datasets and artifacts in staged order to maintain deterministic run provenance.</p>
          <h3 className="dataset-subtitle">Demo Preload</h3>
          <div className="demo-preload-row" data-testid="demo-preload-controls">
            <label className="field-group" htmlFor="demo-profile">
              <span>Dataset Profile</span>
              <select
                id="demo-profile"
                data-testid="input-demo-profile"
                className="text-input"
                value={demoProfile}
                onChange={(e) => setDemoProfile(e.target.value as 'small' | 'medium')}
              >
                <option value="small">small</option>
                <option value="medium">medium</option>
              </select>
            </label>
            <button data-testid="action-preload-demo" className="secondary-action" onClick={handlePreloadDemo}>
              Preload Synthetic Demo Data
            </button>
          </div>

          <h3 className="dataset-subtitle">Structured Data</h3>
          <div className="upload-grid">
            <label className="upload-card" htmlFor="entities-csv">
              <span className="upload-title">Entities Manifest</span>
              <span className="upload-help">Primary profile vectors and targets</span>
              <input
                id="entities-csv"
                data-testid="upload-entities"
                type="file"
                accept=".csv"
                onChange={(e) => handleUpload('entities', e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="upload-card" htmlFor="events-csv">
              <span className="upload-title">Events Stream</span>
              <span className="upload-help">Time-series operational events</span>
              <input
                id="events-csv"
                data-testid="upload-events"
                type="file"
                accept=".csv"
                onChange={(e) => handleUpload('events', e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="upload-card" htmlFor="interactions-csv">
              <span className="upload-title">Interactions Graph</span>
              <span className="upload-help">Cross-entity relationship intelligence</span>
              <input
                id="interactions-csv"
                data-testid="upload-interactions"
                type="file"
                accept=".csv"
                onChange={(e) => handleUpload('interactions', e.target.files?.[0] ?? null)}
              />
            </label>
          </div>

          <h3 className="dataset-subtitle">Artifact Ingestion</h3>
          <div className="upload-grid upload-grid-artifact">
            <label className="upload-card" htmlFor="artifacts-manifest-csv">
              <span className="upload-title">Artifacts Manifest</span>
              <span className="upload-help">Bulk artifact records and file references</span>
              <input
                id="artifacts-manifest-csv"
                data-testid="upload-artifacts-manifest"
                type="file"
                accept=".csv"
                onChange={(e) => handleArtifactsManifestUpload(e.target.files?.[0] ?? null)}
              />
            </label>

            <div className="upload-card artifact-form-card" data-testid="single-artifact-form">
              <span className="upload-title">Single Artifact</span>
              <span className="upload-help">Upload one image/audio/video file with optional metadata</span>
              <div className="artifact-form-grid">
                <label className="field-group" htmlFor="artifact-file">
                  <span>Artifact File</span>
                  <input
                    id="artifact-file"
                    data-testid="upload-single-artifact-file"
                    type="file"
                    accept="image/*,audio/*,video/*"
                    onChange={(e) => setArtifactFile(e.target.files?.[0] ?? null)}
                  />
                </label>

                <label className="field-group" htmlFor="artifact-type">
                  <span>Artifact Type</span>
                  <select
                    id="artifact-type"
                    data-testid="input-artifact-type"
                    className="text-input"
                    value={artifactType}
                    onChange={(e) => setArtifactType(e.target.value)}
                  >
                    <option value="image">image</option>
                    <option value="audio">audio</option>
                    <option value="video">video</option>
                  </select>
                </label>

                <label className="field-group" htmlFor="artifact-entity-id">
                  <span>Entity ID (optional)</span>
                  <input
                    id="artifact-entity-id"
                    data-testid="input-artifact-entity-id"
                    className="text-input"
                    type="text"
                    placeholder="ent_000"
                    value={artifactEntityId}
                    onChange={(e) => setArtifactEntityId(e.target.value)}
                  />
                </label>

                <label className="field-group" htmlFor="artifact-timestamp">
                  <span>Timestamp (optional)</span>
                  <input
                    id="artifact-timestamp"
                    data-testid="input-artifact-timestamp"
                    className="text-input"
                    type="text"
                    placeholder="2026-03-11T12:00:00"
                    value={artifactTimestamp}
                    onChange={(e) => setArtifactTimestamp(e.target.value)}
                  />
                </label>

                <label className="field-group" htmlFor="artifact-metadata">
                  <span>Metadata JSON (optional)</span>
                  <textarea
                    id="artifact-metadata"
                    data-testid="input-artifact-metadata"
                    className="text-input artifact-textarea"
                    placeholder='{"source":"ui"}'
                    value={artifactMetadata}
                    onChange={(e) => setArtifactMetadata(e.target.value)}
                  />
                </label>
              </div>
              <button data-testid="action-upload-single-artifact" className="secondary-action" onClick={handleSingleArtifactUpload}>
                Upload Single Artifact
              </button>
            </div>
          </div>
        </section>
      )}
      {tab === 'train' && (
        <section className="panel fade-in">
          <h2>Model Operation</h2>
          <p className="panel-intro">Execute deterministic training with fixed seed controls and auditable metrics output.</p>
          <button data-testid="action-train" className="primary-action" onClick={handleTrain}>
            Execute Training Operation
          </button>
        </section>
      )}
      {tab === 'runs' && (
        <section className="panel fade-in">
          <h2>Run Ledger</h2>
          <p className="panel-intro">Review run identifiers and timestamps for governance-grade traceability.</p>
          <div className="metrics-grid" data-testid="run-metrics-grid">
            {runCards.map((card) => (
              <article className="metric-card" key={card.label}>
                <Icon className="metric-icon" name={card.icon} />
                <span className="metric-label">{card.label}</span>
                <strong className="metric-value">{card.value}</strong>
              </article>
            ))}
          </div>
          <button data-testid="action-refresh-runs" className="secondary-action" onClick={handleRefreshRuns}>
            Sync Run Ledger
          </button>
          <ul className="run-list" data-testid="runs-list">
            {runs.map((r) => (
              <li key={r.run_id} className="run-item">
                <span className="run-id">{r.run_id}</span>
                <span className="run-date">{new Date(r.created_at).toLocaleString()}</span>
                <div className="metric-chip-row">
                  <span className="metric-chip">
                    <Icon className="chip-icon" name="pulse" />
                    r2 {formatMetric(r.metrics?.reg_r2)}
                  </span>
                  <span className="metric-chip">
                    <Icon className="chip-icon" name="chip" />
                    f1 {formatMetric(r.metrics?.cls_f1)}
                  </span>
                  <span className="metric-chip">
                    <Icon className="chip-icon" name="predict" />
                    ndcg {formatMetric(r.metrics?.['rank_ndcg@10'])}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
      {tab === 'predict' && (
        <section className="panel fade-in">
          <h2>Inference Console</h2>
          <p className="panel-intro">Run explainable predictions against selected entity IDs and inspect ranked outputs.</p>
          <PredictForm onPredict={handlePredict} predictions={predictions} />
        </section>
      )}
      {tab === 'query' && (
        <section className="panel fade-in">
          <h2>Natural Language Query</h2>
          <p className="panel-intro">Ask questions in plain language to retrieve entity predictions and long-form grounded narratives.</p>
          <div className="predict-console">
            <div className="query-prompt-grid" data-testid="query-prompt-grid">
              {QUERY_PROMPT_PRESETS.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  className="prompt-chip"
                  data-testid={`prompt-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
                  onClick={() => handleUseQueryPrompt(item.prompt)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <input
              data-testid="input-query"
              className="text-input"
              type="text"
              placeholder="Choose a preset or enter a custom query"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
            />
            <button data-testid="action-query" className="primary-action" onClick={handleQuery}>
              Run Query
            </button>
            {queryResult && (
              <div data-testid="query-results">
                <p className="panel-intro">Interpretation: {queryResult.interpreted_as}</p>
                <ul className="prediction-list">
                  {queryResult.results.map((r) => (
                    <li key={r.entity_id} className="prediction-item">
                      <div className="prediction-header">
                        <Icon className="metric-icon" name="shield" />
                        <strong>{r.entity_id}</strong>
                      </div>
                      <div className="metric-chip-row">
                        <span className="metric-chip">Reg {r.regression.toFixed(3)}</span>
                        <span className="metric-chip">Prob {r.probability.toFixed(3)}</span>
                        <span className="metric-chip">Rank {r.ranking_score.toFixed(3)}</span>
                      </div>
                      {r.narrative && <p className="narrative-text">{r.narrative}</p>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};

interface PredictFormProps {
  onPredict: (entityIds: string) => void;
  predictions: EntityPrediction[];
}

const PredictForm: React.FC<PredictFormProps> = ({ onPredict, predictions }) => {
  const [ids, setIds] = useState('');
  const predictCards = useMemo(() => {
    if (predictions.length === 0) {
      return [
        { label: 'Entities Predicted', value: '0', icon: 'predict' as IconName },
        { label: 'Avg Probability', value: 'n/a', icon: 'pulse' as IconName },
        { label: 'Avg Rank Score', value: 'n/a', icon: 'chip' as IconName },
      ];
    }
    const avgProb = predictions.reduce((acc, p) => acc + p.probability, 0) / predictions.length;
    const avgRank = predictions.reduce((acc, p) => acc + p.ranking_score, 0) / predictions.length;

    return [
      { label: 'Entities Predicted', value: String(predictions.length), icon: 'predict' as IconName },
      { label: 'Avg Probability', value: formatMetric(avgProb), icon: 'pulse' as IconName },
      { label: 'Avg Rank Score', value: formatMetric(avgRank), icon: 'chip' as IconName },
    ];
  }, [predictions]);

  return (
    <div className="predict-console">
      <div className="metrics-grid" data-testid="predict-metrics-grid">
        {predictCards.map((card) => (
          <article className="metric-card" key={card.label}>
            <Icon className="metric-icon" name={card.icon} />
            <span className="metric-label">{card.label}</span>
            <strong className="metric-value">{card.value}</strong>
          </article>
        ))}
      </div>
      <input
        data-testid="input-predict-ids"
        className="text-input"
        type="text"
        placeholder="Enter entity IDs (comma separated)"
        value={ids}
        onChange={(e) => setIds(e.target.value)}
      />
      <button data-testid="action-predict" className="primary-action" onClick={() => onPredict(ids)}>
        Execute Inference
      </button>
      <ul className="prediction-list" data-testid="prediction-list">
        {predictions.map((p) => (
          <li className="prediction-item" key={p.entity_id}>
            <div className="prediction-header">
              <Icon className="metric-icon" name="shield" />
              <strong>{p.entity_id}</strong>
            </div>
            <div className="metric-chip-row">
              <span className="metric-chip">Reg {p.regression.toFixed(3)}</span>
              <span className="metric-chip">Prob {p.probability.toFixed(3)}</span>
              <span className="metric-chip">Rank {p.ranking_score.toFixed(3)}</span>
            </div>
            {(p.narrative_long || p.narrative) && <p className="narrative-text">{p.narrative_long || p.narrative}</p>}
          </li>
        ))}
      </ul>
    </div>
  );
};
