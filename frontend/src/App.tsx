import React, { useState } from 'react';
import { uploadCsv, triggerTrain, listRuns, predict, RunInfo, EntityPrediction } from './api';

type Tab = 'dataset' | 'train' | 'runs' | 'predict';

export const App: React.FC = () => {
  const [tab, setTab] = useState<Tab>('dataset');
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [predictions, setPredictions] = useState<EntityPrediction[]>([]);
  const [status, setStatus] = useState<string>('');

  const handleUpload = async (kind: 'entities' | 'events' | 'interactions', file: File | null) => {
    if (!file) return;
    setStatus(`Uploading ${kind}...`);
    const path = `/ingest/${kind}`;
    const res = await uploadCsv(path, file);
    setStatus(`Uploaded ${kind}: ${res.success_rows} rows`);
  };

  const handleTrain = async () => {
    setStatus('Training...');
    const res = await triggerTrain();
    setStatus(`Trained run ${res.run_id}`);
  };

  const handleRefreshRuns = async () => {
    const data = await listRuns();
    setRuns(data);
  };

  const handlePredict = async (entityIds: string) => {
    const ids = entityIds.split(',').map((s) => s.trim()).filter(Boolean);
    if (ids.length === 0) return;
    const res = await predict(ids);
    setPredictions(res.predictions);
  };

  return (
    <div style={{ padding: '1rem', fontFamily: 'sans-serif' }}>
      <h1>Deterministic Multimodal Analytics</h1>
      <nav style={{ marginBottom: '1rem' }}>
        <button onClick={() => setTab('dataset')}>Dataset</button>
        <button onClick={() => setTab('train')}>Train</button>
        <button onClick={() => setTab('runs')}>Runs</button>
        <button onClick={() => setTab('predict')}>Predict</button>
      </nav>
      <p>{status}</p>
      {tab === 'dataset' && (
        <section>
          <h2>Upload CSVs</h2>
          <div>
            <label>
              Entities CSV:
              <input type="file" accept=".csv" onChange={(e) => handleUpload('entities', e.target.files?.[0] ?? null)} />
            </label>
          </div>
          <div>
            <label>
              Events CSV:
              <input type="file" accept=".csv" onChange={(e) => handleUpload('events', e.target.files?.[0] ?? null)} />
            </label>
          </div>
          <div>
            <label>
              Interactions CSV:
              <input type="file" accept=".csv" onChange={(e) => handleUpload('interactions', e.target.files?.[0] ?? null)} />
            </label>
          </div>
        </section>
      )}
      {tab === 'train' && (
        <section>
          <h2>Train</h2>
          <button onClick={handleTrain}>Start Training</button>
        </section>
      )}
      {tab === 'runs' && (
        <section>
          <h2>Runs</h2>
          <button onClick={handleRefreshRuns}>Refresh</button>
          <ul>
            {runs.map((r) => (
              <li key={r.run_id}>
                {r.run_id} – {r.created_at}
              </li>
            ))}
          </ul>
        </section>
      )}
      {tab === 'predict' && (
        <section>
          <h2>Predict</h2>
          <PredictForm onPredict={handlePredict} predictions={predictions} />
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
  return (
    <div>
      <input
        type="text"
        placeholder="Entity IDs comma-separated"
        value={ids}
        onChange={(e) => setIds(e.target.value)}
      />
      <button onClick={() => onPredict(ids)}>Predict</button>
      <ul>
        {predictions.map((p) => (
          <li key={p.entity_id}>
            {p.entity_id}: reg={p.regression.toFixed(3)} prob={p.probability.toFixed(3)} rank={p.ranking_score.toFixed(3)}
          </li>
        ))}
      </ul>
    </div>
  );
};
