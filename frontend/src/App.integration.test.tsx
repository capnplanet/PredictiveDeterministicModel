import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import * as api from './api';

vi.mock('./api', () => ({
  uploadCsv: vi.fn(),
  uploadArtifactsManifest: vi.fn(),
  uploadSingleArtifact: vi.fn(),
  preloadDemoData: vi.fn(),
  triggerTrain: vi.fn(),
  listRuns: vi.fn(),
  predict: vi.fn(),
  queryPredictions: vi.fn(),
}));

describe('App integration flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('uploads entities CSV from Dataset tab', async () => {
    vi.mocked(api.uploadCsv).mockResolvedValue({
      total_rows: 1,
      success_rows: 1,
      failed_rows: 0,
      errors: [],
    });

    render(<App />);

    const entitiesInput = screen.getByTestId('upload-entities') as HTMLInputElement;
    const file = new File(['entity_id,attributes\nE1,"{}"'], 'entities.csv', { type: 'text/csv' });
    fireEvent.change(entitiesInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(api.uploadCsv).toHaveBeenCalledWith('/ingest/entities', file);
    });

    expect(screen.getByTestId('status-banner').textContent).toContain('Upload complete: entities (1/1 rows).');
  });

  it('trains model from Train tab', async () => {
    vi.mocked(api.triggerTrain).mockResolvedValue({
      run_id: 'run_123',
      metrics: { reg_mae: 0.1 },
    });

    render(<App />);

    fireEvent.click(screen.getByTestId('tab-train'));
    fireEvent.click(screen.getByTestId('action-train'));

    await waitFor(() => {
      expect(api.triggerTrain).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByTestId('status-banner').textContent).toContain('Training complete. Run ID: run_123');
  });

  it('uploads artifact manifest CSV from Dataset tab', async () => {
    vi.mocked(api.uploadArtifactsManifest).mockResolvedValue({
      total_rows: 2,
      success_rows: 2,
      failed_rows: 0,
      errors: [],
    });

    render(<App />);

    const manifestInput = screen.getByTestId('upload-artifacts-manifest') as HTMLInputElement;
    const file = new File(['sha256,artifact_type\nabc,image'], 'artifacts_manifest.csv', { type: 'text/csv' });
    fireEvent.change(manifestInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(api.uploadArtifactsManifest).toHaveBeenCalledWith(file);
    });

    expect(screen.getByTestId('status-banner').textContent).toContain('Artifact manifest upload complete (2/2 rows).');
  });

  it('uploads single artifact from Dataset tab', async () => {
    vi.mocked(api.uploadSingleArtifact).mockResolvedValue({
      artifact_id: '1',
      sha256: 'abcdef1234567890',
      artifact_type: 'image',
    });

    render(<App />);

    const artifactFile = new File(['fake-image-content'], 'sample.png', { type: 'image/png' });
    fireEvent.change(screen.getByTestId('upload-single-artifact-file'), { target: { files: [artifactFile] } });
    fireEvent.change(screen.getByTestId('input-artifact-type'), { target: { value: 'image' } });
    fireEvent.change(screen.getByTestId('input-artifact-entity-id'), { target: { value: 'E1' } });
    fireEvent.change(screen.getByTestId('input-artifact-timestamp'), { target: { value: '2026-03-11T12:00:00' } });
    fireEvent.change(screen.getByTestId('input-artifact-metadata'), { target: { value: '{"source":"ui"}' } });
    fireEvent.click(screen.getByTestId('action-upload-single-artifact'));

    await waitFor(() => {
      expect(api.uploadSingleArtifact).toHaveBeenCalledWith({
        file: artifactFile,
        artifactType: 'image',
        entityId: 'E1',
        timestamp: '2026-03-11T12:00:00',
        metadata: '{"source":"ui"}',
      });
    });

    expect(screen.getByTestId('status-banner').textContent).toContain('Single artifact upload complete: abcdef123456...');
  });

  it('preloads synthetic demo dataset from Dataset tab', async () => {
    vi.mocked(api.preloadDemoData).mockResolvedValue({
      profile: 'small',
      output_dir: 'data/demo_preload/20260311_000000',
      entities: { total_rows: 18, success_rows: 18, failed_rows: 0, errors: [] },
      events: { total_rows: 120, success_rows: 120, failed_rows: 0, errors: [] },
      interactions: { total_rows: 48, success_rows: 48, failed_rows: 0, errors: [] },
      artifacts_manifest: { total_rows: 12, success_rows: 12, failed_rows: 0, errors: [] },
      single_artifact: { artifact_id: 'x', sha256: 'y', artifact_type: 'image' },
      features: { updated_artifacts: 13 },
      training: { run_id: 'demo_run_123', metrics: { reg_mae: 0.1 } },
    });

    render(<App />);

    fireEvent.change(screen.getByTestId('input-demo-profile'), { target: { value: 'small' } });
    fireEvent.click(screen.getByTestId('action-preload-demo'));

    await waitFor(() => {
      expect(api.preloadDemoData).toHaveBeenCalledWith('small', true, true);
    });

    expect(screen.getByTestId('status-banner').textContent).toContain('Demo preload complete: entities 18/18');
  });

  it('loads run history from Runs tab', async () => {
    vi.mocked(api.listRuns).mockResolvedValue([
      { run_id: 'run_a', created_at: '2026-01-01T00:00:00Z', metrics: { reg_mae: 0.12 } },
    ]);

    render(<App />);

    fireEvent.click(screen.getByTestId('tab-runs'));
    fireEvent.click(screen.getByTestId('action-refresh-runs'));

    await waitFor(() => {
      expect(api.listRuns).toHaveBeenCalledTimes(1);
    });

    expect(screen.getAllByText('run_a').length).toBeGreaterThan(0);
  });

  it('predicts from Predict tab', async () => {
    vi.mocked(api.predict).mockResolvedValue({
      run_id: 'run_123',
      predictions: [{ entity_id: 'E1', regression: 0.11, probability: 0.72, ranking_score: 0.4, narrative: 'Entity E1 narrative.' }],
    });

    render(<App />);

    fireEvent.click(screen.getByTestId('tab-predict'));
    fireEvent.change(screen.getByTestId('input-predict-ids'), { target: { value: 'E1, E2' } });
    fireEvent.click(screen.getByTestId('action-predict'));

    await waitFor(() => {
      expect(api.predict).toHaveBeenCalledWith(['E1', 'E2'], 'both');
    });

    expect(screen.getByText('E1')).toBeInTheDocument();
    expect(screen.getByText('Reg 0.110')).toBeInTheDocument();
    expect(screen.getByText('Prob 0.720')).toBeInTheDocument();
    expect(screen.getByText('Rank 0.400')).toBeInTheDocument();
    expect(screen.getByText('Entity E1 narrative.')).toBeInTheDocument();
  });

  it('runs natural language query and renders results', async () => {
    vi.mocked(api.queryPredictions).mockResolvedValue({
      run_id: 'run_123',
      query: 'show strongest entities',
      interpreted_as: 'Keyword retrieval over entity IDs and deterministic predictions.',
      llm_used: false,
      results: [
        {
          entity_id: 'E2',
          regression: 0.2,
          probability: 0.8,
          ranking_score: 0.5,
          narrative: 'Entity E2 narrative.',
        },
      ],
    });

    render(<App />);

    fireEvent.click(screen.getByTestId('tab-query'));
    fireEvent.change(screen.getByTestId('input-query'), { target: { value: 'show strongest entities' } });
    fireEvent.click(screen.getByTestId('action-query'));

    await waitFor(() => {
      expect(api.queryPredictions).toHaveBeenCalledWith({ query: 'show strongest entities', limit: 5 });
    });

    expect(screen.getByText('Entity E2 narrative.')).toBeInTheDocument();
  });
});
