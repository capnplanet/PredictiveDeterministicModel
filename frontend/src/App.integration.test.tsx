import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import * as api from './api';

vi.mock('./api', () => ({
  uploadCsv: vi.fn(),
  uploadArtifactsManifest: vi.fn(),
  uploadSingleArtifact: vi.fn(),
  triggerTrain: vi.fn(),
  listRuns: vi.fn(),
  predict: vi.fn(),
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
      predictions: [{ entity_id: 'E1', regression: 0.11, probability: 0.72, ranking_score: 0.4 }],
    });

    render(<App />);

    fireEvent.click(screen.getByTestId('tab-predict'));
    fireEvent.change(screen.getByTestId('input-predict-ids'), { target: { value: 'E1, E2' } });
    fireEvent.click(screen.getByTestId('action-predict'));

    await waitFor(() => {
      expect(api.predict).toHaveBeenCalledWith(['E1', 'E2']);
    });

    expect(screen.getByText('E1')).toBeInTheDocument();
    expect(screen.getByText('Reg 0.110')).toBeInTheDocument();
    expect(screen.getByText('Prob 0.720')).toBeInTheDocument();
    expect(screen.getByText('Rank 0.400')).toBeInTheDocument();
  });
});
