import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import * as api from './api';

vi.mock('./api', () => ({
  uploadCsv: vi.fn(),
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

    const entitiesInput = screen.getByLabelText('Entities CSV:') as HTMLInputElement;
    const file = new File(['entity_id,attributes\nE1,"{}"'], 'entities.csv', { type: 'text/csv' });
    fireEvent.change(entitiesInput, { target: { files: [file] } });

    await waitFor(() => {
      expect(api.uploadCsv).toHaveBeenCalledWith('/ingest/entities', file);
    });

    expect(screen.getByText('Uploaded entities: 1 rows')).toBeInTheDocument();
  });

  it('trains model from Train tab', async () => {
    vi.mocked(api.triggerTrain).mockResolvedValue({
      run_id: 'run_123',
      metrics: { reg_mae: 0.1 },
    });

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Train' }));
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => {
      expect(api.triggerTrain).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByText('Trained run run_123')).toBeInTheDocument();
  });

  it('loads run history from Runs tab', async () => {
    vi.mocked(api.listRuns).mockResolvedValue([
      { run_id: 'run_a', created_at: '2026-01-01T00:00:00Z', metrics: { reg_mae: 0.12 } },
    ]);

    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'Runs' }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    await waitFor(() => {
      expect(api.listRuns).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByText(/run_a/)).toBeInTheDocument();
  });

  it('predicts from Predict tab', async () => {
    vi.mocked(api.predict).mockResolvedValue({
      run_id: 'run_123',
      predictions: [{ entity_id: 'E1', regression: 0.11, probability: 0.72, ranking_score: 0.4 }],
    });

    render(<App />);

    fireEvent.click(screen.getAllByRole('button', { name: 'Predict' })[0]);
    fireEvent.change(screen.getByPlaceholderText('Entity IDs comma-separated'), { target: { value: 'E1, E2' } });
    fireEvent.click(screen.getAllByRole('button', { name: 'Predict' })[1]);

    await waitFor(() => {
      expect(api.predict).toHaveBeenCalledWith(['E1', 'E2']);
    });

    expect(screen.getByText(/E1: reg=0.110 prob=0.720 rank=0.400/)).toBeInTheDocument();
  });
});
