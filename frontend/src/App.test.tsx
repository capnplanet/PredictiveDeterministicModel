import { beforeEach, describe, expect, it, vi } from 'vitest';

import { listRuns, predict, triggerTrain, uploadCsv } from './api';

describe('api helpers', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });

  it('uploads CSV with multipart payload and returns ingestion report', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ total_rows: 1, success_rows: 1, failed_rows: 0, errors: [] }), { status: 200 }),
    );

    const file = new File(['x'], 'entities.csv');
    const res = await uploadCsv('/ingest/entities', file);

    expect(res.success_rows).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/ingest/entities', expect.objectContaining({ method: 'POST' }));

    const options = fetchMock.mock.calls[0][1] as RequestInit;
    expect(options.body).toBeInstanceOf(FormData);
  });

  it('triggers train with JSON body and returns run info', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ run_id: 'abc', metrics: { loss: 0.1 } }), { status: 200 }),
    );

    const res = await triggerTrain();

    expect(res.run_id).toBe('abc');
    expect(fetchMock).toHaveBeenCalledWith(
      '/train',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      }),
    );
  });

  it('lists runs from /runs', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify([{ run_id: 'abc', created_at: '2025-01-01T00:00:00', metrics: { loss: 0.1 } }]), {
        status: 200,
      }),
    );

    const runs = await listRuns();

    expect(runs[0].run_id).toBe('abc');
    expect(fetchMock).toHaveBeenCalledWith('/runs');
  });

  it('predicts with entity_ids and explanations=true', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          run_id: 'abc',
          predictions: [{ entity_id: 'E00001', regression: 0.1, probability: 0.6, ranking_score: 0.5 }],
        }),
        { status: 200 },
      ),
    );

    const res = await predict(['E00001']);

    expect(res.predictions[0].entity_id).toBe('E00001');
    expect(fetchMock).toHaveBeenCalledWith(
      '/predict',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_ids: ['E00001'], explanations: true }),
      }),
    );
  });

  it('throws on non-OK responses', async () => {
    fetchMock.mockResolvedValueOnce(new Response('error', { status: 500 }));
    await expect(triggerTrain()).rejects.toThrow('Train failed: 500');
  });

  it('throws if runs endpoint fails', async () => {
    fetchMock.mockResolvedValueOnce(new Response('error', { status: 503 }));
    await expect(listRuns()).rejects.toThrow('List runs failed: 503');
  });

  it('throws if prediction endpoint fails', async () => {
    fetchMock.mockResolvedValueOnce(new Response('error', { status: 400 }));
    await expect(predict(['E00001'])).rejects.toThrow('Predict failed: 400');
  });

  it('throws if upload endpoint fails', async () => {
    fetchMock.mockResolvedValueOnce(new Response('error', { status: 422 }));
    const file = new File(['x'], 'entities.csv');
    await expect(uploadCsv('/ingest/entities', file)).rejects.toThrow('Upload failed: 422');
  });
});
