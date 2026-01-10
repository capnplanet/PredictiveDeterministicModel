import { describe, it, expect, vi, beforeEach } from 'vitest';
import { uploadCsv, triggerTrain, listRuns, predict } from './api';

describe('api helpers', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    // @ts-expect-error override fetch for tests
    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/ingest')) {
        return new Response(JSON.stringify({ total_rows: 1, success_rows: 1, failed_rows: 0, errors: [] }), { status: 200 });
      }
      if (url.endsWith('/train')) {
        return new Response(JSON.stringify({ run_id: 'abc', metrics: { loss: 0.1 } }), { status: 200 });
      }
      if (url.endsWith('/runs')) {
        return new Response(JSON.stringify([{ run_id: 'abc', created_at: '2025-01-01T00:00:00', metrics: { loss: 0.1 } }]), { status: 200 });
      }
      if (url.endsWith('/predict')) {
        return new Response(JSON.stringify({ run_id: 'abc', predictions: [{ entity_id: 'E00001', regression: 0.1, probability: 0.6, ranking_score: 0.5 }] }), { status: 200 });
      }
      return new Response('not found', { status: 404 });
    }) as unknown as typeof fetch;
  });

  it('uploads CSV', async () => {
    const file = new File(['x'], 'entities.csv');
    const res = await uploadCsv('/ingest/entities', file);
    expect(res.success_rows).toBe(1);
  });

  it('triggers train', async () => {
    const res = await triggerTrain();
    expect(res.run_id).toBe('abc');
  });

  it('lists runs', async () => {
    const runs = await listRuns();
    expect(runs[0].run_id).toBe('abc');
  });

  it('predicts', async () => {
    const res = await predict(['E00001']);
    expect(res.predictions[0].entity_id).toBe('E00001');
  });
});
