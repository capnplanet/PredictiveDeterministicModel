import { expect, test } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

test('happy path: ingest -> train -> runs -> predict', async ({ page }) => {
  const suffix = Date.now().toString();
  const entityA = `E2E_${suffix}_A`;
  const entityB = `E2E_${suffix}_B`;

  await page.goto('/');

  const entitiesCsv =
    'entity_id,attributes,created_at\n' +
    `${entityA},"{""x"":0.1,""y"":0.2,""z"":0.3,""target_regression"":0.9,""target_binary"":1,""target_ranking"":0.8}",2025-01-01T00:00:00\n` +
    `${entityB},"{""x"":0.2,""y"":0.1,""z"":0.4,""target_regression"":0.3,""target_binary"":0,""target_ranking"":0.2}",2025-01-01T00:01:00\n`;

  await page.getByTestId('upload-entities').setInputFiles({
    name: 'entities.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(entitiesCsv, 'utf-8'),
  });
  await expect(page.getByTestId('status-banner')).toContainText('Upload complete: entities (2/2 rows).');

  const eventsCsv =
    'timestamp,entity_id,event_type,event_value,event_metadata\n' +
    `2025-01-01T00:10:00,${entityA},purchase,10.5,"{""source"":""e2e""}"\n` +
    `2025-01-01T00:11:00,${entityB},purchase,2.0,"{""source"":""e2e""}"\n`;

  await page.getByTestId('upload-events').setInputFiles({
    name: 'events.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(eventsCsv, 'utf-8'),
  });
  await expect(page.getByTestId('status-banner')).toContainText('Upload complete: events (2/2 rows).');

  const interactionsCsv =
    'timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n' +
    `2025-01-01T00:12:00,${entityA},${entityB},linked,1,"{""source"":""e2e""}"\n`;

  await page.getByTestId('upload-interactions').setInputFiles({
    name: 'interactions.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(interactionsCsv, 'utf-8'),
  });
  await expect(page.getByTestId('status-banner')).toContainText('Upload complete: interactions (1/1 rows).');

  await page.getByTestId('tab-train').click();
  await page.getByTestId('action-train').click();
  await expect(page.getByTestId('status-banner')).toContainText(/Training complete\. Run ID:\s+[a-f0-9]{64}/);

  await page.getByTestId('tab-runs').click();
  await page.getByTestId('action-refresh-runs').click();
  await expect(page.locator('[data-testid="runs-list"] .run-item').first()).toBeVisible();

  await page.getByTestId('tab-predict').click();
  await page.getByTestId('input-predict-ids').fill(entityA);
  await page.getByTestId('action-predict').click();
  await expect(page.locator('[data-testid="prediction-list"] .prediction-item strong', { hasText: entityA }).first()).toBeVisible();
});

test('failure path: invalid entities csv yields zero successful uploads', async ({ page }) => {
  await page.goto('/');

  const invalidCsv = 'entity_id,created_at\nBROKEN_1,2025-01-01T00:00:00\n';
  await page.getByTestId('upload-entities').setInputFiles({
    name: 'invalid_entities.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(invalidCsv, 'utf-8'),
  });

  await expect(page.getByTestId('status-banner')).toContainText('Upload complete: entities (0/1 rows).');
});
