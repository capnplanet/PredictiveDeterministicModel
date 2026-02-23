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

  await page.getByLabel('Entities CSV:').setInputFiles({
    name: 'entities.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(entitiesCsv, 'utf-8'),
  });
  await expect(page.getByText('Uploaded entities: 2 rows')).toBeVisible();

  const eventsCsv =
    'timestamp,entity_id,event_type,event_value,event_metadata\n' +
    `2025-01-01T00:10:00,${entityA},purchase,10.5,"{""source"":""e2e""}"\n` +
    `2025-01-01T00:11:00,${entityB},purchase,2.0,"{""source"":""e2e""}"\n`;

  await page.getByLabel('Events CSV:').setInputFiles({
    name: 'events.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(eventsCsv, 'utf-8'),
  });
  await expect(page.getByText('Uploaded events: 2 rows')).toBeVisible();

  const interactionsCsv =
    'timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n' +
    `2025-01-01T00:12:00,${entityA},${entityB},linked,1,"{""source"":""e2e""}"\n`;

  await page.getByLabel('Interactions CSV:').setInputFiles({
    name: 'interactions.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(interactionsCsv, 'utf-8'),
  });
  await expect(page.getByText('Uploaded interactions: 1 rows')).toBeVisible();

  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await page.getByRole('button', { name: 'Start Training' }).click();
  await expect(page.getByText(/Trained run\s+[a-f0-9]{64}/)).toBeVisible();

  await page.getByRole('button', { name: 'Runs', exact: true }).click();
  await page.getByRole('button', { name: 'Refresh' }).click();
  await expect(page.getByText(/[a-f0-9]{64}/)).toBeVisible();

  await page.getByRole('button', { name: 'Predict', exact: true }).first().click();
  await page.getByPlaceholder('Entity IDs comma-separated').fill(entityA);
  await page.getByRole('button', { name: 'Predict', exact: true }).nth(1).click();
  await expect(page.getByText(new RegExp(`${entityA}: reg=`))).toBeVisible();
});

test('failure path: invalid entities csv yields zero successful uploads', async ({ page }) => {
  await page.goto('/');

  const invalidCsv = 'entity_id,created_at\nBROKEN_1,2025-01-01T00:00:00\n';
  await page.getByLabel('Entities CSV:').setInputFiles({
    name: 'invalid_entities.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(invalidCsv, 'utf-8'),
  });

  await expect(page.getByText('Uploaded entities: 0 rows')).toBeVisible();
});
