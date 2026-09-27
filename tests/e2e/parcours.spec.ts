import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

// Parcours critique : import → visualisation → test → budget → panier → validation → export.
// Données d'import SYNTHÉTIQUES (fixtures/…_SYNTHETIQUE.csv) ; le reste lit le jeu de référence réel.

const FIXTURES = join(__dirname, 'fixtures');
const EXPORT_HEADER =
  'serie,lots,quantite_unites_normalisees,cout_unitaire_hypothese,taille_lot,stock_actuel_hypothese,budget,cout_panier,demande_couverte_esperee_jour,gain_vs_sans_achat,manque_espere_jour,moteur_demande_panier,moteur_gele_projet,scenarios,donnees,horodatage_utc,empreinte_panier';

function watchConsole(page: Page): string[] {
  const problems: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') problems.push(`console: ${msg.text()}`);
  });
  page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
  return problems;
}

// Desktop : menu en haut. Mobile : Streamlit replie la navigation dans un panneau latéral.
async function goTo(page: Page, screen: 'Vérifier' | 'Comprendre' | 'Acheter') {
  const top = page.getByTestId('stTopNavLink').filter({ hasText: screen });
  if (await top.count()) {
    await top.click();
  } else {
    // Panneau replié par translation hors écran : « visible » pour le DOM, mais pas cliquable.
    await page.getByRole('button', { name: 'keyboard_double_arrow_right' }).click();
    const side = page.getByTestId('stSidebarNavLink').filter({ hasText: screen });
    await expect(side).toBeInViewport();
    await side.click();
  }
  await expect(page.getByRole('heading', { level: 1, name: screen })).toBeVisible();
}

async function upload(page: Page, file: string) {
  await page.getByTestId('stFileUploader').locator('input[type="file"]').setInputFiles(join(FIXTURES, file));
}

test('parcours critique complet, zéro erreur console', async ({ page }) => {
  const problems = watchConsole(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'Vérifier' })).toBeVisible();

  // 1. Import : un fichier corrompu est refusé avec le détail, un fichier conforme est accepté.
  await goTo(page, 'Comprendre');
  await upload(page, 'import_corrompu_SYNTHETIQUE.csv');
  await expect(page.getByText('Fichier non conforme au contrat de données')).toBeVisible();
  await expect(page.getByText('DUPLICATE_DATE')).toBeVisible();
  await upload(page, 'import_conforme_SYNTHETIQUE.csv');
  await expect(page.getByText(/Fichier conforme : 10 lignes, 2 série\(s\)/)).toBeVisible();

  // 2. Visualisation : graphique ventes + disponibilité, puis changement de série.
  await expect(page.locator('.js-plotly-plot')).toBeVisible();
  await expect(page.getByText('jours avec au moins une rupture : 30')).toBeVisible();
  const select = page.getByTestId('stSelectbox');
  await select.click();
  await page.keyboard.type('magasin 388 · produit 763');
  await page.keyboard.press('Enter');
  await expect(page.getByText('jours avec au moins une rupture : 37')).toBeVisible();

  // 3. Test : la preuve (validation + test final scellé) est affichée.
  await goTo(page, 'Vérifier');
  await expect(page.getByText(/Moteur gelé : ML\. Sur le test final, exécuté une seule fois/)).toBeVisible();
  await expect(page.getByText(/Test final ouvert 1 fois/)).toBeVisible();
  await expect(page.getByTestId('stDataFrame')).toHaveCount(1);

  // 4. Budget : le curseur recalcule réellement le panier.
  await goTo(page, 'Acheter');
  const cost = page.getByTestId('stMetric').filter({ hasText: 'Coût du panier' }).getByTestId('stMetricValue');
  const slider = page.getByRole('slider');
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(cost).toHaveText('0.00');
  await page.keyboard.press('End');
  await expect(cost).not.toHaveText('0.00');
  const fullCost = Number(await cost.innerText());
  expect(fullCost).toBeGreaterThan(0);

  // 5. Panier : tableau des lots achetés.
  await expect(page.getByTestId('stDataFrame')).toHaveCount(2);
  await expect(page.getByText("L'export n'est possible qu'après validation")).toBeVisible();

  // 6. Validation.
  await page.getByRole('button', { name: 'Valider le panier' }).click();
  await expect(page.getByText(/Panier validé le .* \(empreinte [0-9a-f]{16}\)/)).toBeVisible();

  // 7. Export : le CSV téléchargé porte le panier, ses hypothèses et sa provenance.
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Exporter le panier (CSV)' }).click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^panier_[0-9a-f]{16}\.csv$/);
  const csv = readFileSync(await download.path(), 'utf-8').trim().split('\n');
  expect(csv[0]).toBe(EXPORT_HEADER);
  expect(csv).toHaveLength(4); // en-tête + 3 produits
  const costTotal = Number(csv[1].split(',')[7]); // cout_panier (colonnes 0-7 sans virgule)
  expect(costTotal).toBe(fullCost); // identique à l'écran, arrondi compris
  expect(csv[1]).toContain('B1 estimé');

  // Export JSON : même vue que le CSV et que l'écran.
  const [jsonDownload] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Exporter le panier (JSON)' }).click(),
  ]);
  expect(jsonDownload.suggestedFilename()).toMatch(/^panier_[0-9a-f]{16}\.json$/);
  const view = JSON.parse(readFileSync(await jsonDownload.path(), 'utf-8'));
  expect(view.indicateurs.cout_panier).toBe(fullCost);
  expect(view.panier).toHaveLength(3);
  expect(view.moteur.moteur_gele_projet).toMatch(/^ML/);
  expect(view.empreinte_panier).toBe(download.suggestedFilename().slice(7, 23));

  expect(problems, problems.join('\n')).toEqual([]);
});
