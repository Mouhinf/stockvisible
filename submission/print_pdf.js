// Imprime un HTML statique en PDF A4 avec Chromium (Playwright), JavaScript désactivé.
// Usage : node submission/print_pdf.js <entrée.html> <sortie.pdf> <fr|en>
const { chromium } = require('@playwright/test');
const { readFileSync } = require('node:fs');

(async () => {
  const [input, output, lang] = process.argv.slice(2);
  const browser = await chromium.launch();
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.setContent(readFileSync(input, 'utf-8'), { waitUntil: 'load' });
  const label = lang === 'fr' ? 'Page' : 'Page';
  await page.pdf({
    path: output,
    format: 'A4',
    printBackground: true,
    margin: { top: '15mm', right: '15mm', bottom: '17mm', left: '15mm' },
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate:
      '<div style="width:100%;font-size:7pt;color:#4B5563;padding:0 15mm;display:flex;justify-content:space-between;' +
      'font-family:system-ui,Arial,sans-serif"><span>StockVisible</span>' +
      `<span>${label} <span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`,
  });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
