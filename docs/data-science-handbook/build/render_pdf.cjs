// Render the standalone HTML to an A4 PDF with headless Chromium (Playwright).
// CommonJS so that NODE_PATH can resolve a globally installed playwright.
const { chromium } = require("playwright");
const path = require("node:path");

(async () => {
  const [, , htmlPath, pdfPath] = process.argv;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(htmlPath), { waitUntil: "load" });
  await page.pdf({
    path: pdfPath,
    format: "A4",
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: `<div style="font-size:7.5pt;color:#888;width:100%;padding:0 15mm;text-align:right;font-family:DejaVu Sans">Data Science End-to-End Handbook</div>`,
    footerTemplate: `<div style="font-size:8pt;color:#888;width:100%;text-align:center;font-family:DejaVu Sans"><span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
    margin: { top: "18mm", bottom: "16mm", left: "15mm", right: "15mm" },
  });
  await browser.close();
  console.log("PDF written:", pdfPath);
})();
