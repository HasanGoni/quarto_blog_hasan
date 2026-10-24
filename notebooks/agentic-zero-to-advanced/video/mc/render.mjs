import {chromium} from 'playwright-core';
const b = await chromium.launch(); const p = await b.newPage({viewport:{width:1600,height:900}});
p.on('pageerror', e => console.log('pageerror:', e.message.slice(0,200)));
await p.goto(`http://127.0.0.1:${process.env.PORT||9020}/${process.env.PROJECT ? 'src/' + process.env.PROJECT : ''}`, {waitUntil:'networkidle'}); await p.waitForTimeout(6000);
const t0 = Date.now();
await p.getByRole('button',{name:'RENDER'}).first().click();
for (let i=0;i<120;i++){ await p.waitForTimeout(1000); const txt = await p.locator('body').innerText(); if (/RENDER\n|^RENDER$/m.test(txt) && !/ABORT|STOP|CANCEL/i.test(txt) && i>3) break; }
console.log('render seconds', (Date.now()-t0)/1000); await p.screenshot({path:'after.png'}); await b.close();
