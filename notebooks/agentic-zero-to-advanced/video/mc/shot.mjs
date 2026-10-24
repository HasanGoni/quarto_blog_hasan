import {chromium} from 'playwright-core';
const b = await chromium.launch(); const p = await b.newPage({viewport:{width:1600,height:900}});
p.on('console', m => console.log('console:', m.text().slice(0,150))); p.on('pageerror', e => console.log('pageerror:', e.message.slice(0,200)));
await p.goto(`http://127.0.0.1:${process.env.PORT||9020}/${process.env.PROJECT ? 'src/' + process.env.PROJECT : ''}`, {waitUntil:'networkidle'}); await p.waitForTimeout(8000);
await p.screenshot({path:'editor.png'}); await b.close();
