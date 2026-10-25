import {defineConfig} from 'vite';
import mc from '@motion-canvas/vite-plugin';
import ff from '@motion-canvas/ffmpeg';
const motionCanvas: any = (mc as any).default ?? mc;
const ffmpeg: any = (ff as any).default ?? ff;
export default defineConfig({plugins:[motionCanvas({project: ['./src/p8_tools.ts', './src/p8_notes.ts', './src/p8_choose.ts','./src/p14_desk.ts','./src/p14_fits.ts','./src/p14_brief.ts','./src/p14_practice.ts','./src/p14_frameworks.ts']}),ffmpeg()]});
