import {defineConfig} from 'vite';
import mc from '@motion-canvas/vite-plugin';
import ff from '@motion-canvas/ffmpeg';
const motionCanvas: any = (mc as any).default ?? mc;
const ffmpeg: any = (ff as any).default ?? ff;
export default defineConfig({plugins:[motionCanvas({project: ['./src/p8_tools.ts', './src/p8_notes.ts', './src/p8_choose.ts']}),ffmpeg()]});
