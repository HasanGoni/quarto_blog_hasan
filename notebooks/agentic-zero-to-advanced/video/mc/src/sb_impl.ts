import {makeProject} from '@motion-canvas/core';
import scene from './scenes/sb_impl?scene';
import audio from './sb_impl/audio.wav';
export default makeProject({scenes: [scene], audio});
