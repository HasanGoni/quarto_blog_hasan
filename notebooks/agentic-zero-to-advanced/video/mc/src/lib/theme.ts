// Landscape 1920x1080 (rendered at 60 fps, then ffmpeg scales to 1280x720 / 30 fps like the Manim chapters so tools can be mixed in one cut).
// YELLOW is for the subtitle highlighter ONLY (user rule); everything else is white or ACCENT. Green = plain code, ACCENT = a model, AMBER = warning.
export const YELLOW = '#F6D743';
export const ACCENT = '#6EC6FF';
export const GREEN = '#5BD18B';
export const AMBER = '#F2A93B';
export const RED = '#FF6B5E';
export const GREY = '#9aa0a6';
export const BG = '#181818';
export const PANEL = '#232325';
export const SANS = 'DejaVu Sans, sans-serif';
export const MONO = 'DejaVu Sans Mono, monospace';
// Frame 1920x1080, origin in the middle, y down. Content stays above the subtitle band (the Manim chapters keep the bottom quarter free).
export const CONTENT_TOP = -500;
export const CONTENT_BOTTOM = 290;
export const SUB_Y = 410;
export const SUB_W = 1640;
