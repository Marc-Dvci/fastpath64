/** Dark surface for video; series colours are the validated categorical pair's dark steps. */
export const colors = {
  bg: "#0d0d0d",
  panel: "#161615",
  panelRaised: "#1f1f1e",
  border: "#2c2c2a",
  text: "#ffffff",
  textSoft: "#c3c2b7",
  textMuted: "#898781",
  grid: "#2c2c2a",
  stock: "#3987e5",   // categorical slot 1, dark step
  fast: "#d95926",    // categorical slot 2, dark step
  good: "#0ca30c",
  warn: "#fab219",
};

export const fonts = {
  sans: 'system-ui, -apple-system, "Segoe UI", Inter, sans-serif',
  mono: '"Cascadia Mono", "Cascadia Code", Consolas, "SF Mono", monospace',
};

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

/** Scene boundaries in seconds; total drives DURATION_IN_FRAMES. */
export const scenes = [
  {id: "grep", start: 0, duration: 25},
  {id: "silicon", start: 25, duration: 25},
  {id: "toggle", start: 50, duration: 25},
  {id: "kernel", start: 75, duration: 40},
  {id: "result", start: 115, duration: 30},
  {id: "reproduce", start: 145, duration: 20},
] as const;

export const DURATION_SECONDS = scenes.reduce((a, s) => Math.max(a, s.start + s.duration), 0);
export const DURATION_IN_FRAMES = DURATION_SECONDS * FPS;

export const sec = (s: number) => Math.round(s * FPS);
