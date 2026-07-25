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

/**
 * Scene boundaries are derived from the measured narration, not hand-set: each scene runs for as
 * long as its clip needs plus a short tail, with a floor so the on-screen choreography still has
 * room if a line is short. Rewrite a line, re-run generate-narration.py, and the timings follow.
 */
import narration from "./generated-narration.json";

/** minimum seconds a scene needs for its own animation, independent of the voice */
const FLOORS: Record<string, number> = {
  grep: 22, silicon: 24, toggle: 22, kernel: 38, result: 28, reproduce: 20,
};
const TAIL = 1.3;

/** snap to a whole frame: Remotion requires integer frame counts, and frame-aligned
 *  durations keep the cumulative starts aligned too */
const snap = (seconds: number) => Math.ceil(seconds * FPS) / FPS;

export const scenes = (() => {
  let cursor = 0;
  return narration.map((clip) => {
    const duration = snap(Math.max(FLOORS[clip.id] ?? 20, clip.duration + TAIL));
    const scene = {id: clip.id, start: cursor, duration, audio: clip.file};
    cursor += duration;
    return scene;
  });
})();

export const DURATION_SECONDS = scenes.reduce((a, s) => Math.max(a, s.start + s.duration), 0);
export const DURATION_IN_FRAMES = Math.round(DURATION_SECONDS * FPS);

export const sec = (s: number) => Math.round(s * FPS);
