import React from "react";
import {Composition} from "remotion";
import {FastPath64Demo} from "./Demo";
import {DURATION_IN_FRAMES, FPS, HEIGHT, WIDTH} from "./theme";

export const RemotionRoot: React.FC = () => (
  <Composition
    id="FastPath64Demo"
    component={FastPath64Demo}
    durationInFrames={DURATION_IN_FRAMES}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
);
