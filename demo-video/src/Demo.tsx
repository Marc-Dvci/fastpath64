import React from "react";
import {AbsoluteFill, Audio, Sequence, staticFile} from "remotion";
import {colors, scenes, sec} from "./theme";
import {
  SceneGrep, SceneKernel, SceneReproduce, SceneResult, SceneSilicon, SceneToggle,
} from "./scenes";

const registry: Record<string, React.FC> = {
  grep: SceneGrep,
  silicon: SceneSilicon,
  toggle: SceneToggle,
  kernel: SceneKernel,
  result: SceneResult,
  reproduce: SceneReproduce,
};

export const FastPath64Demo: React.FC = () => (
  <AbsoluteFill style={{backgroundColor: colors.bg}}>
    {scenes.map((s) => {
      const Comp = registry[s.id];
      return (
        <Sequence key={s.id} from={sec(s.start)} durationInFrames={sec(s.duration)} name={s.id}>
          <Audio src={staticFile(s.audio)} />
          <Comp />
        </Sequence>
      );
    })}
  </AbsoluteFill>
);
