import React from "react";
import {interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";
import {colors, fonts} from "./theme";

/** vertical room reserved above every bar for its value label */
const LABEL_SPACE = 52;

export const Fade: React.FC<{at: number; children: React.ReactNode; dur?: number}> = ({
  at, dur = 12, children,
}) => {
  const f = useCurrentFrame();
  const o = interpolate(f - at, [0, dur], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const y = interpolate(f - at, [0, dur], [14, 0], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  return <div style={{opacity: o, transform: `translateY(${y}px)`}}>{children}</div>;
};

/** Monospace terminal panel with per-character reveal. */
export const Terminal: React.FC<{
  lines: {text: string; color?: string; at: number; cps?: number}[];
  width?: number;
}> = ({lines, width = 1180}) => {
  const f = useCurrentFrame();
  return (
    <div style={{
      width, background: colors.panel, border: `1px solid ${colors.border}`,
      borderRadius: 14, padding: "26px 30px", fontFamily: fonts.mono, fontSize: 27,
      lineHeight: 1.65, boxShadow: "0 30px 80px rgba(0,0,0,.5)",
    }}>
      {lines.map((l, i) => {
        const cps = l.cps ?? 42;
        const shown = Math.max(0, Math.floor(((f - l.at) / 30) * cps));
        if (shown <= 0) return <div key={i} style={{height: 45}} />;
        const txt = l.text.slice(0, shown);
        const typing = shown < l.text.length;
        return (
          <div key={i} style={{color: l.color ?? colors.textSoft, whiteSpace: "pre", height: 45}}>
            {txt}
            {typing ? <span style={{opacity: Math.floor(f / 8) % 2 ? 0.15 : 0.9}}>▋</span> : null}
          </div>
        );
      })}
    </div>
  );
};

export const Title: React.FC<{title: string; sub?: string; at?: number}> = ({title, sub, at = 0}) => (
  <Fade at={at}>
    <div style={{fontFamily: fonts.sans}}>
      <div style={{fontSize: 62, fontWeight: 700, color: colors.text, letterSpacing: -1.2}}>{title}</div>
      {sub ? (
        <div style={{fontSize: 29, color: colors.textMuted, marginTop: 12, maxWidth: 1400}}>{sub}</div>
      ) : null}
    </div>
  </Fade>
);

/** Animated grouped bars. Values arrive already measured; the bar grows, the number counts up. */
export const Bars: React.FC<{
  at: number;
  groups: {label: string; sub?: string; values: {v: number; color: string; name: string}[]; ratio?: string}[];
  max: number;
  unit?: string;
  height?: number;
}> = ({at, groups, max, unit = "tokens/s", height = 380}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  return (
    <div style={{fontFamily: fonts.sans}}>
      <div style={{color: colors.textMuted, fontSize: 20, marginBottom: 10}}>{unit}</div>
      {/* the value label lives inside the column, so the bar may only use the height that
          remains once the label is accounted for - otherwise a full-height bar pushes its
          label up out of the container and over whatever sits above it */}
      {/* only the bar row carries a fixed height: the group's labels sit below it and would
          otherwise make the column taller than its parent, overflowing upward */}
      <div style={{display: "flex", gap: 74, alignItems: "flex-start"}}>
        {groups.map((g, gi) => (
          <div key={gi} style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
            <div style={{display: "flex", gap: 10, alignItems: "flex-end", height}}>
              {g.values.map((b, bi) => {
                const delay = at + gi * 6 + bi * 5;
                const p = spring({frame: f - delay, fps, config: {damping: 200, mass: 0.6}});
                const h = Math.max(0, (b.v / max) * (height - LABEL_SPACE) * p);
                return (
                  <div key={bi} style={{
                    display: "flex", flexDirection: "column", alignItems: "center",
                    justifyContent: "flex-end", height: "100%",
                  }}>
                    <div style={{
                      color: colors.text, fontSize: 30, fontWeight: 700, marginBottom: 10,
                      opacity: p > 0.35 ? 1 : 0, fontVariantNumeric: "tabular-nums",
                    }}>
                      {(b.v * p).toFixed(1)}
                    </div>
                    <div style={{
                      width: 96, height: h, background: b.color,
                      borderRadius: "6px 6px 0 0",
                    }} />
                  </div>
                );
              })}
            </div>
            <div style={{width: "100%", height: 1, background: colors.border, marginTop: 2}} />
            <div style={{color: colors.textSoft, fontSize: 25, marginTop: 14, textAlign: "center"}}>{g.label}</div>
            {g.sub ? <div style={{color: colors.textMuted, fontSize: 21, marginTop: 4}}>{g.sub}</div> : null}
            {g.ratio ? (
              <Fade at={at + 26}>
                <div style={{color: colors.text, fontSize: 34, fontWeight: 700, marginTop: 14}}>{g.ratio}</div>
              </Fade>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
};

export const Legend: React.FC<{items: {name: string; color: string}[]; at: number}> = ({items, at}) => (
  <Fade at={at}>
    <div style={{display: "flex", gap: 34, fontFamily: fonts.sans, fontSize: 23, color: colors.textSoft}}>
      {items.map((it) => (
        <div key={it.name} style={{display: "flex", alignItems: "center", gap: 10}}>
          <div style={{width: 16, height: 16, borderRadius: 4, background: it.color}} />
          {it.name}
        </div>
      ))}
    </div>
  </Fade>
);

export const Caption: React.FC<{at: number; text: string}> = ({at, text}) => (
  <Fade at={at}>
    <div style={{
      fontFamily: fonts.sans, fontSize: 27, color: colors.textSoft, maxWidth: 1500,
      lineHeight: 1.5,
    }}>
      {text}
    </div>
  </Fade>
);
