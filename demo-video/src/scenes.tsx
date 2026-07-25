import React from "react";
import {useCurrentFrame, interpolate} from "remotion";
import {Bars, Caption, Fade, Legend, Terminal, Title} from "./components";
import {colors, fonts, sec} from "./theme";
import data from "./generated-data.json";

const Frame: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{
    position: "absolute", inset: 0, background: colors.bg, padding: "84px 96px",
    display: "flex", flexDirection: "column", gap: 34,
  }}>
    {children}
  </div>
);

/** 1. The grep: Arm has no kernel for this format, Intel does. */
export const SceneGrep: React.FC = () => (
  <Frame>
    <Title title="IQ4_XS is how most MoE models ship." sub="This is what llama.cpp says about running them on Arm." />
    <Fade at={sec(3)}>
      <Terminal lines={[
        {text: "$ grep -c IQ4_XS ggml/src/ggml-cpu/repack.cpp    # Arm smmla kernels", at: sec(4), color: colors.textMuted},
        {text: "0", at: sec(9), color: colors.fast},
        {text: "", at: 0},
        {text: "$ grep -n IQ4_XS ggml/src/ggml-cpu/amx/common.h  # Intel AMX kernels", at: sec(13), color: colors.textMuted},
        {text: "114:        (type == GGML_TYPE_IQ4_XS);", at: sec(18), color: colors.stock},
      ]} />
    </Fade>
    <Caption at={sec(21)} text="Intel had a tiled integer path for this format. Arm had none." />
  </Frame>
);

/** 2. The hardware advertises the unit that never gets used. */
export const SceneSilicon: React.FC = () => (
  <Frame>
    <Title title="The instruction is right there." />
    <Fade at={sec(2)}>
      <Terminal lines={[
        {text: "$ lscpu | grep -E 'Model name|Features'", at: sec(2), color: colors.textMuted},
        {text: "Model name:  Neoverse-N2", at: sec(5), color: colors.textSoft},
        {text: "Features:    ... sve2 svei8mm i8mm bf16", at: sec(8), color: colors.textSoft, cps: 30},
      ]} />
    </Fade>
    <Fade at={sec(13)}>
      <div style={{
        fontFamily: fonts.sans, fontSize: 34, color: colors.text, background: colors.panelRaised,
        border: `1px solid ${colors.border}`, borderRadius: 14, padding: "24px 30px", maxWidth: 1400,
      }}>
        The core has an <strong style={{color: colors.fast}}>i8mm</strong> matrix unit.
        An IQ4_XS model never issues a single <code style={{fontFamily: fonts.mono}}>smmla</code>.
      </div>
    </Fade>
    <Caption at={sec(18)} text="Choosing the smallest 4-bit format to fit a cost-efficient instance was exactly the choice that gave back the throughput." />
  </Frame>
);

/** 3. Mechanism: toggling the fast path moves one format and not the other. */
export const SceneToggle: React.FC = () => {
  const t = data.toggle;
  const max = Math.max(t.q4k.on, t.q4k.off, t.iq4xs.on, t.iq4xs.off);
  return (
    <Frame>
      <Title title="Was it really the missing kernel?" sub="Same commit, same runner — only GGML_CPU_REPACK differs." />
      <div style={{marginTop: 14}}>
        <Bars
          at={sec(3)} max={max} height={330}
          groups={[
            {label: "Q4_K", sub: "pp512",
             values: [{v: t.q4k.off, color: colors.stock, name: "off"}, {v: t.q4k.on, color: colors.fast, name: "on"}],
             ratio: `${(t.q4k.on / t.q4k.off).toFixed(2)}x`},
            {label: "IQ4_XS", sub: "pp512",
             values: [{v: t.iq4xs.off, color: colors.stock, name: "off"}, {v: t.iq4xs.on, color: colors.fast, name: "on"}],
             ratio: `${(t.iq4xs.on / t.iq4xs.off).toFixed(2)}x`},
          ]}
        />
      </div>
      <Legend at={sec(4)} items={[{name: "fast path OFF", color: colors.stock}, {name: "fast path ON", color: colors.fast}]} />
      <Caption at={sec(17)} text="Disabling Arm's fast path costs Q4_K 35% of its prefill and costs IQ4_XS 0.6% — inside the noise. It was never on that path." />
    </Frame>
  );
};

/** 4. The kernel itself. */
export const SceneKernel: React.FC = () => {
  const f = useCurrentFrame();
  const tile = (i: number) => {
    const p = interpolate(f - sec(16) - i * 3, [0, 10], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
    return {opacity: 0.25 + 0.75 * p, transform: `scale(${0.9 + 0.1 * p})`};
  };
  return (
    <Frame>
      <Title title="The kernel was the intersection of two that already existed." />
      <Caption at={sec(3)} text="IQ4_XS is IQ4_NL's 16-entry codebook carrying K-quant super-block scales. Arm already had smmla kernels for each half separately — so this one belonged to neither effort." />
      <Fade at={sec(8)}>
        <Terminal width={1500} lines={[
          {text: "const int8x16_t w[4] = {                       // one column pair", at: sec(9), color: colors.textMuted},
          {text: "    vqtbl1q_s8(kvalues, vandq_u8 (raw0, m4b)),  // codebook lookup", at: sec(11), color: colors.textSoft},
          {text: "    vqtbl1q_s8(kvalues, vshrq_n_u8(raw1, 4)),", at: sec(13), color: colors.textSoft},
          {text: "};", at: sec(14), color: colors.textSoft},
          {text: "sb01 = vmmlaq_s32(sb01, w[t], q8_01[t]);       // 2x2 tile, no shuffle", at: sec(15), color: colors.fast},
        ]} />
      </Fade>
      <Fade at={sec(20)}>
        <div style={{display: "flex", gap: 18, alignItems: "center", fontFamily: fonts.sans}}>
          {[0, 1, 2, 3].map((i) => (
            <div key={i} style={{
              ...tile(i), width: 132, height: 92, borderRadius: 10, background: colors.panelRaised,
              border: `2px solid ${colors.fast}`, color: colors.textSoft, fontSize: 21,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              smmla
            </div>
          ))}
          <div style={{color: colors.textMuted, fontSize: 25, marginLeft: 16}}>
            16-byte weight load × 16-byte activation load → 2×2 int32 tile
          </div>
        </div>
      </Fade>
      <Caption at={sec(31)} text="Three kernels ship: smmla for Graviton3/4 and Cobalt, sdot for Graviton2 and Ampere Altra, and a dotprod GEMV so batch-1 decode is never traded away." />
    </Frame>
  );
};

/** 5. The measured result, with the control. */
export const SceneResult: React.FC = () => {
  const a = data.ab;
  const max = Math.max(a.iq4xs.pp512.fast, a.q4k.pp512.stock);
  const r = (a.iq4xs.pp512.fast / a.iq4xs.pp512.stock).toFixed(2);
  return (
    <Frame>
      <Title title={`Prefill: ${r}× faster, same GGUF file.`} sub="Both builds compiled in the same CI job on the same physical Neoverse N2 runner." />
      <div style={{marginTop: 10}}>
        <Bars
          at={sec(3)} max={max} height={310}
          groups={[
            {label: "IQ4_XS", sub: "pp512",
             values: [{v: a.iq4xs.pp512.stock, color: colors.stock, name: "stock"}, {v: a.iq4xs.pp512.fast, color: colors.fast, name: "fast"}],
             ratio: `${r}x`},
            {label: "IQ4_XS", sub: "pp2048",
             values: [{v: a.iq4xs.pp2048.stock, color: colors.stock, name: "stock"}, {v: a.iq4xs.pp2048.fast, color: colors.fast, name: "fast"}],
             ratio: `${(a.iq4xs.pp2048.fast / a.iq4xs.pp2048.stock).toFixed(2)}x`},
            {label: "Q4_K", sub: "control — untouched",
             values: [{v: a.q4k.pp512.stock, color: colors.stock, name: "stock"}, {v: a.q4k.pp512.fast, color: colors.fast, name: "fast"}],
             ratio: `${(a.q4k.pp512.fast / a.q4k.pp512.stock).toFixed(2)}x`},
          ]}
        />
      </div>
      <Legend at={sec(4)} items={[{name: "stock llama.cpp", color: colors.stock}, {name: "FastPath64", color: colors.fast}]} />
      <Caption at={sec(19)} text="The control is the point: Q4_K is untouched and holds at 1.00×, so the gain is the kernel and not the conditions. IQ4_XS went from 0.58× of Q4_K to 1.24× — while remaining the smaller file." />
    </Frame>
  );
};

/** 6. Anyone can re-run it. */
export const SceneReproduce: React.FC = () => (
  <Frame>
    <Title title="Every number re-runs from a button." />
    <Fade at={sec(2)}>
      <Terminal width={1500} lines={[
        {text: "# on GitHub's free Neoverse N2 runners - no account, no spend", at: sec(3), color: colors.textMuted},
        {text: "Actions  ->  A/B - stock vs FastPath64  ->  Run workflow", at: sec(5), color: colors.textSoft},
        {text: "", at: 0},
        {text: "# correctness on an x86 laptop, across both dispatch paths", at: sec(9), color: colors.textMuted},
        {text: "QEMU_CPU=neoverse-n1  ...  24/24 PASS", at: sec(11), color: colors.good},
      ]} />
    </Fade>
    <Caption at={sec(15)} text="Timings are gated behind a numerical equivalence check against the non-repacked path; the workflow refuses to report throughput if it fails." />
    <Fade at={sec(17)}>
      <div style={{fontFamily: fonts.mono, fontSize: 26, color: colors.textMuted}}>
        github.com/Marc-Dvci/fastpath64 · MIT
      </div>
    </Fade>
  </Frame>
);
