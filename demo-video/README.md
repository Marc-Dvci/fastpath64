# Demo video

A fully automated 2:45 demo, rendered from the repository's own measurements. Nothing in it is
hand-typed: `scripts/prepare-data.ts` reads the committed benchmark CSVs and emits
`src/generated-data.json`, which the scenes animate. Re-run the benchmarks and the video changes
the next time it renders.

```bash
npm install
npm run data      # rebuild generated-data.json from results/
npm run studio    # interactive preview
npm run render    # -> out/fastpath64-demo.mp4
```

`npm run render` runs `data` first, so a render cannot silently use stale numbers.

## Structure

| scene | seconds | what it shows |
|---|---|---|
| `grep` | 0–25 | The two greps: no IQ4_XS kernel on Arm, one on Intel AMX |
| `silicon` | 25–50 | `lscpu` reporting `i8mm` on the runner that never issues `smmla` |
| `toggle` | 50–75 | The repack toggle: Q4_K moves 1.55×, IQ4_XS 1.01× |
| `kernel` | 75–115 | The codebook lookup feeding `vmmlaq_s32`, 2×2 tiles |
| `result` | 115–145 | The A/B bars, with the untouched control |
| `reproduce` | 145–165 | Run-workflow button, 24/24 equivalence PASS |

Scene boundaries live in `src/theme.ts`; the composition duration is derived from them, so
changing a scene's length needs no other edit.

## Notes

There is no narration track. The scenes carry on-screen captions and are paced to be read, which
keeps the render deterministic and free of any third-party audio — a challenge requirement. If a
voiceover is wanted later, the caption strings in `src/scenes.tsx` are the script.

Colours are the dark steps of the same validated categorical pair the writeup figures use, so the
video and the document agree visually.
