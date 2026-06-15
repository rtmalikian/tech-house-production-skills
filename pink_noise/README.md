# Pink Noise Calibration File

Generate a 5 minute pink-noise WAV for audio calibration and synth stem gain staging:

```sh
./generate_pink_noise.sh
```

Default output:

```text
pink_noise_5min_-20LUFS.wav
```

The default target is `-20 LUFS`, which is conservative for gain staging and leaves practical room for EQ boosts, saturation, compression, sends, and bus processing. If you want the common studio calibration reference, generate `-18 LUFS` instead:

```sh
./generate_pink_noise.sh -18 300 pink_noise_5min_-18LUFS.wav
```

Recommended use:

1. Play the pink-noise file through the same output path used for your synth.
2. Set your monitoring/interface chain so this reference is comfortable and repeatable.
3. Record synth stems so their average level sits around the same perceived loudness as the reference, not so their peaks approach 0 dBFS.
4. Keep individual recorded stems roughly around `-20` to `-18 LUFS` / about `-18 dBFS RMS`, with peaks usually no hotter than about `-10` to `-6 dBFS`.
5. Do final loudness later on the mix or master bus, not while capturing raw synth stems.

For headroom, lower is better than hotter at the recording stage. A 24-bit WAV recorded around `-20 LUFS` still has plenty of resolution, and the extra 8-12 dB of working room helps prevent FX chains and EQ boosts from clipping.

