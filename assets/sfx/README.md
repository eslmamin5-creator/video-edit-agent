# SFX pack (intentionally empty)

This directory is the home of the tiny curated sound-effect pack described by
`video_edit_agent.sound.registry`. **It ships with no audio files.** No sound
effect has been downloaded, generated or bundled, so the production behaviour
today is: every sound intent resolves to *no sound*, and a render never depends
on one.

## What goes here (when a legally safe pack exists)

- `registry.json`: one entry per asset (`id`, `category`, `file`, `energy`,
  `duration_s`, `style`, `allowed_intents`, `recommended_events`, `license`).
- the audio files those entries name (`.wav`), a few dozen at most.

Categories: `soft_whoosh`, `fast_whoosh`, `reverse_soft`, `transition_sweep`,
`impact_soft`, `text_tick_soft`.

## Rules

- One shared pack. Sounds are not organised per brand; the profile
  (`minimal` / `dynamic` / `none`) decides how much of the pack is used.
- Only add a file whose licence allows redistribution, and record it in the
  entry's `license` field. Do not add anything of unknown origin.
- The AI never browses this directory and never names a file. It only proposes
  a sound *intent*; `sound.resolver` picks the asset deterministically.
- Tests use synthetic fixtures generated at test time, never files from here.
- An optional `brands/<brand>/sfx/` drop-in with the same layout is honoured
  for future use; there is no manager for it.
