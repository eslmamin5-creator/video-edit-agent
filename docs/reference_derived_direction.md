# Reference-derived visual and sound direction

This document records the *principles* the visual/sound director follows. They
were distilled from a reference edit by watching how it behaves, not by copying
it. Nothing here is a timestamp, a sound map, a graphic, a colour, a font, a
logo, an asset or a brand rule. Those belong to a brand file or to the reviewer.

The director is deliberately small:

```
Approved transcript -> semantic beats -> visual director -> sound intent
        -> MasterTimeline -> deterministic render -> QA
```

An AI (or the reviewer) may only *choose from a small approved vocabulary*.
Deterministic code executes the choice. Gates reject unsafe choices, and every
fallback is explicit and visible in the review.

## 1. One question per beat

> Is the speaker still the best visual for this beat?

- **Yes** -> a *speaker* treatment: the speaker stays, only the framing changes
  (`speaker_static`, `punch_in`, `punch_out`, `reframe`, `slow_push`, `hold`,
  `reset_to_base`). Doing nothing is a valid, frequent answer.
- **No** -> a *replacement*: another visual takes the screen while the voice
  continues (`real_broll`, `user_broll`, `motion_graphic`, `kinetic_typography`,
  `illustration`, `generated_visual`, `graphic_data_scene`,
  `full_screen_text_scene`).

Choose the simplest visual that carries the meaning. An abstract idea does not
need literal footage; forcing a literal picture onto it is worse than a clean
graphic or the speaker. Behind-subject text is one treatment among these, never
mandatory.

`SpeakerReplacement` is its own concept, distinct from B-roll: it says *the
speaker is hidden, the voice is not*. B-roll is only one of the visuals that can
do that.

## 2. Camera grammar

The camera is tied to semantic beats, not to a clock.

- Moves: `static`, `punch_in`, `punch_out`, `slow_push`, `reframe_left`,
  `reframe_right`, `reset_to_base`.
- Every move targets an **absolute** framing, so zoom never accumulates.
- After an emphasis, come back: a punch-out or reset follows a punch-in.
- Static is valid. There is no continuous fake motion.
- Face-safe framing is a hard limit; a move that would crop the face is dropped.
- Spacing and density limits stop a shot from being moved every few seconds. A
  dropped move is recorded in the beat's notes, and the beat stays on the
  speaker.

## 3. Speed

Speed is the last resort, never automatic. In priority order:

1. remove genuine dead air;
2. keep the speaker's natural cadence;
3. build rhythm from framing, cuts and replacements;
4. change speed only with an explicit, written justification.

A speed change is a first-class, reviewable timeline event. It is not rendered
unless approved. There is no default 1.1x or 1.15x.

## 4. Transitions

The default is a hard, invisible cut. A stylised transition (whip, zoom, blur,
flash, slide) is allowed only for one of: motion continuity, a semantic change,
spatial continuity, or a deliberate reveal, and never too close to the previous
one. Stylised transitions are not executable yet, so the direct cut plays and
the request is kept visible in the review.

## 5. Captions

The backing plate is a choice, not a default.

| mode       | look                                                                       |
| ---------- | -------------------------------------------------------------------------- |
| `none`     | stroked text, no plate; the default candidate for a minimal or premium look |
| `adaptive` | like `none`, plus a subtle backing only where measured contrast is too low  |
| `plate`    | the translucent brand plate                                                 |

Modes only change how a caption is drawn. Caption timing, chunking and karaoke
word timing are never touched. Under a strong primary treatment (a motion
graphic, a text scene, behind-subject text) captions stay readable but stop
competing with it: smaller, no second emphasis. They are never hidden, because
they are also the accessibility text.

## 6. Behind-subject text needs an editorial gate

The compositing path (hair-safe matte, subject-wins ordering, matte gate) says
the effect *can* be drawn cleanly. A separate editorial gate says whether it
*should* be: mask quality, readable occlusion, phrase timing, shot composition,
caption hierarchy and visual value. The gate is fail-closed: a check that was
not measured counts as unproven. If it does not pass, the director picks
another treatment.

## 7. Sound

```
visual/edit event -> sound intent -> sound profile -> small SFX registry
        -> deterministic asset selection -> MasterTimeline
```

- Intents: `none` (default), `subtle_motion`, `transition`, `accent`, `impact`.
- Profiles: `none`, `minimal` (low density), `dynamic` (medium). A brand only
  picks a profile; there is no per-brand sound library.
- The AI never browses sounds and never names a file. The resolver maps
  profile + intent + event type to an asset id with no randomness.
- Density control: a minimum spacing between sounds, suppression of repeated
  decorative sounds, no stacking, and an importance threshold per profile.
- A sound is aligned to a real visual event (the frame a punch-in starts, the
  moment a replacement begins); its timing lives in the MasterTimeline.
- Mix priority: voice > essential content audio > music > SFX.
- If the registry is empty or an asset is missing, the slot resolves to no
  sound. A render never depends on a sound effect.
- Every slot reports an honest status: `none`, `suppressed`, `unavailable` or
  `scheduled`.

## 8. Music

None. This layer adds no music.

## 9. Review-first

Each slot shows, in plain language: Visual, Speaker, Camera, Transition, Sound
intent, SFX availability, Caption behaviour, Status. Natural replies work
(`4 موافق`, `4 خليه speaker`, `4 من غير sound`, `4 sound accent بدل transition`,
`اعتمد الباقي`). The reviewer is never asked for a file name. Generated visuals
are never triggered without an explicit, per-slot approval, and nothing here
sets `ready_for_final_render`.
