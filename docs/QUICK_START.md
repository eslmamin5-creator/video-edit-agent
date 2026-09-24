# Quick Start

Give it a video, look at a few short previews, say what you would change, approve, get an MP4.

```bash
videoedit edit my_video.mp4
```

The first run analyses the video (speech, captions, camera) and **stops before rendering**. It shows the main
treatments, each with a short preview. Nothing is exported until you approve.

## Pick a style (optional)

| Style | What you get |
|---|---|
| `--profile minimal` | Clean captions, very little camera movement. |
| `--profile balanced` (default) | Captions plus subtle camera movement and, when a key phrase is clearly worth it, one headline. |
| `--profile dynamic` | More frequent camera movement. |

Add `--brand <name>` to use your brand. Without it a clean neutral look is used. Your answers are remembered for
this video, so you are not asked again when you come back.

## How review works

Each treatment is one card: a preview, a friendly name, a short reason, and its state.
You can **Approve**, **Change** or **Remove** it.

```bash
videoedit review my_video.mp4        # see the cards again
```

## Ask for a change, in your own words

```bash
videoedit revise my_video.mp4 "خفف الزوم"
```

Examples: `كبر الهيدر` · `صغره` · `طلعه فوق شوية` · `نزله` · `خفف الزوم` · `زود الزوم` · `خلي النص أطول` ·
`شيل الحركة دي` · `خليه Behind-Subject` · `رجع المتحدث للمنتصف` · `ارجع للنسخة اللي قبلها`.

A new preview is made. The old approval no longer counts, so the changed treatment waits for your OK.
Your wording is never translated or formalised.

## Approve and lock

```bash
videoedit approve my_video.mp4
```

Approving locks each treatment to **exactly what its preview showed**. Run `videoedit edit my_video.mp4` again and
the final video uses those locked treatments. If a locked treatment cannot be reproduced, the run stops with a clear
error instead of quietly rendering something different. Changing something later simply asks for a new approval.

`videoedit undo my_video.mp4` goes back to the version before your last change.

## Export

The final MP4 is a standard H.264/AAC file that plays in Windows Media Player, phones and social apps. It is checked
automatically after rendering. No cloud service is required (`--offline` keeps everything on this computer).

If you run the same command again on an unchanged video, the analysis is reused and your approved choices are kept.
