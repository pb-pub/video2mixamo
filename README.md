# Video to Mixamo

Turn a video of a person into a ready-to-use motion capture animation. Point it at a
video file and it produces a **BVH** animation you can drop into Blender, Mixamo, Unity,
or any tool that reads BVH — no motion-capture suit required.

It uses [MediaPipe Pose](https://developers.google.com/mediapipe) to track the body in
each frame and converts that tracking into skeleton rotations.

## What you get

- 🎥 **Video → animation** in one command
- 🕺 **Webcam capture** if you'd rather record live
- 🎬 **Standard BVH output** that imports cleanly into Blender / Mixamo / Unity
- 🪄 **Built-in smoothing** so the result isn't jittery
- 👀 **Optional live preview** to watch the tracking as it runs

## Install

You'll need **Python 3.10 or newer**.

```bash
# 1. Get the code
git clone https://github.com/pb-pub/video2mixamo.git
cd video-to-maximo

# 2. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install
pip install -e .

# 4. Download the pose model (one-time, ~30 MB)
python -m scripts.download_models
```

> The model download step is required before the first run. You can choose
> `lite`, `full` (default), `heavy`, or `all` — e.g. `python -m scripts.download_models lite`.
> Bigger models are more accurate but slower.

## Quick start

Convert a video to an animation file:

```bash
python -m video_to_maximo.main --input myvideo.mp4
```

That's it. The whole clip is processed automatically and a `.bvh` file is written to the
`output/` folder (named with a timestamp). No windows to click, no keys to press.

Choose where the file goes:

```bash
python -m video_to_maximo.main --input myvideo.mp4 --output dance.bvh
```

Watch the pose tracking while it converts (press **ESC** to stop early):

```bash
python -m video_to_maximo.main --input myvideo.mp4 --preview
```

### Tips for a good result

- Keep the **whole body in frame** and reasonably well lit.
- Face the camera; side-on and heavily occluded poses track poorly.
- A steady camera helps — the person should move, not the camera.

## Recording from your webcam

Run it with no input file to record live. A preview window opens automatically:

```bash
python -m video_to_maximo.main
```

In the window:

| Key | Action |
|-----|--------|
| **R** | Start / stop recording (the clip is exported when you stop) |
| **S** | Stop recording |
| **V** | Toggle the 3D skeleton viewer |
| **ESC** | Quit without saving |

Use a different camera with `--camera 1` (etc.).

## Using the animation

The output is a plain `.bvh` file. To use it:

- **Blender**: *File → Import → Motion Capture (.bvh)*
- **Unity**: import the `.bvh` (via an FBX conversion or a BVH importer package)
- **Mixamo-style rigs**: the skeleton uses Mixamo-compatible bone names (Hips, Spine,
  LeftArm, RightUpLeg, …), so it retargets onto standard humanoid characters.

## Options

| Option | Description |
|--------|-------------|
| `--input FILE` | Video file to convert. Omit to use the webcam. |
| `--output FILE` | Where to save the animation (default: auto-named in `output/`). |
| `--preview` | Show the live tracking window while converting. |
| `--camera ID` | Which webcam to use (default: `0`). |
| `--fps RATE` | Override the frame rate (default: taken from the video, or 30). |
| `--model FILE` | Use a specific downloaded pose model. |
| `--no-smooth` | Turn off smoothing (rawer, jerkier output). |
| `--auto-download` | Download the model automatically if it's missing. |

See everything with:

```bash
python -m video_to_maximo.main --help
```

## Troubleshooting

- **"Model not found"** — run `python -m scripts.download_models` (or add `--auto-download`).
- **"No pose detected" / empty animation** — make sure the person is fully visible and
  well lit, and facing the camera.
- **Camera won't open** — try a different `--camera` number, or check that no other app is
  using the webcam.
- **The animation looks jittery** — that's normal for raw tracking; smoothing is on by
  default. Avoid `--no-smooth` unless you're post-processing yourself.

## License

MIT
