---
name: paper-podcast
description: Use when the user asks for a podcast / NotebookLM-style audio discussion of a paper, repo or topic. Encodes the decisions already made - two Phonon preset voices (Harper + Trevor), the fixed episode structure incl. the mandatory "what is open source / can I use my data / fine-tune recipe / licence" segment, GPU rendering, and how to let the user listen over SSH.
---

# Paper podcast (user's setup, decided 2026-10-24 with the WeVisOne-VL episode)

Reference implementation: `notebooks/podcast-wevisone/` (`script.json`, `phonon_tts.py`, `build_phonon_episode.py`, `pyproject.toml`). For a new episode copy that folder to `notebooks/podcast-<slug>/`, replace `script.json`, and keep the code.

## Decisions (do not re-ask)
- **Voice engine = Phonon** (Gradium `phonon-40m-en`, catalog presets), because the user asked for it: different voice per host, fast. **Host A (explainer) = Harper, Host B (curious sceptic) = Trevor.** "Any two voices, whatever works" - keep this pair unless asked. Other US-English catalog voices: Easton, Micah, Russell, Preston, Miles (male); Everly, Juniper, Zoey, Delia, Hadley (female). List: `voices.jsonl` in the `gradium/phonon-voices` hub cache.
- Phonon CANNOT clone the user's voice (no speaker encoder, verified 2026-10-21). Only use Chatterbox/Qwen cloning (`video-tts-clone` pipeline, `~/.hermes/scripts/tts_batch_*.py`) if the user says "my own voice"; it is slow (minutes per line with verify-retry) and was abandoned for this request. "Phonon" is a TTS model, not "phone" - the first attempt misread this and wasted a clone run.
- Dialogue, not narration: two people, short turns (1-3 sentences), about 1,500-2,300 words (about 9-14 min). Plain spoken English, numbers written as words ("seventy one point three"), acronyms spaced ("G R P O", "I o U"), "Loh-rah" for LoRA, "V L L M" for vLLM. Avoid `=`, `<tags>`, `->`. Run text through `video_speech.spoken()` (STYLE=warm so no Kokoro phoneme markup). Whisper-transcribe the result (`~/.hermes/tts/.venv`, faster-whisper) and fix terms it hears wrongly.
- Every number comes from the paper's tables / the repo, read from the full text (`curl arxiv.org/pdf/<id>` + `pdftotext`), never the abstract. Label what is the hosts' own opinion ("this is my suggestion, not the authors'"). Include honest caveats: in-house benchmarks, authors ran the baselines, paper-stated limitations.
- Jargon gets a daily-life analogy in dialogue (host B restates with a simple example). No Mermaid / no visuals needed.

## Episode structure
1. Hook + one-sentence summary. 2. Problem the paper claims. 3. Method pieces, one per exchange. 4. Data / training. 5. Results with 4-6 headline numbers. 6. Host B's sceptic turn (caveats). 7. Limitations (paper's own). 8. **Practical segment (mandatory, user requirement):**
   - What is open-sourced: weights, code, data, method/training details, eval pipeline, extras (other branches!).
   - What is NOT: training data, training code, benchmarks. Say it plainly.
   - Can I use my own data with the model as-is (zero-shot)? What to install, VRAM/dtype, expected domain gap, and a minimal test: small labelled own set incl. negative/empty cases.
   - Is there a fine-tuning recipe? Official yes/no, and what the paper gives (hyper-parameters, losses, rewards). Then a clearly labelled own recommendation (steps, pitfalls, held-out set).
   - **Licence**: code AND weights, plus base model and third-party parts, commercial use yes/no, "not a lawyer".
   9. Takeaway for the user's work (X-ray / industrial inspection; e.g. "empty image -> None" matters) + sign-off.
   Verify segment 8 against the real release, not the paper: `gh api repos/<org>/<repo>` (licence field can say "Other/NOASSERTION" while LICENSE is Apache - read the file), README TODO checklist, `git ls-tree` on other branches (`git fetch --depth 1 origin <branch>`), `curl https://huggingface.co/api/models/<id>` (licence, safetensors dtype, files), `.../raw/main/LICENSE`, base-model card licence, evaluation README (which data is "not yet released").

## Render
```
cd notebooks/podcast-<slug>
uv sync                                   # pyproject: torch, transformers @ git main (Phonon class is NOT on PyPI), safetensors, soundfile
HF_HUB_OFFLINE=1 .venv/bin/python build_phonon_episode.py Harper Trevor <name>   # -> out/<name>.mp3 + .wav
```
- Models are already in `~/.hermes/cache/huggingface-hub` (set by `phonon_tts.py`); offline works. Gated 40m needs the token in `~/.hermes/cache/huggingface/token` only if it must be re-downloaded.
- **GPU**: the model is moved to CUDA (`PHONON_CPU=1` forces CPU). CPU rendering took 11.5 min for 9 min of audio; GPU took 75 s for 13 min.
- Per-line cache in `assets/phonon_cache/` (keyed by voice + spoken text): after editing the script only changed lines are re-synthesised (a re-render was 4 s).
- Never call `snapshot_download("gradium/phonon-voices")` (pulls 2,247 sample files, hit HTTP 429); use `hf_hub_download(..., "voices.jsonl")` and single voice files.
- Gap between turns 0.35 s; each line normalised to about -22 dBFS RMS.

## Letting the user listen (they work over SSH)
They cannot open files on the machine. Serve the `out/` folder on localhost and give the tunnel: `cd out && nohup python3 -m http.server 8765 --bind 127.0.0.1 &`, then tell them `ssh -L 8765:127.0.0.1:8765 <user>@<this-machine>` and `http://localhost:8765/<file>.mp3` (or `scp <user>@<this-machine>:<path> .`). Stop the server when they are done. Do not publish audio anywhere, post it, or put it on quarto.pub without explicit confirmation.

## Traps
- `pkill -f <pattern>` inside a Bash call can match and kill the calling shell (exit 144); kill by PID.
- Do not start a long clone job and a Phonon job on the same box without checking `free -h` (earlyoom, shared vLLM).
- Keep `.venv/` and `assets/` out of git (already in `.gitignore`).
