"""SAMBA-UNet (arXiv 2610.40029) explainer video: narration for all 8 chapters. A Manim scene that only speaks; make_chapter.sh turns LINES into the
user's cloned voice and writes assets/um/timeline_<slug>[.warm].json. Visuals: Motion Canvas (mc/src/scenes/sb_*.tsx) or Manim (sb_*.py).
Every number below comes from the paper text or from notebooks/paper-sambaunet/out/smoke_test.txt (our own run); nothing is invented.
Build voice + MC chapter:  ./make_mc_chapter.sh sb_story StoryVoice        Build a Manim chapter:  ./make_chapter.sh sb_figure.py FigureChapter sb_figure clone warm"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from manim import *  # noqa: F403
from video_speech import to_markup  # noqa: F401
from vidbase import NarratedScene

config.background_color = "#181818"

# Tool per chapter (brainstormed per scene, user rule): MC = diagrams/cards/code panels; Manim = real-image zoom tours and the scan maths.
STORY = {  # 1 · MC  — why this paper exists (cards)
    "S1": "Medical segmentation needs two things at once: the **big picture**, and the exact **edge**.",
    "S2": "A U-Net sees edges, but only **locally**. A transformer sees everything, but its cost grows with the **square** of the number of pixels.",
    "S3": "Mamba reads the image as a **sequence**, so its cost grows only **linearly**. But it knows nothing about **objects**.",
    "S4": "SAM knows objects, because it has seen a huge number of photos. But skin and CT scans look **nothing** like those photos.",
    "S5": "The new paper, SAMBA-UNet, runs **both**, and lets SAM guide Mamba.",
    "S6": "On the Synapse CT dataset it reports a Dice of **81.94**, against **81.08** for VM-UNet.",
    "S7": "That is a gain of under one point, so we will look at it **carefully**.",
}
FIGURE = {  # 2 · Manim — zoom tour of the paper's real Figure 1(a)
    "F1": "This is the paper's own **Figure one**. We will walk through it piece by piece.",
    "F2": "The same skin image goes in at the **top**, and a **mask** of the lesion comes out at the top right.",
    "F3": "On the left is the **Mamba** branch. Four stages of V S S blocks, with patch merging between them, so the picture gets smaller and the channels grow.",
    "F4": "In the middle is the **SAM** encoder. The snowflake means frozen. The flames on the adapters mean they are **trained**.",
    "F5": "SAM is tapped at blocks **six**, **nine**, and **twelve**.",
    "F6": "Each tap meets a Mamba skip feature in a small module. The figure calls it **C P E M**, and the text calls it H C E M.",
    "F7": "On the right, the Mamba decoder climbs back up, and takes the **guided** features at each level.",
    "F8": "SAM's own mask decoder is drawn too, but the paper only says it **may** be added to the final output.",
}
SAM = {  # 3 · MC — SAM encoder, properly
    "A1": "Now SAM, properly. Its **encoder** is a plain vision transformer, B size: twelve blocks, and seven hundred sixty-eight channels.",
    "A2": "A sixteen pixel patch becomes one **token**. In our run, a two hundred fifty-six pixel image gives a sixteen by sixteen grid of tokens.",
    "A3": "Most blocks look inside small **windows**. Blocks three, six, nine and twelve look at the **whole** image.",
    "A4": "The paper taps six, nine and twelve. So it takes features from **global** blocks.",
    "A5": "Normally SAM adds a **prompt** encoder and a mask decoder. SAMBA-UNet drops the prompts, and keeps only the **encoder**.",
    "A6": "Inside every block the paper adds **two** things: low-rank updates on the query and the value, and two small bottleneck adapters.",
    "A7": "The rest stays **frozen**: about ninety million weights. The trainable extras are two and a half million, in our implementation.",
}
MAMBA = {  # 4 · Manim — selective scan on the real lesion grid
    "M1": "Now Mamba. A state space model reads a **sequence** one step at a time, and keeps a small **memory**.",
    "M2": "At each step the memory fades a little, and the new input is **added**. One step per pixel, so the cost grows **linearly**.",
    "M3": "In Mamba, the fade and the input gate depend on the **content**. That is the **selective** part.",
    "M4": "But an image is not a line. So V Mamba flattens the grid in **four** orders: rows, columns, and both **reversed**.",
    "M5": "Every pixel is then read from four sides, and the four results are **merged**.",
}
ARCH = {  # 5 · MC — the new architecture, redrawn with real shapes
    "N1": "Here is the whole network, redrawn. The image goes into **both** encoders at the same time.",
    "N2": "Mamba makes three skip maps: **sixty-four**, thirty-two, and sixteen pixels wide.",
    "N3": "SAM gives three maps, all **sixteen** by sixteen.",
    "N4": "H C E M upsamples the SAM map to the size of the skip, and **concatenates** them.",
    "N5": "A gate looks at the average and the maximum of every channel, and turns the **volume** of each channel up or down.",
    "N6": "The result is concatenated with the skip once more, and goes into the **decoder**.",
    "N7": "That happens three times, from coarse to fine, and then one last layer gives the **mask**.",
}
JOURNEY = {  # 6 · Manim — what happens to one real image
    "J1": "Let us follow one **real** image. This is the lesion printed in the paper.",
    "J2": "SAM sees it as a sixteen by sixteen grid. Here are its features at block **six**, **nine** and **twelve**, squeezed into three colours.",
    "J3": "Already at block six the lesion stands apart from the skin. By block twelve it is a clear **blob**.",
    "J4": "The Mamba side works at sixty-four, thirty-two and sixteen pixels. In our run those weights are **random**, so we only show the shapes.",
    "J5": "After thirty training steps on this single image, our model draws the mask. That is **memorising**, not a result.",
}
IMPL = {  # 7 · MC — how a data scientist implements it (code panels, real output)
    "I1": "If you want to build this, start with SAM from **transformers**, and freeze it.",
    "I2": "Add low-rank updates and adapters that start at **zero**. Then the model must equal plain SAM. Our check gave a difference of exactly **zero**.",
    "I3": "Hugging Face insists on a **thousand and twenty-four** pixel input. Call the patch convolution directly, and resize the position embedding.",
    "I4": "Take the features from blocks **six**, nine and twelve.",
    "I5": "Build the Mamba U-Net. Use the official CUDA scan and V Mamba weights if you can. Our PyTorch scan matched the slow one to about **one e minus six**.",
    "I6": "Check the gradients. The Laura down matrices get **zero** gradient at the first step. That is normal, because the up matrices start at zero.",
    "I7": "Then overfit one batch before anything else. We reached a Dice of **point nine nine three** on one image. That proves the wiring, not the accuracy.",
}
HONEST = {  # 8 · MC — what the paper does not say
    "H1": "Now the honest part. The code is **not** released. The paper says, upon acceptance.",
    "H2": "The ablation baseline scores **seventy-nine**, which is below V M U Net. And there is no row for H C E M alone.",
    "H3": "There are no seeds and no error bars, and the gains are **under one point**.",
    "H4": "The abstract claims **efficiency**, but the paper prints no parameter count, no flops, and no speed.",
    "H5": "So this is a promising **recipe**. It is not yet a **proof**.",
}


class _Voice(NarratedScene, Scene):
    def construct(self):
        for k in self.LINES:
            self.beat(k, wait=True)
        self.wait(0.8)
        self.save_timeline()


class StoryVoice(_Voice):
    SLUG, LINES = "sb_story", STORY
class FigureVoice(_Voice):
    SLUG, LINES = "sb_figure", FIGURE
class SamVoice(_Voice):
    SLUG, LINES = "sb_sam", SAM
class MambaVoice(_Voice):
    SLUG, LINES = "sb_mamba", MAMBA
class ArchVoice(_Voice):
    SLUG, LINES = "sb_arch", ARCH
class JourneyVoice(_Voice):
    SLUG, LINES = "sb_journey", JOURNEY
class ImplVoice(_Voice):
    SLUG, LINES = "sb_impl", IMPL
class HonestVoice(_Voice):
    SLUG, LINES = "sb_honest", HONEST
