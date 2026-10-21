"""Part 4: the Part 3 kernel plus show() and the inspection tool API. Subclass, so Part 3's file is untouched."""
import base64
from pathlib import Path

from kernel_session import CellResult, KernelSession

_TOOLS_SRC = (Path(__file__).parent / "kernel_tools.py").read_text()


class Session4(KernelSession):
    def __init__(self, image, timeout_sec: int = 60, with_show: bool = False, with_tools: bool = False):
        super().__init__(image, timeout_sec=timeout_sec)
        self.with_show = with_show
        self._seen = set()
        if with_show or with_tools:
            # one exec: the module source, then drop the names the condition does not offer
            drop = []
            if not with_show:
                drop.append("show")
            if not with_tools:
                drop += ["part_mask", "label_blobs", "overlay", "crop"]
            out, err, _, _ = self._exec(_TOOLS_SRC + "\n" + "".join(f"del {n}\n" for n in drop)
                                        + "_BASE |= set(globals())\n", 30)
            assert not err, err

    def run(self, code: str) -> CellResult:
        res = super().run(code)
        res.images = self._collect_images()
        return res

    def _collect_images(self, max_images: int = 2):
        """New PNGs written by show() since the last cell, as (title, data-URL). The extra ones are dropped, and said so."""
        shown = Path(self.workdir.name) / "shown"
        out = []
        for p in sorted(shown.glob("*.png")) if shown.exists() else []:
            if p.name in self._seen:
                continue
            self._seen.add(p.name)
            if len(out) < max_images:
                title = p.with_suffix(".txt").read_text() if p.with_suffix(".txt").exists() else ""
                out.append((title, "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()))
        return out
