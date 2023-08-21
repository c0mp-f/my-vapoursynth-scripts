import vapoursynth as vs

from pathlib import Path

core = vs.core 

filepath = "./" # thanks encode runner :)

subpath = Path("./subs")

sub_tracks = sorted(subpath.glob("*Lupin the Third*.ass"))
sign_tracks = sorted(subpath.glob("*Signs and Songs*.ass"))

def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    """Merges chroma from ref with luma.
    Args:
        luma (vs.VideoNode): Source luma clip.
        ref (vs.VideoNode): Source chroma clip.
    Returns:
        vs.VideoNode:
    """
    return core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)
