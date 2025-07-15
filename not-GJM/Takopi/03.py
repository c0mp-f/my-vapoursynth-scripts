import vapoursynth as vs
import vsmuxtools as mux

from vstools import initialize_clip, depth, finalize_clip, replace_ranges, ChromaLocation, set_output as o
from vsdenoise import mc_degrain, deblock_qed, DFTTest, wnnm, bm3d
from vsdeband import placebo_deband, AddNoise, deband_detail_mask, f3k_deband
from vsexprtools import norm_expr
from vsdehalo import fine_dehalo
from vsaa import based_aa, EEDI3

core = vs.core

lazy = [(17733, 18017), (21109, 21238)]

setup = mux.Setup("03")

CR = mux.src_file("/home/c0mp/Downloads/Takopis.Original.Sin.S01E03.Takopis.Confession.1080p.CR.WEB-DL.AAC2.0.H.264-VARYG.mkv", idx=core.bs.VideoSource)
cr = initialize_clip(CR.src)[120:]

AMZN = mux.src_file("/home/c0mp/Downloads/Takopis.Original.Sin.S01E03.Takopis.Confession.1080p.AMZN.WEB-DL.DDP2.0.H.264-VARYG.mkv", idx=core.bs.VideoSource)
amzn_ft = initialize_clip(AMZN.src, 32)
amzn_ft = norm_expr(amzn_ft, 'x 0.5 255 / +', planes=[1, 2])
amzn = depth(amzn_ft, 16)

merged = core.std.Merge(amzn, cr)

preden = DFTTest(merged, sloc=((0.0, 0), (1, 16))).denoise(merged)
dm = deband_detail_mask(preden, brz=(0.008, 0.035))

deblock = deblock_qed(merged, quant_edge=16, quant_inner=18)
deblock = core.std.MaskedMerge(deblock, merged, dm)

mv = mc_degrain(deblock, thsad=90)
bm = bm3d(deblock, ref=mv, sigma=0.75, radius=1)

aa = based_aa(bm, mask_thr=55, alpha=0.135)

dh = fine_dehalo(aa, rx=2, highsens=40, thmi=42)

deband = replace_ranges(placebo_deband(dh, thr=2, radius=16), f3k_deband(dh, thr=84), lazy)
deband = core.std.MaskedMerge(deband, dh, dm)
grain = AddNoise.GAUSS(strength=0.25, size=1.15).grain(deband)

final = finalize_clip(grain)
#final = final[514:710] + final[900:1105] + final[1629:1795] + final[2350:2694] + final[3850:3980] + final[11913:11999] + final[13530:13639] 

if __name__ == "__main__":
    from vsmuxtools import settings_builder_x265
    settings = settings_builder_x265(
        crf=13.5, aq_strength=0.73, psy_rd=1.95, psy_rdoq=1.8, chroma_qpoffsets=-3,
        qcomp=0.7, rd=3, preset="veryslow",
        deblock=[-1, -1], bframes=16,
        rect=False
    )
    from vsmuxtools import x265, Chapters, SubFile, VideoFile, FFMpeg, Opus, AudioFile, do_audio

    enc = x265(settings).encode(final)

    audio = do_audio(AMZN)

    mux.mux(enc, audio)

else:
    o(final)

    o(merged)
    o(deblock)
    o(mv)
    o(bm)
    o(aa)
    o(dh)
    o(deband)

    o(CR.src[120:])
    o(AMZN.src)

    o(dm)

    o(fine_dehalo(aa, rx=2, highsens=40, thmi=42, show_mask=True))

    #o(core.std.Merge(amzn, cr))
