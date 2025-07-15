import vsmuxtools as mux

from vstools import vs, core, initialize_clip, depth, finalize_clip, replace_ranges, set_output as o
from vsdenoise import mc_degrain, bm3d, deblock_qed, DFTTest
from vsdeband import placebo_deband, AddNoise, deband_detail_mask
from vsdehalo import fine_dehalo
from vsexprtools import norm_expr
from vsaa import based_aa

lazy = [(13168, 19194), (19260, 23754)]

setup = mux.Setup("01")

CR = mux.src_file("/home/c0mp/Downloads/Takopis.Original.Sin.S01E01.To.You.in.2016.1080p.CR.WEB-DL.AAC2.0.H.264-VARYG.mkv", idx=core.bs.VideoSource)
cr = initialize_clip(CR.src)[:240]

AMZN = mux.src_file("/home/c0mp/Downloads/Takopis.Original.Sin.S01E01.Episode.1.1080p.AMZN.WEB-DL.JPN.DDP2.0.H.264.ESub-ToonsHub.mkv", idx=core.bs.VideoSource)
amzn_ft = initialize_clip(AMZN.src, 32)
amzn_ft = norm_expr(amzn_ft, 'x 0.5 255 / +', planes=[1, 2])
amzn = depth(amzn_ft, 16)[:240]

#merged = core.std.Merge(amzn, cr)
merged = amzn

preden = DFTTest(merged, sloc=((0.0, 0), (1, 16))).denoise(merged)
dm = deband_detail_mask(preden, brz=(0.008, 0.035))

deblock = deblock_qed(merged, quant_edge=20, quant_inner=18)
#deblock = DFTTest(merged, sloc=((0, 0), (0.25, 2), (0.5, 2), (1, 6))).denoise(merged)
deblock = core.std.MaskedMerge(deblock, merged, dm)
deblock = replace_ranges(deblock, merged, lazy)

mv = mc_degrain(deblock, thsad=90)
bm = bm3d(deblock, ref=mv, sigma=0.7, radius=1)

aa = based_aa(bm, mask_thr=55, alpha=0.15, beta=0.225)

dh = fine_dehalo(aa, rx=2, highsens=40, thmi=42)

deband = placebo_deband(dh, radius=16, thr=1.8)
deband = core.std.MaskedMerge(deband, dh, dm)
#grain = AddNoise.FBM_SIMPLEX(strength=3, size=1.3).grain(deband)
grain = AddNoise.GAUSS(strength=(0.25, 0), size=1.2, luma_scaling=14).grain(deband)
#grain = AddNoise.SIMPLEX(strength=(2, 0), size=1.35).grain(deband)

final = finalize_clip(grain)
#final = final[514:710] + final[900:1105] + final[1629:1795] + final[2350:2694] + final[3850:3980] + final[11913:11999] + final[13530:13639] 

if __name__ == "__main__":
    from vsmuxtools import settings_builder_x265
    settings = settings_builder_x265(
        crf=13.5, aq_strength=0.72, psy_rd=1.95, psy_rdoq=1.8, chroma_qpoffsets=-3,
        qcomp=0.7, rd=3, preset="veryslow",
        deblock=[-1, -1], bframes=16,
        rect=False
    )
    from vsmuxtools import x265, Chapters, SubFile, VideoFile, FFMpeg, Opus, AudioFile, do_audio

    enc = x265(settings).encode(final)

    #audio = do_audio(AMZN)

    #mux.mux(enc, audio)

else:
    o(final)

    o(merged)
    o(deblock)
    o(mv)
    o(bm)
    o(aa)
    o(dh)
    o(deband)

    o(cr)
    o(amzn)

    #o(core.std.Merge(amzn, cr))
