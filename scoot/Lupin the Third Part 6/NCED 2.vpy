import vapoursynth as vs 
import vsmuxtools as mux
import vsdenoise as vsd
import vsdeband as vdb
import vstools as vst
import vsmasktools
import vsaa

from vsdenoise import MVToolsPresets, MVTools, Profile, CCDMode, Prefilter
from vskernels import Catrom, ZewiaCubic, Point
from vsscale import Waifu2x

from util import get_text_mask, texture_mask, merge_chroma

core = vs.core

#### SETUP
setup = mux.Setup("NCED 2")

NCED = mux.src_file(r"D:/Videos/Anime/TV/[BDMV][220622][Lupin Sansei Part 6][BD-BOX2]/LUPIN_PART6_7/BDMV/STREAM/00004.m2ts", True)
src = NCED.init()

#### MASKS
sloc = {
    0.0: 2.00,
    0.3: 4.00,
    0.6: 7.00,
    1.0: 15.0
}

preden = vsd.DFTTest(src, sloc=sloc).denoise(src)

detail_mask = vdb.deband_detail_mask(preden, brz=(0.009, 0.0095))

linemask = vsmasktools.Sobel.edgemask(preden).std.Binarize(20<<8).std.Maximum().std.Minimum().std.Inflate()

#### AA
ss = Waifu2x(tiles=30).scale(src, 2880, 1620)

eedi = vsaa.Eedi3(alpha=0.17, beta=0.35, gamma=60, vthresh0=12, vthresh1=24, field=1, sclip_aa=None)
aa = eedi.interpolate(ss, False, mclip=Point.scale(linemask, ss.width, ss.height), sclip=ss).std.Transpose()
aa = eedi.interpolate(aa, False, mclip=Point.scale(linemask, ss.width, ss.height).std.Transpose(), sclip=ss.std.Transpose()).std.Transpose()

aa = Catrom.scale(aa, 1920, 1080)
aa = vst.join(aa, src)
aa = core.std.MaskedMerge(src, aa, linemask)

#### DENOISE
pre = vsd.prefilter_to_full_range(preden, 1.25)

cden = vsd.ccd(aa, thr=3, mode=CCDMode.BICUBIC_CHROMA)
mv = MVTools(cden, tr=3, **MVToolsPresets.NOISY).denoise(cden, thSAD=(160, 65))

denoise = vsd.BM3DCudaRTC(aa, sigma=[1.6, 0], tr=2, profile=Profile.NORMAL).final(aa, mv)
denoise = vst.join(denoise, mv)

tmp = core.ttmpsm.TTempSmooth(denoise, maxr=1, thresh=1, mdiff=0, strength=1, pfclip=pre)

#### DEBAND
deband = vdb.Placebo(radius=24, thr=2.5, iterations=4).deband(tmp)

deband = core.std.MaskedMerge(deband, tmp, detail_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(2.2, 1.3), size=1.25, dynamic=True, luma_scaling=10, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.4, 0.7), size=1.20, dynamic=False, luma_scaling=12, seed=2)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13.5, 
        aq_strength=0.74, psy_rd=2.05, psy_rdoq=2.15, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow",
        deblock=[-1, -2], bframes=16,
        rect=False
    )
    v = mux.VideoFile("./_workdir/NCED_2/encoded.265")#mux.x265(settings).encode(final)

    jp_a = mux.do_audio(NCED)

    lyrics = mux.SubTrack(
        fr"D:/lupin/part_6/subs/ep16/NCED.ass", "[Commie] Song lyrics", # 30767 - 30449
        default=True, forced=False
    )

    font = mux.FontFile("./subs/ep16/attachments/arabtype.ttf")

    mux.mux(
        v, jp_a.to_track("Japanese", "ja", default=True), lyrics, font,
        tmdb=mux.TmdbConfig(31572, season=6, write_ids=True, write_title=True, write_date=True)
    )


else: 
    from vstools import set_output as o 
    final.set_output()
    
    o(src)
    #o(ss)
    o(aa)
    o(mv)
    o(denoise)
    o(tmp)
    #o(dering)

    o(deband)
    o(weaker_deband)
    o(stronger_deband)

    o(detail_mask)
    o(texture)
    o(linemask)
    o(diff)

    from lvsfunc import stack_compare as sp

    #o(sp(aa))
    #o(sp(cden))
    #o(sp(denoise))
    o(JP.src)
    o(someBD)
    o(CR)