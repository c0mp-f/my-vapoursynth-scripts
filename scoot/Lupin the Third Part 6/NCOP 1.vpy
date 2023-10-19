import vapoursynth as vs 
import vsmuxtools as mux
import vsdenoise as vsd
import vsdehalo as vdh
import vsdeband as vdb
import vstools as vst
import vsmasktools
import vsaa

from vsdenoise import MVToolsPresets, MVTools, Profile, CCDMode, Prefilter
from vskernels import Catrom, ZewiaCubic, Point
from vsscale import Waifu2x

from util import get_text_mask, texture_mask, merge_chroma, nnedi

core = vs.core

opstart = 24

stronger_deband_ranges = [
    0
]

weaker_deband_ranges = [
    0
]

aa_ranges = [
    (opstart+1794, opstart+1825), (925, 961)
]

dpir_ranges = []

regrain_ranges = [
    (opstart+1333, opstart+1415), (opstart+1794, opstart+1841), (opstart+1794, 1943)
]

#### SETUP
setup = mux.Setup("NCOP 1")

JP = mux.src_file(r"D:\Videos\Anime\TV\[BDMV][220223][Lupin Sansei Part 6][BD-BOX1]\LUPIN_PART6_2\BDMV\STREAM\00001.m2ts", True)
src = vst.initialize_clip(JP.src)

SOME_BD = mux.src_file(r"D:\lupin\Lupin the 3rd Part 6 - 06 - The Imperial City Dreams of Thieves, Part 2.mkv", True)
someBD = core.std.Expr(SOME_BD.init_cut(), "x 128 +")

src = vst.replace_ranges(src, someBD, [(495, 558), 348, (1148, 1356), (1819, 1878)])

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

sraa = vsaa.upscaled_sraa(src, rfactor=1.5)

aa = vst.replace_ranges(aa, sraa, aa_ranges)
aa = core.std.MaskedMerge(src, aa, linemask)

#### DEHALO
ss = nnedi(aa)

dehalo = vdh.dehalo_sigma(ss, sigma=2.2)
dehalo = Point.scale(dehalo, 1920, 1080)
dehalo = vst.join(dehalo, aa)

dehalo = core.std.MaskedMerge(aa, dehalo, vdh.fine_dehalo(preden, thmi=50, show_mask=True))

#### DENOISE
zones = [
    [[(495, 503)], 60],
    [[(opstart+321, opstart+370), (1879, 1943)], 50],
    [[(opstart+1248, opstart+1288), (962, 978)], 25],
    [[(opstart, opstart+149), (559, 579), (1818, 1878)], 15],
    [(395, 396), 10],
    [[(opstart+399, opstart+470)], 6]
]

for [*ranges], strength in zones:

    dpir_ranges += ranges

    if strength > 10:
        regrain_ranges += ranges

deblock = vsd.dpir(dehalo, strength=0, zones=zones)

deblock = vst.replace_ranges(dehalo, deblock, dpir_ranges)

pre = vsd.prefilter_to_full_range(preden, 1.25)

cden = vsd.ccd(deblock, thr=3, mode=CCDMode.BICUBIC_CHROMA)
mv = MVTools(cden, tr=3, **MVToolsPresets.NOISY).denoise(cden, thSAD=(195, 65))

denoise = vsd.BM3DCudaRTC(deblock, sigma=[1.8, 0], tr=2, profile=Profile.NORMAL).final(deblock, mv)
denoise = vst.join(denoise, mv)

tmp = core.ttmpsm.TTempSmooth(denoise, maxr=1, thresh=1, mdiff=0, strength=1, pfclip=pre)

#### DEBAND
deband = vdb.Placebo(radius=24, thr=2.5, iterations=4).deband(tmp)

deband = core.std.MaskedMerge(deband, tmp, detail_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(3, 1.2), size=1.4, seed=1)
regrain = vdb.AddNoise.GAUSS.grain(deband, strength=(2.1, 0.5), size=1.5, seed=69)

grain = vst.replace_ranges(grain, regrain, regrain_ranges)

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
    v = mux.x265(settings).encode(final)

    jp_a = mux.do_audio(JP)
    
    mux.mux(
        v, jp_a.to_track("Japanese", "ja", default=True),
        tmdb=mux.TmdbConfig(31572, season=6, write_ids=True, write_title=True, write_date=True)
    )


else: 
    from vstools import set_output as o 

    
    o(src)

    o(someBD)