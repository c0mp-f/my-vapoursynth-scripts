import vapoursynth as vs 
import vsmuxtools as mux
import vsdenoise as vsd
import vsdeband as vdb
import vstools as vst
import vsmasktools
import vsaa

from vsdenoise import MVToolsPresets, MVTools, Profile, CCDMode
from vskernels import Catrom, ZewiaCubic, Point
from vsscale import Waifu2x

from util import get_text_mask, texture_mask

core = vs.core

OP = (3043, 4961)
opstart = OP[0]

stronger_deband_ranges = [
    (9990, 10127)
]

weaker_deband_ranges = [
    (0, 1515)
]

aa_ranges = [
    (opstart+1794, opstart+1825), (11091, 11136)
]

dpir_ranges = []

regrain_ranges = [
    (opstart+1333, opstart+1415), (opstart+1794, opstart+1841), (opstart+1794, OP[-1])
]

#### SETUP
setup = mux.Setup("01")

# JP seems like the best source, but there seem to be VBV issues. 
# It's especially prevalent in the OP, where it's so badly starved that it ghosts. 
#
JP = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[BDMV][220223][Lupin Sansei Part 6][BD-BOX1]/LUPIN_PART6_1/BDMV/STREAM/00001.m2ts", True)
src = vst.initialize_clip(JP.src)

# CR credits are also masked in because all BDs have multiple orders of ringing on text while CR doesn't. 
# In addition, CR still somehow looks better in some places, so it gets spliced in where necessary
#
CR = mux.src_file("/mnt/HDD1/[LostYears] Lupin III S06 (WEB 1080p x264 AAC E-AC-3) [Dual-Audio]/[LostYears] Lupin III - S06E01v2 (WEB 1080p x264 AAC E-AC-3) [0BFA0A4E].mkv", True)
CR = vst.initialize_clip(CR.src)
CR = core.std.Expr(CR, "x 0.5 128 / +")

# The other BD has a tint and is generally more starved, but performs better in some scenes where the JPBD has the aforementioned issues.
#
someBD = mux.src_file("/mnt/HDD1/lupin/Lupin the 3rd Part 6 - 01 - Enter Sherlock Holmes.mkv", True, trim=(24, None)).init_cut()
someBD = core.std.Expr(someBD, "x 0.5 128 / +")
someBD = core.std.MaskedMerge(someBD, CR, get_text_mask(CR, thr=0.08).std.Maximum())

average_all = core.std.Expr([src, CR, someBD], "x y z + + 3 /")
average_web = core.std.Expr([CR, someBD], "x y + 2 /")

# This mess of code simply splices in better sources whenever they're necessary, including but not limited to the OP
# The averages were literally the only thing that made it look good in motion.
src = vst.replace_ranges(src, someBD, OP)
src = vst.replace_ranges(src, average_all, [(opstart+399, opstart+470), (opstart+1842, opstart+1915)])
src = vst.replace_ranges(src, average_web, [(opstart+471, opstart+479), (opstart+492, opstart+506), (opstart+1842, OP[-1])])
src = vst.replace_ranges(src, CR, [(opstart+938, opstart+954), (opstart+1289, opstart+1332), (opstart+1794, opstart+1841)])

#### MASKS
preden = vsd.DFTTest(src, sigma=5).denoise()

detail_mask = vdb.deband_detail_mask(preden, brz=(0.02, 0.02))
detail_mask = core.akarin.Expr(
    [detail_mask, texture_mask(preden, radius=3, points=[x * 256 for x in (0.85, 1.5, 4, 12)])], 
    "x y max"
)

linemask = vsmasktools.Sobel.edgemask(preden).std.Binarize(35<<8).std.Maximum().std.Minimum().std.Inflate()

texture = texture_mask(preden, radius=4, points=[x * 256 for x in (0.4, 1, 3, 15)]).resize.Point(format=vs.YUV420P16)
texture = core.akarin.Expr([texture, linemask], "y 1.25 / round x max")
texture = vst.replace_ranges(texture, core.std.BlankClip(texture), OP)

#### AA
ss = Waifu2x.scale(src, 2880, 1620)

eedi = vsaa.Eedi3(alpha=0.175, beta=0.35, gamma=90, vthresh0=12, vthresh1=24, field=1, sclip_aa=None)
aa = eedi.interpolate(ss, False, mclip=Point.scale(linemask, 2880, 1620), sclip=ss).std.Transpose()
aa = eedi.interpolate(aa, False, mclip=Point.scale(linemask, 2880, 1620).std.Transpose(), sclip=ss.std.Transpose()).std.Transpose()
aa = Catrom.scale(aa, 1920, 1080)

sraa = vsaa.upscaled_sraa(src, rfactor=1.6)

aa = vst.replace_ranges(aa, sraa, aa_ranges)
aa = core.std.MaskedMerge(src, aa, linemask)

#### DENOISE
zones = [
    [[(opstart+471, opstart+479)], 60],
    [[(opstart+321, opstart+370)], 50],
    [[(opstart+1247, opstart+1288)], 25],
    [[(opstart, opstart+149), (opstart+399, opstart+470)], 6],
    [stronger_deband_ranges, 5],
    [[(opstart+1794, opstart+1841)], 3]
]

for ranges, strength in zones:
    
    dpir_ranges += ranges

    # only regrain strongly deblocked scenes, and only those in the OP.
    if strength > 10:
        regrain_ranges += ranges

deblock = vsd.dpir(aa, zones=zones)
deblock = core.std.MaskedMerge(deblock, aa, vsd.dpir_mask(aa, low=7, high=15, luma_scaling=8).resize.Point(format=vs.YUV420P16))
deblock = vst.replace_ranges(aa, deblock, dpir_ranges)

sloc = {
    0.00: 1,
    0.15: 2,
    0.45: 2,
    0.60: 0,
    1.00: 0
}

dft = vsd.DFTTest(deblock, sloc=sloc, tr=0).denoise(deblock, planes=0)
dft = core.std.MaskedMerge(dft, deblock, texture)

cden = vsd.ccd(dft, thr=3.5, mode=CCDMode.BICUBIC_CHROMA)
mv = MVTools(cden, tr=2, **MVToolsPresets.NOISY).denoise(cden, thSAD=(185, 60))

denoise = vsd.BM3DCudaRTC(mv, sigma=[1.1, 0], tr=2).final(mv, deblock)
denoise = vst.join(denoise, mv)

#### DEBAND
regular_deband = vdb.F3kdb(thr=100).deband(denoise)

stronger_deband = vdb.Placebo(radius=24, thr=4, iterations=4).deband(denoise)

#weaker_deband = vdb.Placebo(radius=24, thr=1.8, iterations=4).deband(dering)
weaker_deband = vdb.F3kdb(thr=86).deband(denoise)

OP_deband = vdb.Placebo(radius=24, thr=2.5, iterations=4).deband(denoise)

deband = vst.replace_ranges(regular_deband, OP_deband, OP)
deband = vst.replace_ranges(regular_deband, stronger_deband, stronger_deband_ranges)
deband = vst.replace_ranges(deband, weaker_deband, weaker_deband_ranges)

deband = core.std.MaskedMerge(deband, denoise, detail_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(2.0, 1.4), size=1.25, dynamic=True, luma_scaling=10, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.2, 1.4), size=1.15, dynamic=False, luma_scaling=12, seed=2)

grain = vst.replace_ranges(grain, vdb.AddNoise.GAUSS.grain(deband, strength=(4.5, 0.5), size=0.75, luma_scaling=5, seed=69), regrain_ranges)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13.5, 
        aq_strength=0.75, psy_rd=2.05, psy_rdoq=2.15, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow",
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    v = mux.x265(settings).encode(final)

    jp_a = mux.do_audio(JP)
    
    eng_a = mux.do_audio(SOME_BD)

    dialogue = mux.SubTrack(
        f"/mnt/HDD1/lupin/part_6/subs/ep{setup.episode}/track3.eng.ass", "[Commie] Dialogue",
        default=True, forced=False
    )
    signs_and_songs = mux.SubTrack(
        f"/mnt/HDD1/lupin/part_6/subs/ep{setup.episode}/Signs and Songs.ass", "[Commie] Signs and Songs",
        default=False, forced=True
    )

    fonts_1 = mux.GlobSearch(f"subs/ep{setup.episode}/attachments/*[Tt][Tt][Ff]", True)
    fonts_2 = mux.GlobSearch(f"subs/ep{setup.episode}/attachments/*[Oo][Tt][Ff]", True)

    fonts = [mux.FontFile(font) for font in fonts_1.paths] + [mux.FontFile(font) for font in fonts_2.paths]

    chapters = mux.Chapters(f"subs/ep{setup.episode}/chapters.xml")

    mux.mux(
        v, 
        jp_a.to_track("Japanese", "ja", default=True), eng_a.to_track("English", "en"),
        dialogue, signs_and_songs, chapters, *fonts,
        tmdb=mux.TmdbConfig(31572, season=6, write_ids=True, write_title=True, write_date=True)
    )


else: 
    from vstools import set_output as o 
    final.set_output()
    
    o(src)
    o(someBD)
    o(CR)
    o(dft)
    o(mv)
    o(denoise)
    #o(dering)

    o(regular_deband)
    o(weaker_deband)
    o(stronger_deband)

    o(detail_mask)
    o(texture)
    o(linemask)
    #o(smooth_dering(denoise, **dering_args, show_mask=True))

    from lvsfunc import stack_planes as sp

    o(sp(src))
    o(sp(CR))
    o(sp(denoise))