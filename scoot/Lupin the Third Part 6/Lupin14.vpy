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

OP = (0, 1917)
opstart = OP[0]

ED = (30448, 32485)

stronger_deband_ranges = [
    0
]

weaker_deband_ranges = [
    0
]

aa_ranges = [
    (opstart+1794, opstart+1825)
]

dpir_ranges = []

regrain_ranges = [
    (opstart+1333, opstart+1415), (opstart+1794, opstart+1841), (opstart+1794, OP[-1])
]

#### SETUP
setup = mux.Setup("14")

# JP seems like the best source, but there seem to be VBV issues. 
# It's especially prevalent in the OP, where it's so badly starved that it ghosts. 
#
JP = mux.src_file(r"D:/Videos/Anime/TV/[BDMV][220622][Lupin Sansei Part 6][BD-BOX2]/LUPIN_PART6_5/BDMV/STREAM/00003.m2ts", True)
src = vst.initialize_clip(JP.src)

# CR credits are also masked in because all BDs have multiple orders of ringing on text while CR doesn't. 
# In addition, CR still somehow looks better in some places, so it gets spliced in where necessary
#
CR = mux.src_file(r"D:/[LostYears] Lupin III S06 (WEB 1080p x264 AAC E-AC-3) [Dual-Audio]/[LostYears] Lupin III - S06E14 (WEB 1080p x264 AAC E-AC-3) [05FBC5A3].mkv", True)
CR = vst.initialize_clip(CR.src)
CR = core.std.Expr(CR, "x 128 +")

# The other BD has a tint and is generally more starved, but performs better in some scenes where the JPBD has the aforementioned issues.
#
SOME_BD = mux.src_file(r"D:/lupin/Lupin the 3rd Part 6 - 14 - The Mirage Women.mkv", True, trim=(24, None))
someBD = core.std.Expr(SOME_BD.init_cut(), "x 128 +")
someBD = vst.replace_ranges(someBD, core.std.MaskedMerge(someBD, CR, get_text_mask(CR, thr=0.08).std.Maximum()), OP)

average_all = core.std.Expr([src, CR, someBD], "x y z + + 3 /")
average_BDs = core.std.Expr([src, someBD], "x y + 2 /")
average_web = core.std.Expr([CR, someBD], "x y + 2 /")
 
# for shitty fucking credit ringing
NCED = core.lsmas.LWLibavSource(r"D:/Videos/Anime/TV/[BDMV][220622][Lupin Sansei Part 6][BD-BOX2]/LUPIN_PART6_7/BDMV/STREAM/00004.m2ts")
NCED = vst.initialize_clip(NCED)
NCED = core.std.BlankClip(src)[:ED[0]] + NCED + core.std.BlankClip(src)[ED[1]:]

diff = core.std.Expr([get_text_mask(src, 0.027), get_text_mask(CR, 0.027)], "x y max")
diff = vst.iterate(diff, core.std.Maximum, 5)

credit_fix = vst.replace_ranges(CR, Prefilter.BILATERAL(src, sigmaR=0.03), (ED[0]+1554, ED[0]+1673))
credit_fix = vdb.AddNoise.SIMPLEX.grain(credit_fix, strength=(7, 0), size=1.5, seed=69)
credit_fix = core.std.MaskedMerge(src, credit_fix, diff)

src = vst.replace_ranges(src, credit_fix, ED)

# This mess of code simply splices in better sources whenever they're necessary, including but not limited to the OP
# The averages were literally the only thing that made it look good in motion.
src = vst.replace_ranges(src, someBD, OP)
src = vst.replace_ranges(src, average_all, [(opstart+399, opstart+470), (opstart+1842, opstart+1916)])
src = vst.replace_ranges(src, average_web, [(opstart+471, opstart+479), (opstart+492, opstart+506), (opstart+1842, OP[-1])])
src = vst.replace_ranges(src, CR, [(opstart+938, opstart+954), (opstart+1289, opstart+1332), (opstart+1794, opstart+1841)])

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

texture = texture_mask(preden, radius=2, points=[x * 256 for x in (0.7, 1.2, 3.8, 15)]).resize.Point(format=vs.YUV420P16)
texture = core.akarin.Expr([texture, linemask.std.Maximum()], f"y 1.25 / round x max")
texture = vst.replace_ranges(texture, core.std.BlankClip(texture), [OP, ED])

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

#### DENOISE
zones = [
    [[(opstart+471, opstart+479)], 60],
    [[(opstart+321, opstart+370)], 50],
    [[(opstart+1248, opstart+1288)], 25],
    [[(opstart, opstart+149), (ED[0]+1554, ED[0]+1673)], 15],
    [[(opstart+399, opstart+470)], 6],
    [[(opstart+1794, opstart+1841)], 3]
]

for [*ranges], strength in zones:

    dpir_ranges += ranges

    # only regrain strongly deblocked scenes, and only those in the OP.
    if len(ranges) == 1 and strength > 10:
        regrain_ranges += ranges

    elif strength > 10:
        regrain_ranges += [range for range in ranges if range[-1] <= OP[-1]]

deblock = vsd.dpir(aa, strength=0, zones=zones)
deblock = core.std.MaskedMerge(deblock, aa, texture)

deblock = core.std.MaskedMerge(aa, deblock, diff)

deblock = vst.replace_ranges(aa, deblock, dpir_ranges)

pre = vsd.prefilter_to_full_range(preden, 1.25)

cden = vsd.ccd(deblock, thr=3, mode=CCDMode.BICUBIC_CHROMA)
mv = MVTools(cden, tr=3, **MVToolsPresets.NOISY).denoise(cden, thSAD=(160, 65))

denoise = vsd.BM3DCudaRTC(deblock, sigma=[1.3, 0], tr=2, profile=Profile.NORMAL).final(deblock, mv)
denoise = vst.join(denoise, mv)

tmp = core.ttmpsm.TTempSmooth(denoise, maxr=1, thresh=1, mdiff=0, strength=1, pfclip=pre)

#### DEBAND
regular_deband = vdb.F3kdb(radius=24, thr=98).deband(tmp)

stronger_deband = vdb.Placebo(radius=24, thr=3, iterations=4).deband(tmp)

#weaker_deband = vdb.Placebo(radius=24, thr=1.8, iterations=4).deband(dering)
weaker_deband = vdb.F3kdb(radius=20, thr=64).deband(tmp)

OPED_deband = vdb.Placebo(radius=24, thr=2.5, iterations=4).deband(tmp)

deband = vst.replace_ranges(regular_deband, OPED_deband, [OP, ED])
deband = vst.replace_ranges(deband, weaker_deband, weaker_deband_ranges)
deband = vst.replace_ranges(deband, stronger_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, tmp, detail_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(2.2, 1.3), size=1.25, dynamic=True, luma_scaling=10, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.4, 0.7), size=1.20, dynamic=False, luma_scaling=12, seed=2)

grain = vst.replace_ranges(grain, vdb.AddNoise.GAUSS.grain(deband, strength=(4.5, 0), size=0.75, luma_scaling=5, seed=69), regrain_ranges)

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
    v = mux.VideoFile(fr"D:/lupin/part_6/_workdir/14/encoded.265")#mux.x265(settings).encode(final)

    jp_a = mux.do_audio(JP)
    
    eng_a = mux.do_audio(SOME_BD)

    dialogue = mux.SubTrack(
        fr"D:/lupin/part_6/subs/ep{setup.episode}/track3.eng.ass", "[Commie] Dialogue",
        default=True, forced=False
    )
    signs_and_songs = mux.SubTrack(
        fr"D:/lupin/part_6/subs/ep{setup.episode}/Signs and Songs.ass", "[Commie] Signs and Songs",
        default=False, forced=True
    )

    fonts_1 = mux.GlobSearch(f"./subs/ep{setup.episode}/attachments/*[Tt][Tt][Ff]", True)
    fonts_2 = mux.GlobSearch(f"./subs/ep{setup.episode}/attachments/*[Oo][Tt][Ff]", True)

    fonts = [mux.FontFile(font) for font in fonts_1.paths] + [mux.FontFile(font) for font in fonts_2.paths]

    chapters = mux.Chapters(f"./subs/ep{setup.episode}/chapters.xml")

    mux.mux(
        v, 
        jp_a.to_track("Japanese", "ja", default=True), eng_a.to_track("English", "en"),
        dialogue, signs_and_songs, chapters, *fonts,
        tmdb=mux.TmdbConfig(31572, season=6, write_ids=True, write_title=True, write_date=True), print_cli=True
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