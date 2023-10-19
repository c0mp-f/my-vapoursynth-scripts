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

from util import merge_chroma

core = vs.core

stronger_deband_ranges = [
    (None, 124)
]

# muxtools and filterchain setup
setup = mux.Setup("00")

JP = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[BDMV][220223][Lupin Sansei Part 6][BD-BOX1]/LUPIN_PART6_1/BDMV/STREAM/00000.m2ts", True)
src = vst.initialize_clip(JP.src)

sloc = {
    0.0: 2.00,
    0.3: 4.00,
    0.6: 7.00,
    1.0: 15.0
}

preden = vsd.DFTTest(src, sloc=sloc).denoise(src)

detail_mask = vdb.deband_detail_mask(preden, brz=(0.01, 0.014))

linemask = vsmasktools.KirschTCanny.edgemask(preden).std.Binarize(20<<8).std.Maximum().std.Minimum().std.Inflate()

# AA
ss = Waifu2x.scale(src, 2880, 1620)

eedi = vsaa.Eedi3(alpha=0.16, beta=0.35, gamma=60, vthresh0=12, vthresh1=24, field=1, sclip_aa=None)
aa = eedi.interpolate(ss, False, mclip=Point.scale(linemask, ss.width, ss.height), sclip=ss).std.Transpose()
aa = eedi.interpolate(aa, False, mclip=Point.scale(linemask, ss.width, ss.height).std.Transpose(), sclip=ss.std.Transpose()).std.Transpose()
aa = Catrom.scale(aa, 1920, 1080)
aa = vst.join(aa, src)

aa = core.std.MaskedMerge(src, aa, linemask)

# denoise
zones = [
    [(None, 503), 20],
    [(2175, 2252), 3]
]
deblock = vsd.dpir(aa, zones=zones)
deblock = core.std.MaskedMerge(deblock, aa, vsd.dpir_mask(aa, low=7, high=15, luma_scaling=8).resize.Point(format=vs.YUV420P16))
deblock = vst.replace_ranges(aa, deblock, [(None, 503), (2175, 2252)])

pre = vsd.prefilter_to_full_range(preden, 0)

cden = vsd.ccd(deblock, thr=3, mode=CCDMode.BICUBIC_CHROMA)
mv = MVTools(cden, tr=3, **MVToolsPresets.NOISY).denoise(cden, thSAD=(150, 65))

denoise = vsd.BM3DCudaRTC(deblock, sigma=[1.2, 0], tr=2, profile=Profile.NORMAL).final(deblock, mv)
denoise = vst.join(denoise, mv)

tmp = core.ttmpsm.TTempSmooth(denoise, maxr=1, thresh=1, mdiff=0, strength=1, pfclip=pre)

# deband
regular_deband = vdb.F3kdb(thr=112).deband(tmp)

stronger_deband = vdb.Placebo(radius=24, thr=3.5, iterations=4).deband(tmp)

deband = vst.replace_ranges(regular_deband, stronger_deband, stronger_deband_ranges)

deband = core.std.MaskedMerge(deband, tmp, detail_mask)

# utterly demolishing the 30th order ringing on the credits
# if the bg weren't completely black this'd be ugly
dering = vsd.Prefilter.BILATERAL(deband, sigmaS=5, sigmaR=0.15) 
dering = vst.replace_ranges(deband, dering, (30442, 32483))

grain = vdb.AddNoise.FBM_SIMPLEX.grain(dering, strength=(2.1, 1.4), size=1.35, dynamic=True, luma_scaling=10, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.2, 0.7), size=1.15, dynamic=False, luma_scaling=12, seed=2)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13.4, 
        aq_strength=0.73, psy_rd=2.05, psy_rdoq=2.15, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow",
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    v = mux.x265(settings).encode(final)

    jp_a = mux.FFMpeg.Extractor(track=0).extract_audio(JPN.file)
    jp_a = mux.Opus(bitrate=192).encode_audio(jp_a)

    # obtained from another blu-ray
    eng_a = mux.AudioFile(f"/mnt/HDD1/lupin/part_6/subs/ep{setup.episode}/{setup.episode}_dub.opus") 

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
    #o(core.lsmas.LWLibavSource("testenc"))

    o(src)
    o(aa)
    o(denoise)
    o(tmp)
    #o(dering)
    o(regular_deband)

    o(detail_mask)
    o(linemask)
    #o(smooth_dering(denoise, **dering_args, show_mask=True))

    from lvsfunc import stack_planes as sp

    o(sp(src))
    o(sp(denoise))