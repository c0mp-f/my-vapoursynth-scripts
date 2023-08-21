import vapoursynth as vs
import vsrgtools as vsrg
import vsmuxtools as mux
import vodesfunc as vof
import vsdenoise as vsd
import awsmfunc as awf
import vsdeband as vdb
import lvsfunc as lvf
import vsscale as vsc
import vstools as vst
import vsmasktools
import vsaa

from vsaa import upscaled_sraa, transpose_aa, clamp_aa, Eedi3
from vsdenoise import MVToolsPresets, Prefilter, Profile
from vstools import depth, get_w, get_y, join, split
from vskernels import Catrom, ZewiaCubic
from lvsfunc import stack_planes as sp
from vodesfunc import set_output as o
from functools import partial
from rekt import rektlvls

from filter import merge_chroma, filepath, sub_tracks, sign_tracks

core = vs.core 

ED = (29969, 32005)
edstart = ED[0]
no_den = [(edstart, edstart+359)]

strongest_deband_ranges = [
    0
]

stronger_deband_ranges = [
    (2157, 4148)
]

weaker_deband_ranges = [
    0
]

aa_ranges = [
    (edstart+1436, edstart+1468)
]

# muxtools/source setup
setup = mux.Setup("15")

usa = mux.src_file(f"{filepath}/Lupin.Sansei.Part 5.1080p.USA.BLURAY.AVC.DTS-HD.MA.2.0-ARiN/LUPIN III PART V DISC02/BDMV/STREAM/00005.m2ts", True)
usa = vst.initialize_clip(usa.src, 16)

JPN = mux.src_file(f"{filepath}/[ルパン三世][2018][TV][ルパン三世 PART5][BDMV][JPN][Vol.01-Vol.05 Fin]/LUPIN_PART5_3/BDMV/STREAM/00008.m2ts", True)
jpn = vst.initialize_clip(JPN.src, 16)

merge = vsd.frequency_merge(usa, jpn)

# dirtylinering
rekt = rektlvls(
    merge, 
    colnum=[0, 1, 2, 1917, 1918, 1919], colval=[11, 2, -3, -4, 3, 12],
    rownum=[0, 1079], rowval=[12, 12]
)
bbmod = awf.bbmod(rekt, left=1, right=2, bottom=1, top=1)

prefilter = vsd.DFTTest(sigma=10).denoise(bbmod)

detail_mask = vsmasktools.detail_mask_neo(prefilter, detail_brz=0.02, lines_brz=0.01)

# AA (scenefiltered)
linemask = vsmasktools.edge.ExSobel().edgemask(get_y(prefilter)).std.Binarize(110<<8).std.Inflate()

sraa = vsaa.upscaled_sraa(bbmod, rfactor=2)

aa = core.std.MaskedMerge(bbmod, sraa, linemask)
aa = vst.rfs(bbmod, aa, aa_ranges)

# chroma repair and also all-plane deblocking, when necessary
dpir_zones = [
    [no_den, 0],
    #[[9965, ], 45]
]

dpir = vsd.dpir(aa, strength=40, zones=dpir_zones)
crepair = merge_chroma(aa, dpir)

# denoise
SCOOTDE = MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)

ref = vsd.MVTools(crepair, tr=2, **SCOOTDE).degrain(thSAD=(125, 65))
denoise = vsd.BM3DCudaRTC(crepair, ref=ref, sigma=[1.6, 0], radius=2, profile=Profile.NORMAL, matrix=1).final(crepair, ref)

denoise = vst.rfs(denoise, crepair, no_den)

# deband
deband = vdb.Placebo(radius=24, thr=[2, 2.5], grain=0, iterations=4).deband(denoise)

stronger_deband = vdb.Placebo(radius=32, thr=3, grain=0, iterations=4).deband(denoise)

strongest_deband = vdb.Placebo(radius=32, thr=4, grain=0, iterations=4).deband(denoise)

weaker_deband = vdb.Placebo(radius=24, thr=1.5, grain=0, iterations=4).deband(denoise)

ed_deband = vdb.F3kdb(radius=31, thr=[192, 180], grain=0).deband(denoise)

deband = vst.rfs(deband, strongest_deband, strongest_deband_ranges)
deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = vst.rfs(deband, weaker_deband, weaker_deband_ranges)
deband = vst.rfs(deband, ed_deband, [ED])

deband = core.std.MaskedMerge(deband, denoise, detail_mask)

grain = vdb.AddNoise(strength=(2.1, 1.8), size=1.35, type=3, luma_scaling=8, dynamic=True, seed=1).grain(deband)
grain = vdb.AddNoise(strength=(0.7, 1.4), size=1.15, type=3, luma_scaling=12, dynamic=False, seed=2).grain(grain)

final = vst.finalize_clip(grain, 10)

#final = final[4120:4164] + final[4567:4646] + final[5064:5122] + final[5530:5603] + final[6010:6070] + final[13330:13502] + final[14592:14711] + final[15240:15383] + final[16950:16978] + final[23186:23289]

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13, 
        aq_strength=0.7, psy_rd=1.95, psy_rdoq=2.1, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow", 
        deblock=[-2, -2], bframes=10,
        rect=False
    )

    v = mux.x265(settings).encode(final)

    jp_a = mux.FFMpeg.Extractor(track=0).extract_audio(JPN.file)
    jp_a = mux.Opus(bitrate=192).encode_audio(jp_a)

    us_a = mux.FFMpeg.Extractor(track=0).extract_audio(USA.file)
    us_a = mux.Opus(bitrate=192).encode_audio(us_a)

    subs_dialogue = mux.SubTrack(sub_tracks[int(setup.episode) - 1], "[Commie] Dialogue", "en", default=True, forced=False)
    subs_signs = mux.SubTrack(sign_tracks[int(setup.episode) - 1], "[Commie] Signs and Songs", default=False, forced=True)

    fonts_1 = mux.GlobSearch(f"subs/ep{setup.episode}/*[Tt][Tt][Ff]", True)
    fonts_2 = mux.GlobSearch(f"subs/ep{setup.episode}/*[Oo][Tt][Ff]", True)

    fonts = [mux.FontFile(font) for font in fonts_1.paths] + [mux.FontFile(font) for font in fonts_2.paths]

    chapters = mux.Chapters(f"subs/ep{setup.episode}/{setup.episode}.xml")
    
    mux.mux(
        v, 
        jp_a.to_track("Japanese", "ja", default=True), us_a.to_track("English", "en"),
        subs_dialogue, subs_signs, chapters, *fonts,
        tmdb=mux.TmdbConfig(31572, season=5, write_date=True, write_ids=True, write_summary=True, write_title=True, write_synposis=True)
    )
else:
    final.set_output()
    #o(core.lsmas.LWLibavSource("testenc.mkv"))

    o(usa)
    o(bbmod)

    o(denoise)
    o(deband)

    o(detail_mask)

    o(sp(bbmod))
    o(sp(denoise))

    o(linemask)