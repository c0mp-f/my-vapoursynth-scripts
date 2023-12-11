import vapoursynth as vs
import vsdenoise as vsd
import vsdehalo as vdh
import vsdeband as vdb
import vstools as vst
import vsaa

from vsmuxtools import Setup, x265, do_audio, settings_builder_x265, src_file, mux, Chapters
from vsdenoise import Prefilter, MVTools, MVToolsPresets, Profile
from vsdeband import AddNoise
from vsscale import Waifu2x

core = vs.core

OP = (2158, 4315)
opstart = OP[0]
opend = OP[1]

sraa_ranges = [
    (opstart+1321, opstart+1419)
]

stronger_deband_ranges = [
    (opstart+1321, opstart+1449), (13791, 13826)
]

setup = Setup("09")

### SOURCES
CR = src_file(r"D:\Videos\Anime\TV\[SubsPlease] Spy x Family - 34 (1080p) [86289122].mkv", False)
src = CR.init_cut()

AMZN = src_file(r"D:\Videos\Anime\TV\Spy x Family S2 - 09 (Amazon Prime CBR 1080p).mkv", False)
amzn = AMZN.init_cut()

DSNP = src_file(r"D:\Videos\Anime\TV\Spy x Family S2 - 09 (Disney+ 1080p).mkv", False, (24, -24))
dsnp = DSNP.init_cut()

src = vst.replace_ranges(vsd.frequency_merge(src, dsnp), src, 0)

y = vst.get_y(src)
y = Prefilter.DFTTEST(y)

linemask = core.std.Sobel(y).std.Binarize(25<<8).std.Maximum()

detail_mask = vdb.deband_detail_mask(y, brz=(0.02, 0.02))

pre = vsd.prefilter_to_full_range(src, 4)
pre = Prefilter.DFTTEST(pre)

## AA + DEHALO
aa = vsaa.based_aa(src, rfactor=1.7, alpha=0.14, supersampler=Waifu2x(tiles=5, num_streams=2))

sraa = vsaa.upscaled_sraa(src, rfactor=1.8)

aa = vst.replace_ranges(aa, sraa, sraa_ranges)

dehalo = vdh.dehalo_sigma(aa, sigma=2.2, pre_ss=2)

dehalo_m = vdh.fine_dehalo.mask(pre, thlimi=60, thmi=60)

dehalo = core.std.MaskedMerge(aa, dehalo, dehalo_m)

## DENOISE
zones = [
    [0, 0]
]
#deblock = vsd.dpir(dehalo, strength=0, zones=zones)
#deblock = vst.replace_ranges(dehalo, deblock, 0)

deblock = vsd.ccd(dehalo, thr=4)

mv = vsd.MVTools(deblock, tr=2, **MVToolsPresets.NOISY).denoise(deblock, thSAD=(145, 70))
denoise = vsd.BM3DCudaRTC(deblock, tr=2, sigma=[0.85, 0], profile=Profile.NORMAL).final(deblock, mv)

tmp = core.ttmpsm.TTempSmooth(denoise, maxr=1, thresh=1, mdiff=0, strength=1, pfclip=pre)

## DEBAND
from functools import partial
deband = vdb.pfdeband(tmp, radius=24, thr=1.4, prefilter=Prefilter.GAUSS, debander=partial(vdb.Placebo, iterations=16))

stronger_deband = vdb.Placebo(radius=24, thr=3, iterations=16).deband(tmp)

deband = vst.replace_ranges(deband, stronger_deband, stronger_deband_ranges)
deband = core.std.MaskedMerge(deband, tmp, detail_mask)

grain = AddNoise.FBM_SIMPLEX.grain(deband, strength=(2.0, 1.6), size=1.25, dynamic=True, seed=1)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = settings_builder_x265(
        crf=13, 
        aq_strength=0.7, psy_rd=2.05, psy_rdoq=2.05, chroma_qpoffsets=-3,
        qcomp=0.73, rd=3,
        preset="veryslow",
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    enc = x265(settings).encode(final)

    audio = do_audio(AMZN, track=0).to_track("Japanese 2.0 EAC3")

    #chapters = Chapters([(0, "Introduction"), (opstart, "Opening"), (opend+1, "Part A"), (15257, "Part B"), (31649, "Ending"), (33807, "Preview")])

    mux(enc, audio)#, chapters)

else:
    from vstools import set_output as o 
    final.set_output()

    o(src)
    #o(core.lsmas.LWLibavSource(r"D:\spy\testenc.mkv"))
    o(aa)
    o(dehalo)
    o(mv)
    o(denoise)
    o(tmp)
    o(deband)

    o(linemask)
    o(detail_mask)
    o(dehalo_m)

    o(sraa)

    o(amzn)
    o(dsnp)
    o(CR.init())