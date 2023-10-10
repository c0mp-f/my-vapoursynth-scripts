import vapoursynth as vs
import vsdenoise as vsd
import vsdehalo as vdh
import vsdeband as vdb
import vstools as vst
import vsaa

from vsmuxtools import Setup, x265, do_audio, settings_builder_x265, src_file, mux, Chapters
from vsdenoise import Prefilter, MVTools, MVToolsPresets, Profile
from vskernels import Point, Catrom, NoScale
from vsdeband import AddNoise
from vsscale import Waifu2x
from filter import nnedi

core = vs.core

OP = (1606, 3764)
opstart = OP[0]
opend = OP[1]

sraa_ranges = [
    (opstart+1321, opstart+1419)
]

stronger_deband_ranges = [
    (opstart+1321, opstart+1449), (265, 385)
]

setup = Setup("01")

### SOURCES
CR = src_file(r"D:\Videos\Anime\TV\[SubsPlease] Spy x Family - 26 (1080p) [2A971129].mkv", True)
src = CR.init_cut()

AMZN = src_file(r"D:\Videos\Anime\TV\Spy x Family S2 - 01 (Amazon Prime CBR 1080p).mkv", True)
amzn = AMZN.init_cut()

DSNP = src_file(r"D:\Videos\Anime\TV\Spy x Family S2 - 01 (Disney+ 1080p).mkv", True, (24, None))
dsnp = DSNP.init_cut()

src = vst.replace_ranges(src, dsnp, [(OP[0], OP[0]+242), (4701, 5004), (6886, 7281), (10396, 10610), (20498, 20737)])
src = vst.replace_ranges(src, amzn, (10611, 10773))

y = vst.get_y(src)
y = Prefilter.DFTTEST(y)

linemask = core.std.Sobel(y).std.Binarize(25<<8).std.Maximum()

detail_mask = vdb.deband_detail_mask(y, brz=(0.0185, 0.0185))

## AA + DEHALO
aa = vsaa.based_aa(src, rfactor=1.7, alpha=0.14, supersampler=Waifu2x(tiles=15, num_streams=2))

sraa = vsaa.upscaled_sraa(src, rfactor=1.8)

aa = vst.replace_ranges(aa, sraa, sraa_ranges)

ss = nnedi(aa)
dehalo = vdh.dehalo_sigma(ss, sigma=2)
dehalo = Point.scale(dehalo, 1920, 1080)
dehalo = vst.join(dehalo, src)

dehalo_m = vdh.fine_dehalo(src, thlimi=60, thmi=60, show_mask=True)

dehalo = core.std.MaskedMerge(aa, dehalo, dehalo_m)

## DENOISE
zones = [
    [[(265, 322), (10396, 10653)], 10],
    [(323, 385), 3]
]
deblock = vsd.dpir(dehalo, strength=0, zones=zones)
deblock = vst.replace_ranges(dehalo, deblock, [(265, 385), (10396, 10773)])

cden = vsd.ccd(deblock, thr=2)
mv = vsd.MVTools(cden, tr=2, **MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(140, 65))
denoise = vsd.BM3DCudaRTC(deblock, tr=2, sigma=[1.4, 0], profile=Profile.NORMAL).final(deblock, mv)
denoise = vst.join(denoise, mv)

pre = vsd.prefilter_to_full_range(src, 1.25)
pre = Prefilter.DFTTEST(pre)

tmp = core.ttmpsm.TTempSmooth(denoise, maxr=1, thresh=1, mdiff=0, strength=1, pfclip=pre)
denoise = vst.replace_ranges(denoise, tmp, [(265, 385), (16628, 17099)])

## DEBAND
deband = vdb.Placebo(radius=24, thr=2.1, iterations=4).deband(denoise)

stronger_deband = vdb.Placebo(radius=24, thr=3, iterations=4).deband(denoise)

deband = vst.replace_ranges(deband, stronger_deband, stronger_deband_ranges)
deband = core.std.MaskedMerge(deband, denoise, detail_mask)

grain = AddNoise.FBM_SIMPLEX.grain(deband, strength=(1.8, 1.1), size=1.2, dynamic=True, seed=1)
grain = AddNoise.FBM_SIMPLEX.grain(grain, strength=(0.8, 0.7), size=1.2, dynamic=False, seed=2)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = settings_builder_x265(
        crf=13.8, 
        aq_strength=0.7, psy_rd=2.05, psy_rdoq=2.1, chroma_qpoffsets=-2,
        qcomp=0.72, rd=3,
        preset="veryslow",
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    enc = x265(settings).encode(final)

    audio = do_audio(AMZN, track=0).to_track("Japanese 2.0 EAC3")

    chapters = Chapters([(0, "Introduction"), (opstart, "Opening"), (opend+1, "Episode"), (31649, "Ending"), (33807, "Preview")])

    mux(enc, audio, chapters)

else:
    from vstools import set_output as o 
    final.set_output()

    o(src)
    o(aa)
    o(dehalo)
    o(denoise)
    o(deband)

    o(amzn)
    o(dsnp)
    o(linemask)
    o(detail_mask)
    o(vdh.fine_dehalo(src, thlimi=60, thmi=60, show_mask=True))

    from vstools import plane
    o(core.std.StackHorizontal([plane(src, 1), plane(src, 2)]))
    o(core.std.StackHorizontal([plane(denoise, 1), plane(denoise, 2)]))