import vapoursynth as vs
import lvsfunc as lvf
import vstools as vst

from vsaa import Eedi3, transpose_aa, fine_aa, clamp_aa, upscaled_sraa
from vstools import depth, join, split, get_y, set_output as o
from vsrgtools import lehmer_diff_merge

from vsdeband.noise import multi_graining, MultiGrainerT, AddNoise
from vsdeband import Placebo, F3kdb, deband_detail_mask

from vsdenoise.mvtools import MVTools, MVToolsPresets
from vsdenoise import BM3DCudaRTC, Profile, Prefilter
from vsdenoise.ccd import ccd, CCDMode
from vsdenoise import dpir

core = vs.core

ED = [(32008, 34164)]
edstart = ED[0][0]

sraa_ranges = [(9222, 9304)]

stronger_deband_ranges = [(1946, 2047)]

regrain_ranges = [(edstart+1430, edstart+1469), (edstart+1698, edstart+1868), (1852, 2047)]

# sources
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop S01 [CR WEB-DL 1080p x264 OPUS] [Sneedentric]/[Sneedentric] Slow Loop - S01E03 [CR WEB-DL 1080p x264 OPUS].mkv")
src = depth(gt, 16)

src2 = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Slow Loop (01-12) (1080p) [Batch]/[SubsPlease] Slow Loop - 03 (1080p) [CD9DF61E].mkv")[240:]
src2 = depth(src2, 16)

merge = lehmer_diff_merge([src, src2])
merge = vst.rfs(merge, src, [(3420, 3888)] + ED)

# denoising
dpir_zones = [
    [stronger_deband_ranges, 15]
]
deblock = dpir(merge, strength=0, matrix=1, zones=dpir_zones)

cden = ccd(deblock, thr=3, mode=CCDMode.BICUBIC_LUMA, matrix=1)
denoise = MVTools(cden, **MVToolsPresets.SMDE(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(100, 50))
denoise = BM3DCudaRTC(denoise, sigma=[1.3, 0.7], profile=Profile.NORMAL, radius=2, matrix=1).final(deblock, denoise)

# AA
eedi = Eedi3(gamma=55)

eedis = Eedi3(alpha=0.3, gamma=20)

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise, aafunc=eedis)
#sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aat = transpose_aa(denoise, eedi)
aa = core.std.MaskedMerge(denoise, aat, linemask)

aa_stronger = clamp_aa(denoise, aa, sraa, 2)

aa = vst.rfs(aa, aa_stronger, sraa_ranges)

# deband
deband = Placebo(radius=24, thr=2, iterations=2).deband(aa)

stronger_deband = Placebo(radius=24, thr=4, iterations=2).deband(aa)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)

preden = denoise.std.BoxBlur()
detail_mask = deband_detail_mask(preden, brz=(0.075, 0.075))

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = vst.rfs(core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep03/lossless.mkv"), deband, regrain_ranges + sraa_ranges + stronger_deband_ranges)

grain = AddNoise(strength=(3.0, 0), type=3, size=1.25, luma_scaling=24, dynamic=True, seed=1).grain(deband)
grain = AddNoise(strength=(2.5, 0), type=3, size=1.30, luma_scaling=8, dynamic=False, seed=2).grain(grain)

# Flashback regraining.
grain = vst.rfs(grain, AddNoise(strength=(2, 0), size=1.3, type=0, luma_scaling=None, seed=3).grain(deband), regrain_ranges)

final = depth(grain, 10)

final.set_output()
o(core.lsmas.LWLibavSource("/mnt/HDD1/9v/Slow Loop 03.mkv"))

o(gt)
o(src2)
o(merge)
o(denoise)
o(deband)
o(detail_mask)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 03.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.71 --rc-lookahead 40 --crf 14 --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 1.7 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/slow\ loop/ep03/final.hevc
