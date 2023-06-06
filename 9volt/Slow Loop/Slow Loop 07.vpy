import vapoursynth as vs
import lvsfunc as lvf
import vstools as vst

from vsaa import Eedi3, transpose_aa, fine_aa, clamp_aa, upscaled_sraa
from vstools import depth, join, split, get_y, set_output as o
from vsrgtools import lehmer_diff_merge

from vsdeband.noise import multi_graining, MultiGrainerT, AddNoise
from vsdeband import Placebo, F3kdb, deband_detail_mask

from vsdenoise import BM3DCudaRTC, BM3DCPU, Profile, Prefilter
from vsdenoise.mvtools import MVTools, MVToolsPresets
from vsdenoise.ccd import ccd, CCDMode
from vsdenoise import dpir

core = vs.core

ED = [(32007, 34164)]
edstart = ED[0][0]

sraa_ranges = [(23661, 23835), (34382, 34447)]

regrain_ranges = [(edstart+1430, edstart+1469), (edstart+1698, edstart+1868), (29735, 30080)]

dont_merge = [
    (504, 2757), (4568, 4614), (23243, 23342), (26049, 26104), (26578, 27259)
] + ED

# sources
src = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop S01 [CR WEB-DL 1080p x264 OPUS] [Sneedentric]/[Sneedentric] Slow Loop - S01E07 [CR WEB-DL 1080p x264 OPUS].mkv")
src = depth(src, 16)

src2 = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Slow Loop (01-12) (1080p) [Batch]/[SubsPlease] Slow Loop - 07 (1080p) [840BACEC].mkv")[240:]
src2 = depth(src2, 16)

merge = lehmer_diff_merge([src, src2])
merge = vst.rfs(merge, src, dont_merge)

# denoising
dpir_zones = [
    [regrain_ranges, 6.5]
]
deblock = dpir(merge, matrix=1, zones=dpir_zones)
deblock = vst.rfs(merge, deblock, regrain_ranges)

cden = ccd(deblock, thr=4, mode=CCDMode.BICUBIC_LUMA, matrix=1)
mvtools = MVTools(cden, **MVToolsPresets.SMDE(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(100, 50))
denoise = BM3DCudaRTC(mvtools, sigma=[1.5, 0.7], radius=2, matrix=1).final(deblock, mvtools)

# AA
eedi = Eedi3(opencl=True)

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise, aafunc=eedi)
sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aat = transpose_aa(denoise, eedi)
aa = core.std.MaskedMerge(denoise, aat, linemask)

#aa_stronger = clamp_aa(denoise, aa, sraa, 2)

aa = vst.rfs(aa, sraa, sraa_ranges)

# deband
deband = Placebo(radius=24, thr=2, iterations=2).deband(aa)

stronger_deband = Placebo(radius=24, thr=3.5, iterations=1).deband(aa)

deband = vst.rfs(deband, stronger_deband, [(14529, 15969)])

preden = denoise.std.BoxBlur()
detail_mask = deband_detail_mask(preden, brz=(0.075, 0.075))

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep07/lossless.mkv")

grain = AddNoise(strength=(2.8, 0), type=3, size=1.3, luma_scaling=18, dynamic=True, seed=1).grain(deband)
grain = AddNoise(strength=(2.4, 0), type=3, size=1.2, luma_scaling=8, dynamic=False, seed=2).grain(grain)

# Flashback regraining.
grain = vst.rfs(grain, AddNoise(strength=(3, 0), size=1.3, type=0, luma_scaling=None, seed=3).grain(deband), regrain_ranges)

final = vst.depth(grain, 10)

final.set_output()

o(src)
o(src2)
o(merge)
o(denoise)
o(deband)
o(detail_mask)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 07.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.72 --rc-lookahead 40 --crf 15 --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 2 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/slow\ loop/ep07/final.hevc