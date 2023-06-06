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

core = vs.core
sp = lvf.comparison.stack_planes

no_den = [(442, 654), (1670, 1782), (26195, 26269)]
sraa_ranges = [
    (10326, 12931), (13166, 13543), (18172, 18461), (22828, 22892), (29905, 30132)
]
stronger_deband_ranges = [
    (13115, 13444)
]

# sources
src = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop S01 [CR WEB-DL 1080p x264 OPUS] [Sneedentric]/[Sneedentric] Slow Loop - S01E01 [CR WEB-DL 1080p x264 OPUS].mkv")
src = depth(src, 16)

src2 = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Slow Loop (01-12) (1080p) [Batch]/[SubsPlease] Slow Loop - 01v2 (1080p) [DF99A231].mkv")[240:]
src2 = depth(src2, 16)

merge = lehmer_diff_merge([src, src2])
merge = vst.rfs(src, merge, [(31951, None)])

# denoising
cden = ccd(merge, thr=2, mode=CCDMode.BICUBIC_LUMA, matrix=1)
denoise = MVTools(cden, **MVToolsPresets.SMDE(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(80, 50))
denoise = BM3DCudaRTC(denoise, sigma=[1.2, 0.85, 0.85], profile=Profile.NORMAL, radius=2, matrix=1).final(merge, denoise)

# It destroys a dark scene. Lazy fix
denoise_weaker = BM3DCudaRTC(denoise, sigma=0.5, profile=Profile.NORMAL, radius=2, matrix=1).final(merge, denoise)
denoise = vst.rfs(denoise, denoise_weaker, [(31951, 33269)])

# AA
eedi = Eedi3(gamma=80, opencl=False)

eediw = Eedi3(opencl=False)

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise)
sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aas = transpose_aa(denoise, eedi)

aaw = fine_aa(denoise, True, eedi)

aa = clamp_aa(denoise, aaw, aas)

aa = vst.rfs(aa, sraa, sraa_ranges)

# deband
deband = Placebo(radius=24, thr=2, iterations=2).deband(aa)

stronger_deband = Placebo(radius=24, thr=4, iterations=2).deband(aa)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)

preden = denoise.std.BoxBlur()
detail_mask = deband_detail_mask(preden, brz=(0.075, 0.075))

deband = core.std.MaskedMerge(deband, aa, detail_mask)
deband = core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep01/lossless.mkv") # 49 fps

grain = AddNoise(strength=(3.0, 0), type=3, size=1.25, luma_scaling=24, dynamic=True, seed=1).grain(deband)
grain = AddNoise(strength=(2.5, 0), type=3, size=1.30, luma_scaling=8, dynamic=False, seed=2).grain(grain)

# Flashback regraining.
grain = vst.rfs(grain, AddNoise(strength=(4, 0), size=1.5, type=0, luma_scaling=None, seed=3).grain(deband), no_den)

final = depth(grain, 10) # 22 fps

final.set_output()

o(merge)
o(denoise)
o(final)
o(detail_mask)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 01.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.7 --rc-lookahead 40 --crf 14 --aq-mode 3 --aq-strength 0.77 --psy-rd 2.1 --psy-rdoq 2.05 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/slow\ loop/ep01/final.hevc
