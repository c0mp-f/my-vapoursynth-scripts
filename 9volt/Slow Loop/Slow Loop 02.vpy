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

OP = [(1774, 4028)]
opstart = OP[0][0]

ED = [(32009, 34166)]

no_den = [(9528, 9802), (14174, 14709), (26543, 27547), (28732, 28680), (33439, 33478), (33707, 33847)]
sraa_ranges = [
    (opstart+902, opstart+940), (438, 741), (5170, 5273), (6413, 6914)
]

# sources
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop S01 [CR WEB-DL 1080p x264 OPUS] [Sneedentric]/[Sneedentric] Slow Loop - S01E02 [CR WEB-DL 1080p x264 OPUS].mkv")
src = depth(gt, 16)

# denoising
cden = ccd(src, thr=2, mode=CCDMode.BICUBIC_LUMA, matrix=1)
denoise = MVTools(cden, **MVToolsPresets.SMDE(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(80, 50))
denoise = BM3DCudaRTC(denoise, sigma=[1.2, 0.85, 0.85], profile=Profile.NORMAL, radius=2, matrix=1).final(src, denoise)

# AA
eedi = Eedi3()

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise, aafunc=eedi)
#sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aat = transpose_aa(denoise, eedi)
aa = core.std.MaskedMerge(denoise, aat, linemask)

aa_stronger = clamp_aa(denoise, aa, sraa, 2)

aa = vst.rfs(aa, aa_stronger, sraa_ranges)

# deband
deband = Placebo(radius=24, thr=2, iterations=2).deband(aa)

preden = denoise.std.BoxBlur()
detail_mask = deband_detail_mask(preden, brz=(0.19, 0.12))

deband = core.std.MaskedMerge(deband, aa, detail_mask)

grain = AddNoise(strength=(3.0, 0), type=3, size=1.25, luma_scaling=24, dynamic=True, seed=1).grain(deband)
grain = AddNoise(strength=(2.5, 0), type=3, size=1.30, luma_scaling=8, dynamic=False, seed=2).grain(grain)

# Flashback regraining.
grain = vst.rfs(grain, AddNoise(strength=(4, 0), size=1.3, type=0, luma_scaling=None, seed=3).grain(deband), no_den)

final = depth(grain, 10)
final = core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep02/lossless.mkv")

final.set_output()
o(lvf.diff(final, core.lsmas.LWLibavSource("/mnt/HDD1/9v/Slow Loop 02.mkv")))

o(src)
o(denoise)
o(deband)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 02.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.7 --rc-lookahead 40 --crf 14 --aq-mode 3 --aq-strength 0.77 --psy-rd 2.1 --psy-rdoq 2.05 --ipratio 1.4 --pbratio 1.3 --zones 9528,9802,b=0.8/14174,14709,b=0.8/26543,27547,b=0.8/28732,28680,b=0.8/33439,33478,b=0.8/33707,33847,b=0.8 - -o /mnt/HDD1/slow\ loop/ep02/final.hevc