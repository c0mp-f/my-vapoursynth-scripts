import vapoursynth as vs
import vardefunc as vdf
import lvsfunc as lvf
import vstools as vst

from vsaa import Eedi3, Eedi3SR, transpose_aa, fine_aa, clamp_aa, upscaled_sraa
from vstools import depth, join, split, get_y, set_output as o
from vardefunc import Graigasm, AddGrain
from vsdeband import Placebo, F3kdb
from vsdehalo import HQDeringmod
from vsdenoise import BM3DCuda

core = vs.core

ED = [(32008, 34164)]
edstart = ED[0][0]

no_den = [(edstart+1698, edstart+1868)]

sraa_ranges = [(9222, 9397)]

stronger_deband_ranges = [(1946, 2047)]

# sources
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop Vol. 1 JP BDMV/BDMV/STREAM/00003.m2ts")
src = depth(gt, 16)

# denoising
deblock = core.dfttest.DFTTest(src, slocation=[0.0, 0.75, 0.3, 0.75, 0.5, 5, 0.7, 5, 1.0, 5])
deblock = vst.rfs(src, deblock, stronger_deband_ranges)

cden = depth(core.wnnm.WNNM(depth(deblock, 32), sigma=[0, 0.8, 0.8], radius=2), 16)
denoise = BM3DCuda(cden, sigma=[1, 0, 0], radius=2, matrix=1).clip
denoise = vst.rfs(denoise, cden, no_den)

# AA
eedi = Eedi3()
eedi.opencl = True

eedis = Eedi3SR()
eedis.opencl = True
eedis.alpha = 0.3
eedis.gamma = 20

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise, aafunc=eedis)
sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aat = [transpose_aa(p, eedi) for p in split(denoise)]
aas = join(aat)
aa = core.std.MaskedMerge(denoise, aas, linemask)

aa = vst.rfs(aa, sraa, sraa_ranges)

aa = vdf.merge_chroma(aa, aas)

# dering 
blur = core.bilateral.Bilateral(aa, sigmaS=1.5, sigmaR=4/255)
blur = core.bilateral.Bilateral(aa, ref=blur, sigmaS=0.8, sigmaR=3/255)
dering = HQDeringmod(aa, smooth=blur, mrad=2, msmooth=1, mthr=30, elast=1)

# deband
deband = F3kdb(24, 2, 0, use_neo=True).deband(dering)

stronger_deband = Placebo(40, 2, 0, 2).deband(dering)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)

preden = denoise.std.BoxBlur()
detail_mask = lvf.mask.detail_mask(preden, None, 3, 0.01, 0.017)

deband = core.std.MaskedMerge(deband, dering, detail_mask)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 110, 140)], 
    strengths=[(0.3, 0), (0.25, 0), (0.3, 0)], 
    sizes=[1.3, 1.3, 1.3], sharps=[65, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True), AddGrain(seed=3, constant=True)]
).graining(deband)

regrain = core.noise.Add(deband, var=2.5, xsize=5, ysize=5, type=2)
regrain = core.std.MaskedMerge(deband, regrain, core.adg.Mask(regrain.std.PlaneStats(), 50))
grain = vst.rfs(grain, regrain, stronger_deband_ranges)

final = depth(grain, 10)

final.set_output()

o(gt)
o(denoise)
o(deband)
o(detail_mask)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 03.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-1 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.71 --rc-lookahead 240 --crf 13 --aq-mode 3 --aq-strength 0.7 --psy-rd 2 --psy-rdoq 1.7 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/slow\ loop/ep03/final.hevc
