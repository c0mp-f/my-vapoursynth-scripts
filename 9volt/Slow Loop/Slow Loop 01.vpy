import vapoursynth as vs
import vardefunc as vdf
import lvsfunc as lvf
import vstools as vst

from vsaa import Eedi3, Eedi3SR, transpose_aa, fine_aa, clamp_aa, upscaled_sraa
from vstools import depth, join, split, get_y, set_output as o
from vardefunc import Graigasm, AddGrain
from vsdehalo import HQDeringmod
from vsdenoise import BM3DCuda
from vsdeband import Placebo, F3kdb

core = vs.core
sp = lvf.comparison.stack_planes

no_den = [(442, 654), (1670, 1782), (26195, 26269)]
sraa_ranges = [
    (10326, 13543), (18172, 18461), (22828, 22892), (29905, 30132)
]

# sources
src = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop Vol. 1 JP BDMV/BDMV/STREAM/00001.m2ts")
src = depth(src, 16)

# denoising
cden = depth(core.wnnm.WNNM(depth(src, 32), sigma=[0, 0.8, 0.8], radius=2), 16)
denoise = BM3DCuda(cden, sigma=[1, 0, 0], radius=2, matrix=1).clip
denoise = vst.rfs(denoise, cden, no_den)

# AA
eedi = Eedi3()
eedi.opencl = True
eedi.gamma = 80

eediw = Eedi3SR()
eediw.opencl = True

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise)
sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aat = [transpose_aa(p, eedi) for p in split(denoise)]
aas = join([aat[0], aat[1], aat[2]])

aaw = fine_aa(denoise, True, eediw)

aa = clamp_aa(denoise, aaw, aas)

aa = vst.rfs(aa, sraa, sraa_ranges)

aa = vdf.merge_chroma(aa, aas)

# dering 
blur = core.bilateral.Bilateral(aa, sigmaS=1.5, sigmaR=4/255)
blur = core.bilateral.Bilateral(aa, ref=blur, sigmaS=0.75, sigmaR=3/255)
dering = HQDeringmod(aa, smooth=blur, mrad=2, msmooth=1, mthr=30, elast=1)

# deband
deband = F3kdb(24, 2, 0, use_neo=True).deband(dering)

preden = denoise.std.BoxBlur()
detail_mask = lvf.mask.detail_mask(preden, None, 3, 0.012, 0.017)

deband = core.std.MaskedMerge(deband, dering, detail_mask)

grain = Graigasm(
    thrs=[x << 8 for x in (45, 110, 140)], 
    strengths=[(0.3, 0), (0.25, 0), (0.25, 0)], 
    sizes=[1.25, 1.25, 1.25], sharps=[65, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True)]
).graining(deband)

final = depth(grain, 10)

final.set_output()

o(src)
o(denoise)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 01.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.72 --rc-lookahead 240 --crf 14 --aq-mode 3 --aq-strength 0.8 --psy-rd 2.15 --psy-rdoq 2.1 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/slow\ loop/ep01/final.hevc
