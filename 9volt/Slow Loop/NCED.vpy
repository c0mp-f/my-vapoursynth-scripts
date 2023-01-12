import vapoursynth as vs
import vardefunc as vdf
import lvsfunc as lvf
import vstools as vst
import vsdeband

from vsaa import Eedi3, Eedi3SR, transpose_aa, fine_aa, clamp_aa, upscaled_sraa
from vstools import depth, join, split, get_y, set_output as o
from vardefunc import Graigasm, AddGrain
from vsdehalo import HQDeringmod
from vsdenoise import BM3DCuda

core = vs.core
sp = lvf.comparison.stack_planes

no_den = [(698, 1868)]

# sources
src = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop Vol. 1 JP BDMV/BDMV/STREAM/00012.m2ts")
src = depth(src, 16)

# denoising
cden = depth(core.wnnm.WNNM(depth(src, 32), sigma=[0, 0.8, 0.8], radius=2), 16)
denoise = BM3DCuda(cden, sigma=[0.75, 0, 0], radius=2, matrix=1).clip
denoise = vst.rfs(denoise, cden, no_den)

# AA
eedi = Eedi3()
eedi.opencl = True

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()

aat = [transpose_aa(p, eedi) for p in split(denoise)]
aas = join(aat)
aa = core.std.MaskedMerge(denoise, aas, linemask)

aa = vdf.merge_chroma(aa, aas)

# dering 
blur = core.bilateral.Bilateral(aa, sigmaS=1.5, sigmaR=4/255)
blur = core.bilateral.Bilateral(aa, ref=blur, sigmaS=0.75, sigmaR=3/255)
dering = HQDeringmod(aa, smooth=blur, mrad=2, msmooth=1, mthr=30, elast=1)

# deband
deband = vsdeband.F3kdb(24, 2, 0, use_neo=True).deband(dering)

preden = denoise.std.BoxBlur()
detail_mask = lvf.mask.detail_mask(preden, None, 3, 0.012, 0.017)

deband = core.std.MaskedMerge(deband, dering, detail_mask)

grain = Graigasm(
    thrs=[x << 8 for x in (50, 90, 140)], 
    strengths=[(0.25, 0), (0.25, 0), (0.2, 0)], 
    sizes=[1.3, 1.3, 1.15], sharps=[60, 60, 50],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True), AddGrain(seed=3, constant=True)]
).graining(deband)

final = depth(grain, 10)

final.set_output()
o(src)
o(denoise)
o(deband)
o(grain)