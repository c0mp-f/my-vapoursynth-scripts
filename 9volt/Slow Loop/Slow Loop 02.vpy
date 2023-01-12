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
from acsuite import eztrim

core = vs.core

OP = [(1774, 4028)]
opstart = OP[0][0]

ED = [(32009, 34166)]

no_den = [(9528, 9802), (14174, 14709), (26543, 27547), (28732, 28680), (33707, 33847)]
sraa_ranges = [
    (opstart+902, opstart+940), (438, 741), (5170, 5273), (6413, 6914)
]

# sources
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/Slow Loop Vol. 1 JP BDMV/BDMV/STREAM/00002.m2ts")
src = depth(gt, 16)

# denoising
cden = depth(core.wnnm.WNNM(depth(src, 32), sigma=[0, 0.8, 0.8], radius=2), 16)
denoise = BM3DCuda(cden, sigma=[1, 0, 0], radius=2, matrix=1).clip
denoise = vst.rfs(denoise, cden, no_den)

# AA
eedi = Eedi3()
eedi.opencl = True

eedis = Eedi3SR()
eedis.opencl = True

linemask = core.std.Sobel(get_y(denoise)).std.Binarize(30<<8).std.Inflate()
sraa = upscaled_sraa(denoise, aafunc=eedis)
#sraa = core.std.MaskedMerge(denoise, sraa, linemask)

aat = [transpose_aa(p, eedi) for p in split(denoise)]
aas = join(aat)
aa = core.std.MaskedMerge(denoise, aas, linemask)

aa_stronger = clamp_aa(denoise, aas, sraa, 2)

aa = vst.rfs(aa, aa_stronger, sraa_ranges)

aa = vdf.merge_chroma(aa, aas)

# dering 
blur = core.bilateral.Bilateral(aa, sigmaS=1.5, sigmaR=4/255)
blur = core.bilateral.Bilateral(aa, ref=blur, sigmaS=0.75, sigmaR=3/255)
dering = HQDeringmod(aa, smooth=blur, mrad=2, msmooth=1, mthr=30, elast=1)

# deband
deband = F3kdb(24, 2, 0, use_neo=True).deband(dering)

preden = denoise.std.BoxBlur()
detail_mask = lvf.mask.detail_mask(preden, None, 3, 0.01, 0.017)

deband = core.std.MaskedMerge(deband, dering, detail_mask)

grain = Graigasm(
    thrs=[x << 8 for x in (40, 110, 140)], 
    strengths=[(0.3, 0), (0.25, 0), (0.3, 0)], 
    sizes=[1.3, 1.3, 1.3], sharps=[65, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True), AddGrain(seed=3, constant=True)]
).graining(deband)

final = depth(grain, 10)

final = core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep02/lossless.mkv")

final.set_output()

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 02.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-1 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.7 --rc-lookahead 240 --crf 14 --aq-mode 3 --aq-strength 0.7 --psy-rd 2.25 --psy-rdoq 2.25 --ipratio 1.3 --pbratio 1.2  - -o /mnt/HDD1/slow\ loop/ep02/final.hevc