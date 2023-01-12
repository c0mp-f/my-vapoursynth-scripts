import vapoursynth as vs
import vardefunc as vdf
import lvsfunc as lvf
import vstools as vst
import vsdeband

from vsaa import Eedi3, Eedi3SR, transpose_aa, fine_aa, clamp_aa, upscaled_sraa
from vstools import depth, join, split, get_y, set_output as o
from vardefunc import Graigasm, AddGrain
from vsdeband import Placebo, F3kdb
from vsdehalo import HQDeringmod
from vsdenoise import BM3DCuda

core = vs.core

# sources
gt = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[BDMV][220525][Slow Loop][Vol.03]/BDMV/STREAM/00004.m2ts")
src = depth(gt, 16)

sraa_ranges = [(src.num_frames - 1, None)]

stronger_deband_ranges = [(src.num_frames - 1, None)]

# denoising
dpir_zones = [
    [[(src.num_frames - 1, None)], 2]
]

grays = core.resize.Bicubic(src, format=vs.GRAYS, matrix_in=1)
deblock = lvf.deblock.dpir(grays, zones=dpir_zones)
gray = depth(deblock, 16)

deblock = vdf.merge_chroma(gray, src)

cden = depth(core.wnnm.WNNM(depth(deblock, 32), sigma=[0, 0.8, 0.8], radius=2), 16)
denoise = BM3DCuda(cden, sigma=[1.35, 0, 0], radius=2, matrix=1).clip

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

aa = vst.rfs(aa, sraa, sraa_ranges)

aa = vdf.merge_chroma(aa, aas)

# dering 
blur = core.bilateral.Bilateral(aa, sigmaS=2, sigmaR=4/255)
blur = core.bilateral.Bilateral(aa, ref=blur, sigmaS=0.9, sigmaR=3/255)

dering = HQDeringmod(aa, smooth=blur, mrad=2, msmooth=1, mthr=30, elast=2)

# deband
deband = F3kdb(24, 2, 0, use_neo=True).deband(dering)

stronger_deband = Placebo(24, 2.3, 0, 3).deband(dering)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)

preden = denoise.std.BoxBlur()
detail_mask = lvf.mask.detail_mask(preden, None, 3, 0.0085, 0.0085)

deband = core.std.MaskedMerge(deband, dering, detail_mask)

grain = Graigasm(
    thrs=[x << 8 for x in (52, 110, 140)], 
    strengths=[(0.25, 0), (0.25, 0), (0.3, 0)], 
    sizes=[1.25, 1.2, 1.3], sharps=[60, 60, 65],
    grainers=[AddGrain(seed=1), AddGrain(seed=2, constant=True), AddGrain(seed=3, constant=True)]
).graining(deband)

final = depth(grain, 10)

final = core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep12/lossless.mkv") + core.lsmas.LWLibavSource("/mnt/HDD1/slow loop/ep12/lossless2.mkv")

final.set_output()

o(gt)
o(denoise)
o(aa)
o(dering)
o(deband)

#vspipe -c y4m ~/my-vapoursynth-scripts/9volt/Slow\ Loop/Slow\ Loop\ 12.vpy - | x265 --preset veryslow --bframes 16 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --no-amp --no-rect --crqpoffs -3 --qcomp 0.72 --rc-lookahead 240 --crf 14 --aq-mode 3 --aq-strength 0.7 --psy-rd 2.05 --psy-rdoq 2.1 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/slow\ loop/ep12/final.hevc
