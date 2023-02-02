import vapoursynth as vs
import vardefunc as vdf
import vodesfunc as vof
import lvsfunc as lvf
import vstools as vst
import vsscale as vsc
import vsdenoise
import vsdeband
import vsmask
import vsaa

from vskernels import Spline64, Bicubic
from dfttest2 import DFTTest, Backend
from vstools import set_output as o
from vsdenoise import ccd, BM3DCPU
from vsdeband import Placebo

core = vs.core

OP = [(30329, 32487)]
opstart = OP[0][0]

sraa_ranges = [
    (opstart+1273, opstart+1302), (None, 154), (3532, 3578), (10568, 10710)
]

taa_ranges = [
    (18810, 19335), (20229, 20399)
]

stronger_deband_ranges = [
    (6843, 6978)
]

# sources
src = core.lsmas.LWLibavSource("/mnt/SSD/[SubsPlease] Sugar Apple Fairy Tale - 01 (1080p) [641C2F45].mkv")
src = vst.depth(src, 16)

srcy = vst.get_y(src)

preden = DFTTest(srcy, backend=Backend().CPU)

# AA
singlerater = vsaa.Eedi3SR()
singlerater.opencl = True

aafunc = vsaa.Eedi3()
aafunc.gamma = 50
aafunc.opencl = True

aamask = vsmask.edge.Robinson5().edgemask(src, multi=1.4).std.Binarize(10<<8).std.Inflate()

taa = vsaa.transpose_aa(src, aafunc)
taam = core.std.MaskedMerge(src, taa, aamask)

faa = vsaa.fine_aa(src, taa=True, singlerater=singlerater)

sraa = vsaa.upscaled_sraa(src, aafunc=singlerater)
sraa = core.std.MaskedMerge(src, sraa, aamask)

aa = vsaa.clamp_aa(src, faa, taa, 0.8)
aa = vst.rfs(aa, sraa, sraa_ranges)
aa = vst.rfs(aa, taam, taa_ranges)

with vdf.YUVPlanes(aa) as caa:
    caa.U = vsaa.transpose_aa(caa.U, aafunc)
    caa.V = vsaa.transpose_aa(caa.V, aafunc)

aa = caa.clip

# denoise
cden = core.std.MaskedMerge(
    ccd(aa, thr=5, tr=1, matrix=1, mode=vsdenoise.CCDMode.BICUBIC_LUMA),
    ccd(aa, thr=3, tr=1, matrix=1, mode=vsdenoise.CCDMode.BICUBIC_LUMA),
    core.adg.Mask(aa.std.PlaneStats(), 36)
)
denoise = BM3DCPU(cden, sigma=[1.85, 0, 0], radius=2, matrix=1).clip

# deband
detail_mask = vsdeband.deband_detail_mask(preden)

deband = Placebo(radius=24, thr=[2, 1], grains=0, iterations=2).deband(denoise)

stronger_deband = Placebo(radius=24, thr=[3.3, 2], grains=0, iterations=3).deband(denoise)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = core.std.MaskedMerge(deband, denoise, detail_mask)

grain = vdf.Graigasm(
    thrs=[x << 8 for x in (35, 85, 120)], 
    strengths=[(0.2, 0), (0.15, 0), (0.25, 0)], 
    sizes=[1.25, 1.15, 1.25], sharps=[65, 60, 65],
    grainers=[vdf.AddGrain(seed=1), vdf.AddGrain(seed=2, constant=True), vdf.AddGrain(seed=3, constant=True)]
).graining(deband)

final = vst.depth(grain, 10)

#final = final[771:806] + final[6742:6938] + final[10489:10588] + final[13915:14002] + final[14169:14284] + final[17590:17632]

final.set_output()

o(src)
o(aa)
o(denoise)
o(grain)

#o(detail_mask)

#o(linemask)

#o(core.adg.Mask(aa.std.PlaneStats(), 36))

#vspipe -c y4m ~/my-vapoursynth-scripts/Cyan/Sugar\ Apple\ Fairy\ Tale/01.vpy - | x265 --preset veryslow --bframes 12 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --no-rect --no-amp --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -2:-2 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.7 --rc-lookahead 60 --crf 14.5 --aq-mode 3 --aq-strength 0.8 --psy-rd 1.9 --psy-rdoq 2.25 --ipratio 1.4 --pbratio 1.3 --zones 6843,6978,b=1.3/13893,13995,b=1.15/14244,14329,b=1.2/18024,18095,b=1.3/18141,18254,b=1.15  - -o /mnt/HDD1/apple/ep01/final.hevc
