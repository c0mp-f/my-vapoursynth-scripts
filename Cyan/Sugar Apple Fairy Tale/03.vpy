import vapoursynth as vs
import vardefunc as vdf
import vodesfunc as vof
import havsfunc as haf
import lvsfunc as lvf
import vstools as vst
import vsscale as vsc
import vsdenoise
import vsdehalo
import vsdeband
import vsmask
import vsaa

from vskernels import Spline64, Bicubic, BicubicSharp
from dfttest2 import DFTTest, Backend
from vstools import set_output as o
from vsdenoise import ccd, BM3DCPU
from vsdeband import Placebo

core = vs.core

OP = [(1368, 3524)]

taa_ranges = [
    (13823, 14004), (28638, 28792), (29631, 29767)
]

stronger_deband_ranges = [
    (10385, 10485), (16121, 16449), (29410, 30184)
]

weaker_deband_ranges = [(25723, 25769)]

dont = (32489, None)

# sources
src = core.lsmas.LWLibavSource("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Sugar Apple Fairy Tale - 03 (1080p) [96F8E06F].mkv")
src = vst.depth(src, 16)

preden = DFTTest(vst.get_y(src), backend=Backend().CPU)

dehalo = vsdehalo.fine_dehalo(src, brightstr=1, highsens=80, thmi=20, thma=60)
dehalo = vsdehalo.fine_dehalo(dehalo, brightstr=0.75, highsens=80, thmi=20, thma=60)
dehalo = vst.rfs(src, dehalo, (13823, 13911))

# AA
singlerater = vsaa.Eedi3SR(opencl=True)

aafunc = vsaa.Eedi3(opencl=True, beta=0.45)

aamask = vsmask.edge.Robinson5().edgemask(preden, multi=1.4).std.Binarize(10<<8).std.Inflate()

sraa = vsaa.upscaled_sraa(dehalo, aafunc=singlerater)
sraa = core.std.MaskedMerge(dehalo, sraa, aamask)

taa = vsaa.transpose_aa(dehalo, aafunc)
taam = core.std.MaskedMerge(dehalo, sraa, aamask) # i don't remember why it's masking sraa but it looks fine, w/e

faa = vsaa.fine_aa(dehalo, taa=True, singlerater=singlerater)

aa = vsaa.clamp_aa(dehalo, faa, taa, 1.5)
aa = vst.rfs(aa, taam, taa_ranges)

with vdf.YUVPlanes(aa) as caa:
    caa.U = vsaa.transpose_aa(caa.U, aafunc)
    caa.V = vsaa.transpose_aa(caa.V, aafunc)

aa = caa.clip

# denoise
ccd_args = dict(tr=1, matrix=1, mode=vsdenoise.CCDMode.BICUBIC_LUMA)

cden = core.std.MaskedMerge(
    ccd(aa, thr=6, **ccd_args),
    ccd(aa, thr=3, **ccd_args),
    core.adg.Mask(aa.std.PlaneStats(), 36)
)

denoise = BM3DCPU(cden, sigma=[1.4, 0, 0], radius=2, matrix=1).clip

zone = [
    [[(13823, 14004)], 10]
]
deblock = lvf.deblock.dpir(denoise, zones=zone, matrix=1)

# deband
detail_mask = vsdeband.deband_detail_mask(preden, brz=[925, 925])

deband = Placebo(radius=24, thr=1.7, grains=0, iterations=2).deband(deblock)

stronger_deband = Placebo(radius=24, thr=3, grains=0, iterations=2).deband(deblock)

weaker_deband = Placebo(radius=24, thr=[1.5, 1], grains=0, iterations=1).deband(deblock)

deband = vst.rfs(deband, stronger_deband, stronger_deband_ranges)
deband = vst.rfs(deband, weaker_deband, weaker_deband_ranges)
deband = core.std.MaskedMerge(deband, deblock, detail_mask)

grain = vdf.Graigasm(
    thrs=[x << 8 for x in (36, 85, 120, 150)], 
    strengths=[(0.3, 0), (0.1, 0), (0.2, 0), (0.25, 0)], 
    sizes=[1.3, 1.2, 1.25, 1.3], sharps=[70, 80, 65, 75],
    grainers=[vdf.AddGrain(seed=1), vdf.AddGrain(seed=2, constant=True), vdf.AddGrain(seed=3, constant=True)]
).graining(deband)

final = vst.depth(grain, 10)

#final = final[3773:3800] + final[4258:4284] + final[11330:11400] + final[11579:11711] + final[15577:15600] + final[22275:22351] + final[29025:29125] + final[29200:29328] + final[29901:30140]

final.set_output()

o(src)
o(dehalo)
o(aa)
o(denoise)
o(deband)

o(vsdehalo.fine_dehalo(src, rx=2, highsens=60, show_mask=True, thmi=30, thma=90, thlimi=75, edgemask=vsmask.edge.Sobel()))

#o(linemask)

#o(core.adg.Mask(aa.std.PlaneStats(), 36))

#vspipe -c y4m ~/my-vapoursynth-scripts/Cyan/Sugar\ Apple\ Fairy\ Tale/03.vpy - | x265 --preset veryslow --bframes 12 --no-strong-intra-smoothing --y4m --rd 3 --no-open-gop --no-sao --no-cutree --no-rect --no-amp --merange 57 --high-tier --range limited --aud --repeat-headers --input-depth 10 --output-depth 10 --deblock -1:-1 --colormatrix 1 --colorprim 1 --transfer 1 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.71 --rc-lookahead 60 --crf 13 --aq-mode 3 --aq-strength 0.71 --psy-rd 2.08 --psy-rdoq 2.21 --ipratio 1.4 --pbratio 1.3  - -o /mnt/HDD1/apple/ep03/test.hevc
