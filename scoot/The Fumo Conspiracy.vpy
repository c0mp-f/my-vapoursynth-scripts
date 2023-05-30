import vapoursynth as vs
import vsdenoise as vsd
import vsdehalo as vsdh
import awsmfunc as awf
import lvsfunc as lvf
import vstools as vst
import vsmasktools

from vstools import set_output as o
from vsdeinterlace import sivtc
from rekt import rektlvls

core = vs.core
sp = lvf.comparison.stack_planes

def nnedi(clip: vs.VideoNode) -> vs.VideoNode:
    up = vst.get_y(clip).znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3)
    up = up.std.Transpose()
    up = up.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3)
    return up.std.Transpose()

# source import
srcs = [core.d2v.Source(f"fumo_conspiracy{i}.d2v") for i in range(1, 6)]

src = srcs[0] + srcs[1] + srcs[2] + srcs[3] + srcs[4]
src = vst.depth(src, 16)

ivtc = sivtc(src, pattern=1, tff=False)

# edgefixing
rekt = rektlvls(ivtc, 
    rownum=[477, 478], rowval=[3, 2],
)
edgefix = awf.bbmod(rekt, bottom=1)

# denoise
denoise = vsd.nl_means(edgefix, strength=0.6, tr=2, sr=4, planes=[1, 2])

# dehaloing
fd_args = dict(rx=4, thmi=50, thma=100, thlimi=70, thlima=128, edgemask=vsmasktools.edge.Robinson5())

up = nnedi(denoise)

blur = vsd.Prefilter.GAUSSBLUR2

dehalo = vsdh.dehalo_sigma(up, brightstr=1, darkstr=0.01, blur_func=blur(sigma=2.5)).resize.Point(720, 480)

hmask = vsdh.fine_dehalo(up, **fd_args, show_mask=True).resize.Bicubic(720, 480)

dehalo = core.std.MaskedMerge(vst.get_y(denoise), dehalo, hmask)

dehalo = core.std.ShufflePlanes([dehalo, denoise], [0, 1, 2], vs.YUV)

# deringing
deringm = vsmasktools.edge.Prewitt().edgemask(vst.get_y(dehalo))
deringm = deringm.std.Binarize(50<<8).std.Maximum().std.Maximum().std.Inflate()

zones = [
    [[(13120, 13539)], 25],
    [(24614, 24804), 15]
]
dpir = vsd.dpir(dehalo, strength=10, zones=zones, matrix=6)

dering = core.std.MaskedMerge(dehalo, dpir, deringm)

deblock = vst.rfs(dering, dpir, [(13120, 13539), (24614, 24804)]) #nvm im too lazy for this

dering = vsdh.edge_cleaner(deblock, strength=3, hot=True)

final = vst.depth(dering, 10)

final = core.lsmas.LWLibavSource("/mnt/HDD1/lupin/movie.mkv")

final.set_output()

o(ivtc)
o(denoise)
o(dering)
o(dering)
o(deringm)
o(sp(ivtc))
o(sp(final))

#vspipe -c y4m ~/my-vapoursynth-scripts/scoot/The\ Fumo\ Conspiracy.vpy - | \
#x265 - \
#--y4m --preset veryslow --no-strong-intra-smoothing --no-sao --no-open-gop --no-cutree --no-amp --no-rect --high-tier --aud --repeat-headers \
#--input-depth 10 --output-depth 10 --range limited --colormatrix 6 --colorprim 6 --transfer 6 --sar 80:79 \
#--bframes 16 --rd 3 --merange 57 --deblock -2:-2 --rc-lookahead 60 \
#--aq-mode 3 --aq-strength 0.7 \
#--crf 10 --qcomp 0.74 --cbqpoffs -3 --crqpoffs -3 --psy-rd 2.05 --psy-rdoq 2.15 --ipratio 1.4 --pbratio 1.3 \
#-o /mnt/HDD1/lupin/movie.hevc
