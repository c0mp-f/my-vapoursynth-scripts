import vapoursynth as vs
import vsmuxtools as mux
import vsdenoise as vsd
import vsdeband as vdb
import awsmfunc as awf
import vstools as vst 
import vsscale as vsc
import vsmasktools
import vsaa

from vsdenoise import MVToolsPresets, Prefilter, Profile
from vskernels import Catrom, ZewiaCubic, NoScale
from lvsfunc import stack_planes as sp
from vstools import set_output as o
from rekt import rektlvls

from filter import texture_mask, merge_chroma

core = vs.core

OP = [(2638, 4794)]
opstart = OP[0][0]
opend = OP[0][1]

ED = [0]

regrain_ranges = [
    (opstart+216, opstart+251), (opstart+755, opstart+764)
]

stronger_deband_ranges = [    
    (7124, 7201), (9356, 9555), (10646, 10816), (11222, 11362), 
    (12602, 12743), (13875, 13992), (14773, 14808), 
    (18583, 18636), (21313, 21951), (32968, 33038)
]

weaker_deband_ranges = [
    (309, 763), (7551, 7723), (17879, 18036)
] + ED

dpir_ranges = [
    (12602, 12648), (13774, 13814), (14211, 14324)
]

NO_EDGEFIX = OP
NO_DESCALE = OP

OP_STRONGER_DEBAND_RANGES = [
    (opstart+730, opstart+754), (opstart+937, opstart+1039), (opstart+1347, opstart+1483), 
    (opstart+1759, opstart+2031)
] + regrain_ranges

cr_merge = [
    (opstart+749, opstart+754), (7055, 7082), (29770, 34046)
]

# muxtools setup
setup = mux.Setup("12")

jp = mux.src_file("/mnt/HDD1/Videos/Anime/TV/MOBPSYCHO100_3_DISC3/BDMV/STREAM/00004.m2ts", True)
src = vst.initialize_clip(jp.src)

cr = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[SubsPlease] Mob Psycho 100 S3 (01-12) (1080p) [Batch]/[SubsPlease] Mob Psycho 100 S3 - 12 (1080p) [E5058D7B].mkv", True)
cr = vst.initialize_clip(cr.src)

cr = vst.replace_ranges(cr, merge_chroma(src, cr), (29770, None))

src = vst.replace_ranges(src, cr, cr_merge)
# edgefixing
rekt = rektlvls(
    src,
    colnum=[0, 1919], colval=[15, 15],
    rownum=[0, 1079], rowval=[17, 16]
)
bb = awf.bbmod(rekt, top=1)
rekt = vst.replace_ranges(bb, src, NO_EDGEFIX)

preden = vsd.DFTTest(rekt, sigma=5).denoise(rekt)

linemask = vsmasktools.edge.KirschTCanny().edgemask(vst.get_y(preden)).std.Binarize(50<<8).std.Maximum().std.Minimum().std.Inflate()
detail_mask = vsmasktools.detail_mask_neo(preden, detail_brz=0.01, lines_brz=0.02)
textures = texture_mask(preden, 2, points=[x * 256 for x in (1.25, 2.2, 5, 15)])

based_mask = vst.replace_ranges(core.std.Expr([detail_mask, textures], "x y max"), linemask, regrain_ranges + [OP_STRONGER_DEBAND_RANGES[0]] + OP_STRONGER_DEBAND_RANGES[2:])

# descale 
catrom = Catrom.descale(vst.get_y(rekt), 1280, 720)

rescale = vsc.Waifu2x.scale(catrom, 3840, 2160)

# rfactor=2
# based_aa is just constant configgings
aa = vsaa.Eedi3(alpha=0.135, beta=0.5, gamma=50, vthresh0=12, vthresh1=24, sclip_aa=None).aa(rescale)

stronger_aa = vsaa.Eedi3(alpha=0.2, beta=0.5, gamma=50, vthresh0=12, vthresh1=24, sclip_aa=None).aa(rescale)

aa = vst.replace_ranges(aa, stronger_aa, (29266, 29343))
aa = Catrom.scale(aa, 1920, 1080)
aa = merge_chroma(aa, rekt)

# purely for the sake of comparison
rescale = Catrom.scale(rescale, 1920, 1080)
rescale = merge_chroma(rescale, rekt)
rescale = core.std.MaskedMerge(rekt, rescale, linemask)

aa = core.std.MaskedMerge(rekt, aa, linemask)

error = vsc.descale_detail_mask(rekt, Catrom().scale(catrom, 1920, 1080), thr=0.03)

aa = core.std.MaskedMerge(aa, rekt, error)

aa = vst.replace_ranges(aa, rekt, NO_DESCALE)

# denoise
zones = [
    [[(opstart+749, opstart+764)], 30],
    [[(opstart+216, opstart+251)], 20],    
    [[(opstart+490, opstart+556), (opstart+1347, opstart+1483)], 9],
    [dpir_ranges, 2.5]
]

deblock = vsd.dpir(aa, strength=0, zones=zones)
deblock = vst.replace_ranges(aa, deblock, OP)

cden = vsd.ccd(deblock, thr=4)
mv = vsd.MVTools(cden, **MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)).denoise(cden, thSAD=(120, 70))
denoise = vsd.BM3DCudaRTC(deblock, sigma=[1.4, 0], tr=2, profile=Profile.NORMAL).final(deblock, mv)

# deband
deband = vdb.Placebo(radius=28, thr=[2, 1.8], iterations=4).deband(denoise)

weaker_deband = vdb.Placebo(radius=28, thr=1.6, iterations=4).deband(denoise)

stronger_deband = vdb.Placebo(radius=28, thr=3.5, iterations=4).deband(denoise)

OP_STRONGER_DEBAND = vdb.Placebo(radius=24, thr=4, iterations=4).deband(denoise)

deband = vst.replace_ranges(deband, stronger_deband, stronger_deband_ranges)
deband = vst.replace_ranges(deband, weaker_deband, weaker_deband_ranges)
deband = vst.replace_ranges(deband, OP_STRONGER_DEBAND, OP_STRONGER_DEBAND_RANGES)

deband = core.std.MaskedMerge(deband, denoise, based_mask)

grain = vdb.AddNoise.FBM_SIMPLEX.grain(deband, strength=(1.85, 1.9), size=1.3, dynamic=True, luma_scaling=12, seed=1)
grain = vdb.AddNoise.FBM_SIMPLEX.grain(grain, strength=(1.85, 1.8), size=1.25, dynamic=False, luma_scaling=10, seed=2)

grain = vst.replace_ranges(
    grain, vdb.AddNoise.GAUSS.grain(deband, strength=(5, 1), seed=69),
    regrain_ranges
)

final = vst.finalize_clip(grain)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13.5, 
        aq_strength=0.7, psy_rd=2.05, psy_rdoq=2.15, chroma_qpoffsets=-2,
        qcomp=0.72, rd=3,
        preset="veryslow", 
        deblock=[-2, -2], bframes=16,
        rect=False
    )
    mux.x265(settings).encode(final, "/mnt/HDD1/mob/ep12/final")
else:
    final.set_output()

    o(src)

    o(cr)
    o(rescale)
    o(aa)
    o(denoise)
    o(deband)

    o(sp(src))
    o(sp(deband))

    o(linemask)
    o(based_mask)
