import vapoursynth as vs
import vsmuxtools as mux
import vsdenoise as vsd
import awsmfunc as awf
import vsdeband as vdb
import lvsfunc as lvf
import vstools as vst
import vsmasktools
import vsaa

from vsdenoise import MVToolsPresets, Prefilter, Profile
from vstools import depth, get_w, get_y, join, split
from lvsfunc import stack_planes as sp
from vodesfunc import set_output as o
from vskernels import Lanczos
from rekt import rektlvls

from filter import merge_chroma

core = vs.core 

no_den = [(0, 359)]

# muxtools setup
setup = mux.Setup("NCED")

USA = mux.src_file("/mnt/HDD1/Videos/Anime/TV/Lupin.Sansei.Part 5.1080p.USA.BLURAY.AVC.DTS-HD.MA.2.0-ARiN/LUPIN III PART V DISC02/BDMV/STREAM/00016.m2ts", True)
usa = vst.initialize_clip(USA.src, 16)

ita = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[BDMV] Lupin III (S5) [ITA]/LUPIN_BD_1/BDMV/STREAM/00014.m2ts", True, (24, None))
ita = vst.initialize_clip(ita.src_cut, 16)

JPN = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[ルパン三世][2018][TV][ルパン三世 PART5][BDMV][JPN][Vol.01-Vol.05 Fin]/LUPIN_PART5_4/BDMV/STREAM/00012.m2ts", True)
jpn = vst.initialize_clip(JPN.src, 16)

merge = vsd.frequency_merge(usa, ita, jpn)

# edging
rekt = rektlvls(
    merge, 
    colnum=[0, 1, 2, 1917, 1918, 1919], colval=[11, 2, -3, -4, 3, 12],
    rownum=[0, 1079], rowval=[5, 5]
)
bbmod = awf.bbmod(rekt, left=2, right=3, cBottom=1, cTop=1)

prefilter = vsd.DFTTest(sigma=10).denoise(bbmod)

detail_mask = vsmasktools.detail_mask_neo(prefilter, detail_brz=0.002, lines_brz=0.02)

# AA
linemask = vsmasktools.edge.ExSobel().edgemask(get_y(prefilter)).std.Binarize(110<<8).std.Maximum().std.Inflate()

aa = vsaa.upscaled_sraa(bbmod, rfactor=2)
aa = core.std.MaskedMerge(bbmod, aa, linemask)

aa = vst.rfs(bbmod, aa, [(1436, 1468)])

# chroma therapy
dpir_zones = [
    [no_den, 5]
]

dpir = vsd.dpir(aa, strength=40, zones=dpir_zones)
crepair = merge_chroma(aa, dpir)

crepair = vst.rfs(crepair, dpir, [(360, 443)])

# denoisering
ref = vsd.MVTools(crepair, tr=2, **MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)).degrain(thSAD=(125, 55))
denoise = vsd.BM3DCudaRTC(crepair, ref=ref, sigma=[1.5, 0], radius=2, profile=Profile.NORMAL, matrix=1).final(crepair, ref)

# debandering
deband = vdb.F3kdb(radius=31, thr=[192, 180]).deband(denoise)
deband = core.std.MaskedMerge(deband, denoise, detail_mask)

deband = vst.rfs(deband, denoise, no_den)

grain = vdb.AddNoise(strength=(2.1, 1.7), size=1.35, type=3, luma_scaling=8, dynamic=True, seed=1).grain(deband)
grain = vdb.AddNoise(strength=(0.7, 1.4), size=1.1, type=3, luma_scaling=12, dynamic=False, seed=2).grain(grain)

final = vst.finalize_clip(grain, 10)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13.5, 
        aq_strength=0.7, psy_rd=2, psy_rdoq=2, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow",
        deblock=[-2, -1], bframes=10,
        rect=False
    )
    v = mux.x265(settings).encode(final)

    jp_a = mux.FFMpeg.Extractor(track=0).extract_audio(JPN.file)
    jp_a = mux.Opus(bitrate=192).encode_audio(jp_a)
    
    song = mux.SubTrack("/mnt/HDD1/lupin/NCED.ass")
    font = mux.FontFile("/home/c0mp/.fonts/part5/KeepCalm-Medium.ttf")
    
    mux.mux(
        v, jp_a, song, font, 
        tmdb=mux.TmdbConfig(31572, season=5, write_date=True, write_ids=True)
    )
else:
    final.set_output()

    o(usa)

    o(bbmod)
    o(denoise)
    o(deband)

    o(detail_mask)
    o(linemask)

    o(sp(bbmod))
    o(sp(crepair))