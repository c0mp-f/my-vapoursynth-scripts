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
from rekt import rektlvls

from filter import merge_chroma

core = vs.core 

# muxtools setup
setup = mux.Setup("NCOP")

USA = mux.src_file("/mnt/HDD1/Videos/Anime/TV/Lupin.Sansei.Part 5.1080p.USA.BLURAY.AVC.DTS-HD.MA.2.0-ARiN/LUPIN III PART V DISC02/BDMV/STREAM/00015.m2ts", True)
usa = depth(USA.src, 16)

ita = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[BDMV] Lupin III (S5) [ITA]/LUPIN_BD_1/BDMV/STREAM/00013.m2ts", True, (24, None))
ita = depth(ita.src_cut, 16)

JPN = mux.src_file("/mnt/HDD1/Videos/Anime/TV/[ルパン三世][2018][TV][ルパン三世 PART5][BDMV][JPN][Vol.01-Vol.05 Fin]/LUPIN_PART5_4/BDMV/STREAM/00011.m2ts", True)
jpn = depth(JPN.src, 16)

merge = vsd.frequency_merge(usa, ita, jpn)

# edging
rekt = rektlvls(
    merge, 
    colnum=[0, 1, 2, 1917, 1918, 1919], colval=[11, 2, -3, -4, 3, 12],
    rownum=[0, 1079], rowval=[5, 5]
)
bbmod = awf.bbmod(rekt, left=2, right=2, cBottom=1, cTop=1)

prefilter = vsd.DFTTest(sigma=10).denoise(bbmod)

detail_mask = vsmasktools.detail_mask_neo(prefilter, detail_brz=0.002, lines_brz=0.02)

# chroma therapy
#dpir = vsd.dpir(bbmod, strength=5)
#crepair = merge_chroma(bbmod, dpir)
# looks fine without therapy
crepair = bbmod

# denoisering
ref = vsd.MVTools(crepair, tr=2, **MVToolsPresets.FAST(prefilter=Prefilter.DFTTEST)).degrain(thSAD=(125, 65))
denoise = vsd.BM3DCudaRTC(crepair, ref=ref, sigma=[1.5, 0], radius=2, profile=Profile.NORMAL, matrix=1).final(crepair, ref)

# debandering
deband = vdb.Placebo(radius=24, thr=2, iterations=4).deband(denoise)
deband = core.std.MaskedMerge(deband, denoise, detail_mask)

grain = vdb.AddNoise(strength=(1.8, 1.7), size=1.3, type=3, luma_scaling=12, dynamic=True, seed=1).grain(deband)
grain = vdb.AddNoise(strength=(1.6, 1.7), size=1.3, type=3, luma_scaling=10, dynamic=False, seed=2).grain(grain)

final = depth(grain, 10)

if __name__ == "__main__":
    settings = mux.settings_builder_x265(
        crf=13.5, 
        aq_strength=0.7, psy_rd=2, psy_rdoq=2, chroma_qpoffsets=-3,
        qcomp=0.74, rd=3,
        preset="veryslow",
        deblock=[-2, -1], bframes=10,
        rect=False
    )
    v = mux.VideoTrack("/home/c0mp/my-vapoursynth-scripts/_workdir/NCOP/encoded.265", lang="")#v = mux.x265(settings).encode(final)
    #a = mux.FFMpeg.Extractor(track=0).extract_audio(JPN.file)
    #a = mux.Opus(bitrate=192).encode_audio(a)
    a = mux.AudioTrack("/mnt/HDD1/lupin/NCOP/NCOP.opus")
    mux.mux(v, a, tmdb=mux.TmdbConfig(31572, season=5, write_date=True, write_ids=True), outfile="/mnt/HDD1/lupin/NCOP/NCOP.mkv")
else:
    final.set_output()

    o(usa)

    o(bbmod)
    o(denoise)
    o(deband)

    o(detail_mask)

    o(sp(bbmod))
    o(sp(crepair))