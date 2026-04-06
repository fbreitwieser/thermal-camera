#!/usr/bin/env bash

set -eu -o pipefail
VID=/dev/video2

DRAW_TEXT='x=3:y=3:borderw=1:bordercolor=white:fontfile=FreeSerif.ttf:text=MIN\\: %{metadata\\:lavfi.signalstats.YMIN}    MAX\\: %{metadata\\:lavfi.signalstats.YMAX}'
VF_MAIN="normalize=smoothing=10, format=pix_fmts=rgb48, pseudocolor=p=inferno, scale=w=2*iw:h=2*ih, drawtext=$DRAW_TEXT [thermal]"
VF_SECONDARY='drawgraph=m1=lavfi.signalstats.YMIN:fg1=0xFFFF9040:m2=lavfi.signalstats.YMAX:fg2=0xFF0000FF:bg=0x303030:min=18500:max=24500:slide=scroll:size=512x64 [graph]'

ffmpeg -i $VID -input_format yuyv422 \
    -video_size 256x384 \
    -vf 'crop=h=(ih/2):y=(ih/2)' \
    -pix_fmt yuyv422 \
    -f rawvideo - \
| ffplay \
    -pixel_format gray16le \
    -video_size 256x192 \
    -f rawvideo \
    -i - \
    -vf "signalstats, split [main][secondary]; [main] $VF_MAIN; [secondary] $VF_SECONDARY; [thermal][graph] vstack"

