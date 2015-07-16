#!/bin/bash

features=$1
datadir=$2

fDesc="audio2sphinx,1:1:0:0:0:0,13,0:0:0"
fDescCLR="audio2sphinx,1:3:2:0:0:0,13,1:1:300:4"

show=sound

h=3
c=2
ubm=ubm.gmm


java -cp lium_spkdiarization-8.4.1.jar fr.lium.spkDiarization.programs.MSegInit --fInputMask=$features --fInputDesc=$fDesc --sInputMask=$uem --sOutputMask=$datadir/%s.i.seg --sOutputFormat=seg.xml,UTF8 $show

#GLR based segmentation, make small segments
java -cp lium_spkdiarization-8.4.1.jar fr.lium.spkDiarization.programs.MSeg  --kind=FULL --sMethod=GLR --fInputMask=$features --fInputDesc=$fDesc --sInputMask=$datadir/%s.i.seg --sOutputMask=$datadir/%s.s.seg --sModelWindowSize=50 --sMinimumWindowSize=50 --sOutputFormat=seg.xml,UTF8 --sInputFormat=seg.xml,UTF8 $show

# linear clustering
java -cp lium_spkdiarization-8.4.1.jar fr.lium.spkDiarization.programs.MClust --fInputMask=$features --fInputDesc=$fDesc --sInputMask=$datadir/%s.s.seg --sOutputMask=$datadir/%s.l.seg --cMethod=l --cThr=2 --sOutputFormat=seg.xml,UTF8 --sInputFormat=seg.xml,UTF8 $show
 
# hierarchical clustering
java -cp lium_spkdiarization-8.4.1.jar fr.lium.spkDiarization.programs.MClust --fInputMask=$features --fInputDesc=$fDesc --sInputMask=$datadir/%s.l.seg --sOutputMask=$datadir/%s.h.$h.seg --cMethod=h --cThr=$h --sOutputFormat=seg.xml,UTF8 --sInputFormat=seg.xml,UTF8 $show
 
#CLR clustering
# Features contain static and delta and are centered and reduced (--fdesc)
java -cp lium_spkdiarization-8.4.1.jar fr.lium.spkDiarization.programs.MClust --fInputMask=$features --fInputDesc=$fDescCLR --sInputMask=$datadir/%s.h.$h.seg --sOutputMask=$datadir/segments.seg --cMethod=ce --cThr=$c --tInputMask=$ubm --emCtrl=1,5,0.01 --sTop=5,$ubm --cMinimumOfCluster=3 --sOutputFormat=seg.xml,UTF8 --sInputFormat=seg.xml,UTF8 $show
 
