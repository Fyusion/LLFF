mkdir -p $2
mkdir -p $2/images
mkdir -p $2/outputs
unzip -j $1 -d $2/images
python imgs2poses.py $2
python imgs2mpis.py $2 $2/mpis_$3 --height $3
python imgs2renderpath.py $2 $2/outputs/test_path.txt --spiral
cd cuda_renderer && make && cd ..
cuda_renderer/cuda_renderer $2/mpis_$3 $2/outputs/test_path.txt $2/outputs/test_vid.mp4 -1 .8 18
# python mpis2video.py $2/mpis_$3 $2/outputs/test_path.npy $2/outputs/test_vid.mp4 --crop_factor .8