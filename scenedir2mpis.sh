mkdir -p $1/outputs
python imgs2poses.py $1
python imgs2mpis.py $1 $1/mpis_$2 --height $2
python imgs2renderpath.py $1 $1/outputs/test_path.txt
cuda_renderer/cuda_renderer $1/mpis_$2 $1/outputs/test_path.txt $1/outputs/test_vid.mp4 -1 .8 18