# Use COLMAP to compute 6DoF camera poses
python imgs2poses.py data/testscene/

# Create MPIs using pretrained network
python imgs2mpis.py \
    data/testscene/ \
    data/testscene/mpis_360 \
    --height 360
    
# Generate smooth path of poses for new views
mkdir data/testscene/outputs/
python imgs2renderpath.py \
    data/testscene/ \
    data/testscene/outputs/test_path.txt \
    --spiral
    
cd cuda_renderer && make && cd ..
    
# Render novel views using input MPIs and poses
cuda_renderer/cuda_renderer \
    data/testscene/mpis_360 \
    data/testscene/outputs/test_path.txt \
    data/testscene/outputs/test_vid.mp4 \
    360 .8 18