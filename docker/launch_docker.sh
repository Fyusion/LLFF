sudo docker build -t tf_colmap:latest .

sudo nvidia-docker run -it --rm -p 9999:8888 --volume /:/host --workdir /host$PWD tf_colmap bash