
docker run -it --rm \
    --env="DISPLAY=host.docker.internal:0" \
    --name lab_rob_container \
    --net=host \
    --privileged \
    --mount type=bind,source="/mnt/c/Users/carme/lab_rob_shared",target="/home/grupo14/lab_rob_shared" \
    --env="ROS_MASTER_URI=http://172.20.10.2:11311" \
    --env="ROS_HOSTNAME=172.20.10.2" \
    lab_rob_image \
    bash
