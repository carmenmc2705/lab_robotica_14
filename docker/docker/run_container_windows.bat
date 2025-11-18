docker rm lab_rob_container

docker run -it ^
--env="DISPLAY=host.docker.internal:0" ^
--name lab_rob_container ^
--net=host ^
--privileged ^
--mount type=bind,source=C:\Users\carme\lab_rob_shared,target=/home/grupo14/lab_rob_shared ^
lab_rob_image ^
bash



