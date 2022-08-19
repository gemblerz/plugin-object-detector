FROM kwelbeck/base-ros2-with-empty-overlay:ml

RUN apt-get update \
  && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /
RUN pip3 install --upgrade pip \
  && pip3 install --no-cache-dir -r /requirements.txt

RUN mkdir -p $ROS_WS/src
WORKDIR $ROS_WS/src
COPY ros-packages .
RUN vcs import < repos
WORKDIR $ROS_WS
SHELL ["/bin/bash", "-c"]
RUN source $ROS_ROOT/setup.bash && colcon build --symlink-install && source $ROS_WS/install/setup.bash

WORKDIR /app
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
COPY ./ /app/
ENTRYPOINT ["/app/ros_entrypoint.sh"]
CMD ["python3", "/app/app.py"]
