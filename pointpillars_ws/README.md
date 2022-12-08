
## Quick Start

```shell
caktin_make && source devel/setup.bash && \
roslaunch det3d det3d.launch
```
**可视化**
```shell
wadda ros # pip3 install wadda
```

## launch 参数说明

- topic : 输入点云的topic
- result_topic : 输出结果的topic,结果以autoware_msgs/DetectedObjectArray的形式发布
- model_path : 模型路径
- score_threshold : 检测阈值


## benchmark
在[nuscenes的mini数据集](https://drive.google.com/file/d/1ZmojrOFJqZ5P52im_brHNSoIOT_A4yOi/view?usp=share_link)的`/LIDAR_TOP`上的benchmark结果如下

- Jeston Orin : 21.6ms
- Jeston Orin(Clocks Running) : 14.7ms
- Jeston Xavier NX : 49ms
- RTX2080 : 15.5ms
- RTX3050 : 21ms
- RTX3090 : 6.9ms
