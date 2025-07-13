# Adaptive 3D Placement of Aerial Base Stations via Deep Reinforcement Learning

This repository is the implementation of the deep reinforcement learning (DRL) framework proposed in the paper [Adaptive 3D Placement of Multiple UAV-Mounted Base Stations in 6G Airborne Small Cells With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/10949495).

> Uncrewed Aerial Vehicle-mounted Base Stations (UAV-BSs) have been envisioned as a promising solution to enable high-quality services in next-generation mobile networks. With inherent flexibility, one key challenge is placing the UAV-BSs adaptively to time-varying network conditions to maintain stable connections. Conventional methods mainly focus on optimizing UAV-BS deployment in static networks where users are static or with limited mobility. This study considers a dynamic network where multiple non-stationary UAV-BSs are deployed to serve mobile users with time-varying heterogeneous traffic. Incomplete downloads of users are backlogged in download queues at the UAV-BSs, and together with the newly requested traffic, the download queue size reflects the user’s instantaneous traffic demand. With constraints on the queue stability, we aim to maximize the user’s long-term mean opinion score (MOS), reflecting how the perceived data rate satisfies their traffic demand. Since the users’ location and traffic demand vary over time, we dynamically group users into clusters using a K-means-based algorithm and adjust the 3D location of UAV-BSs using an actor-critic deep reinforcement learning (DRL) framework. The actor module is encoded using a deep neural network (DNN) that obtains the UAV-BS’s current location and a traffic heatmap of users to predict the optimal movement for UAV-BSs. The critic module utilizes Lyapunov optimization to control the queue stability constraint and evaluate the actor’s decisions. Extensive simulations demonstrate the proposed method’s superior performance over conventional rate-maximization approaches.

<p align="center">
  <img src="simulation/proposed-1000s.gif"  width="550"><br>
  <em>Illustration: Adaptive 3D placement of multiple UAVs for user experience improvement</em>
</p>

## How to run the simulation
1. Configure the simulation parameters in [`params_sys.py`](params_sys.py)
2. Run [`main.ipynb`](main.ipynb) start the simulation. This notebook also have plots used for check the system configuration and simulation outputs.
3. Check [`simulation`](simulation) folder to investigate all figures and numerical results (exported to xlsx files) created during the simulation at Step 2.

(Optional)
4. Use [`SingleDroneEnv_A2C.ipynb`](SingleDroneEnv_A2C.ipynb) to train an A2C agent following [Gymasinum](https://gymnasium.farama.org/)'s API standard for reinforcement learning. This trained agent could be used in the `K-Means + A2C` benchmark method in [`main.ipynb`](main.ipynb).
5. Investigate the user mobility model in [`mobility_model.py`](mobility_model.py), the user traffic model in [`traffic_model.py`](traffic_model.py), the wireless communication channel modelling in [`channel_model.py`](channel_model.py), and the proposed UAV movement method in [`uav_controller.py`](uav_controller.py).

## Cite this work
```
@ARTICLE{10949495,
  author={Hoang, Linh T. and Nguyen, Chuyen T. and Le, Hoang D. and Pham, Anh T.},
  journal={IEEE Transactions on Networking}, 
  title={Adaptive 3D Placement of Multiple UAV-Mounted Base Stations in 6G Airborne Small Cells With Deep Reinforcement Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Three-dimensional displays;Optimization;Heuristic algorithms;Quality of experience;Base stations;Autonomous aerial vehicles;Adaptive systems;Quality of service;Iterative methods;6G mobile communication;Aerial base stations;UAV-BS deployment;quality of experience;machine learning for communications},
  doi={10.1109/TON.2025.3552097}}
```
