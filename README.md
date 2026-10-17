# World Model Agent Lab

机器人世界模型推理、在线规划与 Agent 运行项目。此仓库负责加载已发布模型、预测候选动作后果、校准与执行门控、MuJoCo/ROS 2 会话、任务恢复、监控和本地指令客户端。

模型训练、训练数据采集与导入、优化器和训练命令已迁入独立的 [world-model-training](../world-model-training/README.md)。两个项目共享 `wmal` 命名空间；训练项目依赖本项目提供模型架构、预测/保存/加载 API、数据契约和仿真接口。检查点与校准格式保持兼容。权重、原始数据和机器人策略资产通常不随 Git 分发。

## 安装与验证

在本仓库根目录、选定的 Python 环境中安装：

```bash
python -m pip install -e '.[model-runtime,simulation,walking,monitor]'
python -m unittest discover -s tests -v
```

`model-runtime` 包含 NumPy、PyTorch 和 Pillow；`simulation` 添加 MuJoCo，`walking` 添加 ONNX Runtime，`monitor` 提供监控依赖。无需神经模型或仿真时可只安装基础包及需要的 extra。

也可用 [environment.yml](environment.yml) 创建 `wmal-runtime` Conda 环境。PyTorch 构建按目标硬件选择；本项目不固定 CUDA 版本。视频导出需要系统 `ffmpeg`/`ffprobe`。ROS 2 中间件、Python ABI 与 overlay 安装按 [本机 ROS 2 指南](docs/26_ros2_local_setup.md) 配置。

macOS 本地 MuJoCo 检查使用 `MUJOCO_GL=glfw`；Linux 无窗口渲染按主机实际 EGL/OSMesa 配置。安装、项目边界和模型选择见 [架构说明](docs/03_architecture.md)。

## 模型选择与运行

G1 的模型选择由运行配置中的 `world_model.factory` 与 `world_model.config` 决定；加载器验证状态/动作 schema、模型版本和动作时长。示例入口：

```bash
python scripts/g1_agent_sim.py --config configs/g1_neural.json
python scripts/g1_task_agent.py --help
python scripts/g1_llm_agent.py --help
```

具身推理由 `scripts/g1_task_agent.py --model-family embodied --inference-url URL` 显式选择 HTTP 服务；省略该选项时使用本地推理。远程服务需要与本地模型、语义、超时和校准绑定匹配。选择接口和适配器留在运行项目中，见 [持续具身 Agent 指南](docs/30_scalable_embodied_world.md)、[G1 闭环指南](docs/17_g1_world_model_agent_loop.md) 和 [任务图 Agent 指南](docs/28_task_graph_agent.md)。

本地指令客户端将任务、预测、真实反馈、实时画面及视频接到同一页面：

```bash
python scripts/agent_command_client.py --help
```

完整资产条件与启动参数见 [指令客户端指南](docs/29_agent_command_client.md)。只读监控可独立启动：

```bash
wmal-monitor --runs-root runs --port 8765
```

监控入口见 [监控指南](docs/24_agent_monitor.md)。运行包仅发布 `wmal-agent`、`wmal-serve`、`wmal-monitor` 三个命令；仿真、机器人与指令客户端入口继续保留在 `scripts/`。

## 项目职责

| 路径 | 职责 |
| --- | --- |
| `src/wmal/models/`、`perception/` | 模型构建、推理、保存/加载与独立校准 |
| `src/wmal/datasets/` | 共享数据格式、序列与来源契约 |
| `src/wmal/agents/`、`locomotion/`、`planners/` | 任务、反馈、候选轨迹预测与在线规划 |
| `src/wmal/envs/`、`robots/`、`ros2/` | 仿真、机器人资产与 ROS 2 接口 |
| `src/wmal/monitor/` | 实验监控、任务会话与视频展示 |
| `configs/`、`scripts/`、`tests/` | 运行配置、运行入口和运行契约测试 |

模型权重拟合、训练配置、数据准备脚本及训练专属测试属于训练项目；独立校准和在线动作序列搜索属于运行项目。训练项目按 [训练安装说明](../world-model-training/README.md) 先安装 `../world-model-agent-lab[model-runtime]`，再安装自己的训练 extra。

## 研究资料与证据

[规划网络架构](docs/architecture/planning-network.md) 与 [验证记录](docs/architecture/planning-network-verification.md) 是跨项目共用的规范资料，保留在此仓库。另见 [预测可信度研究路线](docs/21_research_plan_prediction_reliability.md)、[视觉模型与 Agent](docs/23_visual_world_training_and_agent.md)、[堆叠实验进展](docs/20_stack_block_experiment_progress.md)、[文献依据](docs/01_literature.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。历史验证结果有各自的数据、资产与环境边界；本地测试不等于真机或 sim-to-real 验证。
