# 架构与时序

## 运行与训练项目边界

本仓库发布运行包 `world-model-agent-lab`：模型架构、推理/保存/加载、数据契约、独立校准、在线动作序列搜索、Agent、仿真/机器人与监控均归此项目。`../world-model-training` 发布独立的 `wmal.training`，拥有模型拟合、优化器、训练数据准备和训练命令；运行包不导入训练包。两个发行包通过隐式 `wmal` 命名空间共享接口。

运行依赖安装 `.[model-runtime]`；MuJoCo、ONNX 行走和监控分别通过 `simulation`、`walking`、`monitor` 选择。训练项目先安装本运行包，再按自己的 README 安装训练 extra。

模型提供者由运行层选择：G1 配置使用 `world_model.factory` 和 `world_model.config`；具身 Agent 通过 `--inference-url` 显式选择 HTTP 传输，省略时为本地推理。训练项目不会替运行层选择提供者。模型预测、数据与校准 schema 保持兼容。

以下时序图保留原始关节规划设计背景；当前入口与安装方式以 [运行 README](../README.md) 为准。

## 关节规划设计背景

```mermaid
flowchart TD
 T[用户任务] --> A[API Agent]
 A --> G[受限 joint_goal 校验]
 G --> P[ROS 2 世界模型规划服务]
 P --> M[动作条件状态预测与滚动规划]
 M --> E[机器人能力/限位校验]
 E --> U[ROS 2 执行动作]
 U --> S[MuJoCo / 机器人适配器]
 S --> O[RobotState topic]
 O --> A
 S --> D[轨迹与事件日志]
 D --> L[未来训练数据]
 L --> V[模型训练与版本发布 尚未实现]
 V --> P
 D --> R[冻结评估 尚未实现]
```

| 模块 | 输入→输出 | 职责与边界 |
|---|---|---|
| agents | 指令/Observation→受限 joint_goal | 调用 HTTP API；不访问物理积分器 |
| models | 状态与动作候选→未来状态预测 | DynamicsModel 插件接口；当前无已训练模型 |
| planners | Goal/Observation→PredictionReport 与首个动作 | 滚动规划；不代表安全证明 |
| skills | SkillSpec→ActionCommand | 技能库预留；尚未接入当前 joint_goal 路径 |
| envs/robots | ActionCommand→Observation | 唯一状态修改入口；坐标与控制模式适配 |
| communication | 消息→确认/错误 | 版本、时序和 episode 隔离 |
| datasets | 仿真轨迹→共享序列契约 | 数据格式及来源边界；拟合在独立训练项目 |
| evaluation/logging/analysis | 冻结快照→指标及图表 | 独立检测器与可追溯证据 |

```mermaid
sequenceDiagram
 participant A as Agent
 participant P as ROS 2 世界模型规划
 participant S as ROS 2 机器人 action
 participant E as 仿真owner
 participant L as 日志/检测
 A->>P: profile + observation + goal
 P-->>A: model version + predicted state + action plan
 A->>S: 带 episode/step 的第一步命令
 S->>E: command
 E->>L: MuJoCo step and new observation
 L-->>S: action result
 S-->>A: command receipt
 E-->>A: fresh RobotState
 A->>P: 新观测上重新规划
```

```mermaid
sequenceDiagram
 participant E as 采样器
 participant B as 数据/回放
 participant T as 训练器
 participant V as 验证器
 E->>B: 完整回合及技能片段
 B->>T: 仅train分组序列
 T->>V: 临时检查点
 V-->>T: 验证指标与模型哈希
 T->>E: 下个回合原子切换已发布版本
```

当前可运行例子只覆盖 joint_goal：Agent 得到关节目标后，请 planner 依据 Observation 和世界模型插件评估目标动作候选；执行首个命令，机器人节点在 MuJoCo owner 中步进并发布新观测；Agent 确认实际关节误差后结束，或再次规划。物体、夹爪或自然语言任务完成判定尚未实现。

若删除 planner 的动作条件预测，可比较固定目标动作与模型选出的动作，在相同关节初态下测量收敛步数、跟踪误差和模型查询延迟。当前 API Agent 只产生 joint_goal，无法衡量物体级任务恢复能力。
