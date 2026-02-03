# Continue From Checkpoint - 从历史节点继续执行

## 概述

`continue_from_checkpoint.py` 提供了从历史执行记录的特定步骤继续执行的功能。与 `resample_prm_from_log.py`（只重新采样PRM候选但不执行）不同，这个工具会**实际继续执行**任务。

## 主要特性

1. **三种继续策略**：
   - `replace_from`: 丢弃指定步骤及之后的所有步骤，从该点重新执行
   - `continue_after`: 保留到指定步骤（包含），从下一步继续
   - `branch_from`: 保留指定步骤之前的内容，创建分支执行

2. **状态恢复**：自动从历史步骤中提取并恢复 `progress_state`

3. **灵活配置**：
   - 支持指定额外执行步数
   - 支持任务ID过滤
   - 支持断点续传（resume模式）
   - 可启用/禁用PRM

## 使用方法

### 基本用法

```bash
# 从步骤3开始替换执行（默认策略）
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input existing_output.jsonl \
    --output continued_output.jsonl \
    --continue_from_step 3
```

### 使用Shell脚本

```bash
# 方式1: 使用默认配置
bash scripts/run_continue_from_checkpoint.sh input.jsonl output.jsonl 3

# 方式2: 启用PRM
USE_PROCESS_REWARD=true bash scripts/run_continue_from_checkpoint.sh input.jsonl output.jsonl 3

# 方式3: 指定不同策略
bash scripts/run_continue_from_checkpoint.sh input.jsonl output.jsonl 3 continue_after
```

### 高级用法示例

#### 1. 从步骤5继续，但只执行3个额外步骤

```bash
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input input.jsonl \
    --output output.jsonl \
    --continue_from_step 5 \
    --max_additional_steps 3
```

#### 2. 只处理特定任务ID

```bash
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input input.jsonl \
    --output output.jsonl \
    --continue_from_step 3 \
    --task_ids "task001,task003,task005"
```

#### 3. 断点续传模式（跳过已处理的任务）

```bash
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input input.jsonl \
    --output output.jsonl \
    --continue_from_step 3 \
    --resume
```

#### 4. 使用branch_from策略创建多个分支

```bash
# 创建分支1：从步骤2开始的新路径
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input original.jsonl \
    --output branch1.jsonl \
    --continue_from_step 2 \
    --strategy branch_from

# 创建分支2：从步骤3开始的新路径
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input original.jsonl \
    --output branch2.jsonl \
    --continue_from_step 3 \
    --strategy branch_from
```

#### 5. 启用PRM进行继续执行

```bash
# 通过环境变量启用PRM
export USE_PROCESS_REWARD=true
export RPM_MODEL_URL="claude:claude-3.7"
export NUM_ACTION_CANDIDATES=5

python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input input.jsonl \
    --output output_with_prm.jsonl \
    --continue_from_step 3 \
    --enable_prm
```

## 三种策略详解

### 1. replace_from（替换策略）

**场景**：发现某一步出错，想要重新执行该步骤及之后的内容

```
原始执行: Step0 -> Step1 -> Step2 -> Step3(错误) -> Step4 -> Step5
继续执行: Step0 -> Step1 -> Step2 -> Step3(新) -> Step4(新) -> ...
```

**命令**：
```bash
--continue_from_step 3 --strategy replace_from
```

**说明**：
- 保留 Step 0, 1, 2
- 丢弃 Step 3, 4, 5
- 从 Step 3 开始重新执行

### 2. continue_after（接续策略）

**场景**：执行因超时等原因中断，想要从中断点继续

```
原始执行: Step0 -> Step1 -> Step2 -> Step3(最后一步)
继续执行: Step0 -> Step1 -> Step2 -> Step3 -> Step4(新) -> Step5(新) -> ...
```

**命令**：
```bash
--continue_from_step 3 --strategy continue_after
```

**说明**：
- 保留所有 Step 0, 1, 2, 3
- 从 Step 4 开始添加新步骤

### 3. branch_from（分支策略）

**场景**：想要尝试不同的执行路径，探索替代方案

```
原始执行:  Step0 -> Step1 -> Step2 -> Step3 -> Step4
分支1执行: Step0 -> Step1 -> Step2 -> Step3(新路径1) -> ...
分支2执行: Step0 -> Step1 -> Step2 -> Step3(新路径2) -> ...
```

**命令**：
```bash
--continue_from_step 3 --strategy branch_from
```

**说明**：
- 保留 Step 0, 1, 2
- 丢弃 Step 3, 4
- 从 Step 3 开始探索新路径

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--input` | ✓ | 输入JSONL文件，包含历史session |
| `--output` | ✓ | 输出JSONL文件，保存继续执行的结果 |
| `--continue_from_step` | ✓ | 从哪个步骤继续（0-indexed） |
| `--strategy` | | 继续策略：replace_from(默认)/continue_after/branch_from |
| `--max_additional_steps` | | 最多执行多少额外步骤 |
| `--task_ids` | | 逗号分隔的任务ID列表（只处理这些任务） |
| `--limit` | | 限制处理的任务数量 |
| `--resume` | | 启用断点续传，跳过已存在的任务 |
| `--enable_prm` | | 启用Process Reward Model |

## 环境变量配置

### 模型配置
```bash
export LLM_URL="gpt:gpt-4o"                    # 主模型
export RPM_MODEL_URL="claude:claude-3.7"       # PRM模型
```

### PRM配置
```bash
export USE_PROCESS_REWARD=true                 # 启用PRM
export NUM_ACTION_CANDIDATES=3                 # 候选动作数量
export SAMPLING_TEMPERATURE=0.0                # 采样温度
export MIN_SCORE_THRESHOLD=0.3                 # 最低分数阈值
```

### Planning PRM配置
```bash
export ENABLE_PLANNING_PRM=true                # 启用planning PRM
export PLANNING_SCORE_WEIGHT=0.7               # planning分数权重
export DIVERSITY_WEIGHT=0.3                    # 多样性权重
```

### 多样化采样配置
```bash
export ENABLE_DIVERSE_SAMPLING=true            # 启用多样化采样
export SEQUENTIAL_MODE=true                    # 顺序模式
export DIVERSITY_THRESHOLD=0.4                 # 多样性阈值
export MAX_SAMPLING_ATTEMPTS=3                 # 最大采样尝试次数
```

## 实际应用场景

### 场景1：错误修正

假设发现第4步的web搜索使用了错误的关键词：

```bash
# 从第4步重新开始，使用replace_from策略
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input gaia_output_error.jsonl \
    --output gaia_output_fixed.jsonl \
    --continue_from_step 4 \
    --strategy replace_from
```

### 场景2：中断恢复

任务因为超时或网络问题在第7步中断：

```bash
# 从第7步之后继续，使用continue_after策略
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input interrupted.jsonl \
    --output resumed.jsonl \
    --continue_from_step 7 \
    --strategy continue_after \
    --max_additional_steps 5
```

### 场景3：A/B测试不同路径

对同一个任务，从第3步开始尝试不同的方法：

```bash
# 路径A：使用web_agent
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input base.jsonl \
    --output path_a.jsonl \
    --continue_from_step 3 \
    --strategy branch_from

# 路径B：使用file_agent
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input base.jsonl \
    --output path_b.jsonl \
    --continue_from_step 3 \
    --strategy branch_from
```

### 场景4：批量重试失败的任务

```bash
# 第一次运行：处理所有任务
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input all_tasks.jsonl \
    --output first_attempt.jsonl \
    --continue_from_step 0

# 第二次运行：只重试失败的任务
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input all_tasks.jsonl \
    --output retry.jsonl \
    --continue_from_step 0 \
    --resume  # 跳过已成功的任务
```

## 输出格式

输出的JSONL文件会包含额外字段：

```json
{
  "id": "task001",
  "task": "原始任务描述",
  "session": {
    "steps": [...],  // 包含保留的历史步骤 + 新执行的步骤
    "info": {...}
  },
  "continued_from_step": 3,           // 新增：从哪步继续的
  "continuation_strategy": "replace_from",  // 新增：使用的策略
  "continuation_error": "..."         // 新增：如果失败，错误信息
}
```

## 与 resample_prm_from_log.py 的区别

| 特性 | continue_from_checkpoint.py | resample_prm_from_log.py |
|------|----------------------------|--------------------------|
| **是否执行** | ✓ 实际执行代码 | ✗ 仅重新采样PRM |
| **修改session** | ✓ 添加新的执行步骤 | ✗ 只添加PRM信息 |
| **时间消耗** | 长（需要真实执行） | 短（只调用PRM） |
| **用途** | 修正错误、继续中断任务 | 分析候选、生成训练数据 |
| **状态恢复** | ✓ 恢复progress_state | 仅用于PRM上下文 |

## 技术实现细节

### 1. 状态恢复机制

```python
# 从最后一个历史步骤中提取state
def extract_state_from_steps(steps, step_idx):
    for i in range(step_idx, -1, -1):
        if "plan" in steps[i] and "state" in steps[i]["plan"]:
            return steps[i]["plan"]["state"]
    return {}
```

### 2. Session重建

```python
# 使用AgentSession.init_from_data创建session
session = AgentSession.init_from_data(
    task=task,
    steps=kept_steps,  # 根据策略保留的步骤
    **session_info
)
```

### 3. 继续执行

```python
# agent.run会检测到session已有步骤，自动从当前点继续
result_session = agent.run(
    task=task,
    session=session,  # 传入已有步骤的session
    max_steps=additional_steps
)
```

## 注意事项

1. **步骤索引从0开始**：第一步是step 0，第二步是step 1，以此类推

2. **max_steps计算**：
   - 默认：`agent.max_steps - len(kept_steps)`
   - 自定义：使用 `--max_additional_steps`

3. **任务一致性**：确保输入文件中的任务描述与原始执行时一致

4. **状态依赖**：继续执行依赖于正确的`progress_state`恢复，如果历史步骤中没有state信息，将从空state开始

5. **工具可用性**：确保sub-agents和tools在继续执行时仍然可用（如web环境、文件路径等）

## 故障排查

### 问题1：State恢复失败

```
[continue] No state found in history, using empty state
```

**原因**：历史步骤中没有保存state信息

**解决**：
- 从更早的步骤继续（那里可能有state）
- 或接受空state（agent会重新构建）

### 问题2：任务ID不匹配

```
[continue] Task mismatch: session.task='...' vs provided task='...'
```

**原因**：输入文件中的task与session中保存的不一致

**解决**：检查输入文件，确保任务描述正确

### 问题3：步骤索引超出范围

```
[continue] Invalid continue_from_step=10 for 5 steps
```

**原因**：指定的步骤号大于实际步骤数

**解决**：检查历史session有多少步，使用有效的索引

## 完整示例工作流

```bash
# 1. 初始运行（假设在第5步遇到问题）
bash scripts/run_gaia_with_planning.sh \
    --input gaia_dev.jsonl \
    --output initial_run.jsonl

# 2. 检查结果，发现task003在第5步出错

# 3. 仅对task003从第5步重新执行
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input initial_run.jsonl \
    --output task003_fixed.jsonl \
    --continue_from_step 5 \
    --strategy replace_from \
    --task_ids task003

# 4. 启用PRM再试一次，看是否能获得更好的结果
export USE_PROCESS_REWARD=true
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input initial_run.jsonl \
    --output task003_prm.jsonl \
    --continue_from_step 5 \
    --strategy replace_from \
    --task_ids task003 \
    --enable_prm

# 5. 比较两个结果，选择更好的
python visualize_comparison.py task003_fixed.jsonl task003_prm.jsonl
```

## 参考

- 相关脚本：`resample_prm_from_log.py` - 重新采样PRM候选
- 可视化工具：`prm_visualizer_server.py` - 查看执行过程
- 主执行脚本：`main.py` - 标准执行入口







