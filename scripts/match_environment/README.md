# A/B 指定开局对战环境（Linux 运行，Windows 可编辑/生成）

需要 Python **3.10+**（仅标准库）、Bash、`flock`（util-linux），以及匹配引擎后端的驱动/动态库。
脚本、模型、开局和引擎放在同一个目录，整个目录可以搬到 Linux；不要在 Windows 上运行 Linux 引擎。

```text
evaluation_match_openings/
  auto_match_config.py         # 所有批量对战设置
  script_generator.py         # 生成 run_matches.sh 和 plans/<run_id>.json
  calculate_elo.py             # 将所有轮次的结果 JSON 混合计算 Elo
  match_tools.py              # 完整性校验与断点检查
  match.cfg                   # 规则、搜索参数基础配置
  engine/katago               # 支持 matchOpeningFile 的 CUDA 原生模型引擎
  engine/katago_onnx           # 可选：同样支持 matchOpeningFile 的 ONNX 后端引擎
  lib/                        # 非系统 .so 可放此处，也可配置其他路径
  openings/renju5_982.txt
  modelsGroupA/
    b14c192h6tflrs/model.bin.gz
    b16c128h4tflrs/model.bin.gz
    b16c128h4tflrs_200po/      # 为空，复用 A 组无后缀模型，200 playouts
    b16c128h4tflrs_400po/
    ...                       # 共 12 个目录
  modelsGroupB/
    ...                       # 同样 12 个目录，模型独立复制，不跨组读取
  results/<run_id>/pair_0001/ # 每场独立日志、SGF、matchresult、complete.json
```

## 使用

1. 放入模型并编辑 `auto_match_config.py`。
2. `python3 script_generator.py`（Windows 可以使用 `python script_generator.py`）。
3. 将整个目录复制到 Linux，`bash run_matches.sh`。也可在 Linux 上生成脚本。
4. 脚本按顺序完成筛选后的 A × B 对战，默认最后自动汇总**所有轮次**的 Elo；单独重算：`python3 calculate_elo.py`。

本地交付只生成脚本，**未启动正式对战**。去重开关默认开启，66 场 × 1000 局 = 66000 局。
不开启则 144 场 × 1000 局 = 144000 局。

两组各自包含以下 12 个参赛目录：

```text
b11c96h3tfrs_renju15x
b11c96h3tfrs_renju15x_200po
b11c96h3tfrs_renju15x_400po
b14c192h6tflrs
b16c128h4tflrs
b16c128h4tflrs_200po
b16c128h4tflrs_400po
b18retrain
b24c256h8tflrs
b28retrain
b28s1600
b28s5500
```

模型来源与校验和记录在 `provenance/all_models.json`；两个新增 checkpoint 使用 SWA、普通 v102，未使用 PTQ。
`b28s1600` 虽从原 `_200po` 目录取得模型，但此处不带该后缀，使用 `DEFAULT_PLAYOUTS`（当前 100）。

后台运行可用：`screen -S abmatch bash run_matches.sh`，按 Ctrl-A D 脱离。

### 分组、别名和参数

- 只枚举两组的直接子目录，按目录名排序，忽略以 `.` 开头的目录；无模型的普通目录会报错。
- 不做 A-A/B-B。`DEDUPLICATE_MATCHES = True` 时跳过 AB 同名目录，只保留同一目录名配对的一个方向；
  按目录排序保留先遇到的方向。不论保留哪一方向，该场内部仍会按开局交换黑白。
  `False` 时不处理配对列表，完整执行 A × B，包含同名对战和反向重复对战。
- 参赛名统一带 po：`DEFAULT_PLAYOUTS = 100` 时，无后缀的 `xxxx` 会记录为 `xxxx_100po`。
  默认值可以设置为任何正整数，例如设成 150，就使用 150 playouts 并记录为 `xxxx_150po`。
  显式的 `_200po`、`_400po` 不受默认值影响。模型目录本身无需重命名。
- 配对去重、Elo 合并均依据**补齐 po 后的名字**，不比较权重：即使同名模型更换权重，也按要求合并。
  若不希望混合某次新权重，使用不同目录名。运行前 SHA256 检查仍用于防止本轮模型意外改动，不用于区分 Elo。
- 引擎 botName 为 `A/xxxx_100po`、`B/xxxx_100po`；汇总时去掉 A/B，统一为 `xxxx_100po`。
  默认 100po、200po、400po 仍是三个不同参赛项，彼此可对战。
  因此这套 12×12 库不论是否去重，都输出 12 个 Elo。关闭去重时同名对战会运行并保留结果，
  但不提供相对强度信息，其局数在汇总中单独列出、不参与拟合；反向对战会合并计分。
- 末尾严格匹配 `_数字po` 的目录，必须在**同组**找到去掉这段后缀的目录。
  它总是复用无后缀目录的模型（忽略别名目录里的模型），同时覆盖该方 `maxPlayouts`；两组均支持。
  原目录自身也会参赛，使用 `DEFAULT_PLAYOUTS`。别名是独立的 Elo 参赛项。
- 模型优先顺序默认 `model.bin.gz`、`model.bin`、`model.onnx`；同时有多个时使用第一个。
  需要 ONNX 时把 `MODEL_FILE_PREFERENCE` 设成 `["model.onnx"]` 并提供 ONNX 引擎。
  当前打包的 CUDA 引擎不支持 ONNX；一个 match 不能混用原生和 ONNX 模型，生成时会明确报错。
- 默认每对 **1000 局**，使用 982 开局库的**前 500 个**，每个开局交换黑白两次。
  n 局使用前 ceil(n/2) 个，库不足则循环；奇数最后一局 A 执黑。
- `OPENING_FILE = ""`、`'""'` 或 `None` 使用旧随机开局行为，非空但不存在/无有效内容会报错。
  所有开局的落子合法性由引擎启动时严格检查。
- `VISIBLE_GPUS = [2, 3]` 表示物理 GPU 2、3；引擎使用重映射后的 0、1，NN server 线程轮流分配。
  每模型 NN server 线程数至少要覆盖可见 GPU 数量。
- `LIBRARY_PATHS` 可用 `lib`、`~/lib`、`/usr/local/cuda/lib64` 等 Linux 路径；保留已有 `LD_LIBRARY_PATH`。
  生成时不要求这些库目录在 Windows 存在，但 Bash 启动时要求它们在 Linux 存在。
- 引擎/模型/基础 cfg/开局文件使用环境目录内相对路径，运行与当前工作目录无关。
  参数通过 Bash 安全引用；引擎 override 不支持逗号、换行、`#`，所以这些字符会被拒绝。
- 专用设置和模型名称/路径不能用 `EXTRA_OVERRIDES` 再覆盖。不要在基础 cfg 添加隐藏的调度/GPU/访问数限制；
  `maxPlayouts` 别名不能消除另设的 `maxVisits`/`maxTime` 限制。
- 默认沿用 RENJU/NOVC、禁手/VCF 特征开启、graph search 开启、FP16、INT8 关闭，
  200 局并发 × 每局 1 搜索线程、B36、每模型 4 NN server、cache=24、默认 100 playouts。

### 结果和重跑

每次生成的计划包含输入文件 SHA256、配对、GPU 和对战参数。相同输入复用同一 run_id；改变参数、
模型或 `EXPERIMENT_TAG` 会得到新结果目录。`run_matches.sh` 固定引用生成当时的计划，运行前检查文件校验和。
修改配置后必须重新运行生成器。不要删除旧 `results/`；建议也保留 `plans/`。

运行时用文件锁阻止同一目录同时启动两套 sweep。已经完整通过检查的 pair 会跳过。
中断的 pair 不会被自动拼接、覆盖或计分：检查该 pair 的日志，将其整个目录**另行备份挪走**后再运行，
只重跑该 pair。不要在运行期间修改/重新生成脚本或替换资产。

### 跨轮结果直接混合算 Elo

默认命令：

```bash
python3 calculate_elo.py
```

递归读取本环境 `results/` 和 `matchresult/` 下所有对战结果 JSON，一起拟合。
你改天换模型、重新生成和跑对战后，旧结果仍会自动加入，不需要旧模型权重文件。
不是把各轮已经算出的 Elo 做平均，而是把各轮原始胜/负/和计数合并后重新拟合。
终端先列出每场对战明细（双方、胜/负/和、局数），再显示各 bot 的 Elo 排名表（名次、名字、Elo、参赛局数）。
不显示按 bot 汇总的胜率或得分率；Elo 仍使用全部对战关系联合拟合，算法不变。
输出为 **`elo_all/matches.csv`（逐场明细）、`elo_all/elo.csv`（Elo 排名）、`elo_all/elo.json`（两表及来源记录）**。
每份被采纳的结果 JSON 对应一个明细条目，多轮同一对手不会在明细中合成一行；同名自我对战明确标注不参与 Elo。
JSON 同时包含输入文件清单和跳过原因。

也可以像旧评测脚本那样，把结果 JSON 混放到一个文件夹：

```bash
python3 calculate_elo.py --results-dir mixed_results
# 或同时读取多个目录
python3 calculate_elo.py --results-dir round1_results round2_results
```

- 只按名字和 po 合并，忽略权重。旧名字没有 po 时：保留原计划则读取当时的实际 po；
  只有裸 JSON 时默认补 `_100po`，可通过 `--default-playouts 150` 更改这个旧数据兜底值。
- 本环境有计划的对战，只采纳 `complete.json` 校验通过的完整结果；未完成的跳过，已完成但损坏的报错。
  裸结果无需计划或完成标记，但必须有合法一致的胜负和计数；有 `numGamesRequested` 时还必须达到该局数。
- `elo.json`、计划和校验标记等非对战 JSON 不计入。
- 不重复扫描同一路径；有计划的结果按 run/pair 识别独立对战。
  对裸 JSON，仅“同文件名且内容完全相同”的副本去重。不同文件名的独立场次即使比分相同也照常累加。
  如确有同名同内容的独立场次，可使用 `--keep-copies`。复制后改名的裸 JSON 无法可靠识别为副本，请勿重复放入。
- 不同轮次必须通过共同参赛模型/对手连接，才能得出统一排名；完全不相连时会报错。
  只应混合你认为规则和对战条件可比较的结果，脚本按你的要求不会按规则/权重自动分组。
- 如仍需只算一轮：`python3 calculate_elo.py --plan plans/<run_id>.json`，输出仍在该轮 `results/<run_id>/`。

使用 Bradley–Terry 的 400 分尺度，和棋半分，默认全体平均 Elo=0，也可指定 `ELO_REFERENCE`。
默认每对双方各加 0.5 个虚拟胜局，避免全胜/全败导致无穷分；这个平滑不是置信区间。
Elo 只代表所选对战池中的相对强度，不是外部标定等级分。

## 开发回归测试

`python3 -m unittest discover -s tests -v`。模板源文件不附带大模型和引擎，需要自行复制资产。
可选真实引擎小样本测试：`python3 tests/cuda_smoke.py`，仅用 b14/b16 各两个低 playouts 设置，
在独立 `validation/cuda_smoke` 目录进行 4 场共 32 局；不会执行正式 66 场计划。
