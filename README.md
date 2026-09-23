# ArenaAuto

基于 **ADB + OpenCV + OCR** 的《胜利者的竞技场》桌面自动化工具。

支持：自动选择战力阈值内的对手 → 备战「去获胜」→ 等待结算 → 退出 → 检查挑战次数（次数为 0 时先选对手点击，由购买弹窗点「购买」）→ 循环。内置状态机、安全恢复、YAML 配置、PySide6 GUI、调试截图与 PyInstaller 打包。

内置**自动标定系统**：一键截图分析生成 1~5 对手 ROI、点击点、挑战次数与按钮配置，支持半自动拖动修正与手动 YAML 兜底（详见第 8 节）。

---

## 1. 项目介绍

- 通过 ADB 控制 Android 设备/模拟器
- OCR 识别 1~5 号对手战斗力（支持多引擎：PaddleOCR / Tesseract / EasyOCR）
- 按 **1→5 顺序**选择第一个 `power <= threshold` 的对手（不排序、不随机）
- Template Matching + OCR Fallback 识别「去获胜 / 退出 / 购买 / 确认 / 胜利 / 失败」
- 明确状态机 + 超时 + 异常恢复，识别失败**绝不**瞎点
- 调试截图、ROI 可视化、模板制作工具、YAML 配置、运行统计

## 2. 环境要求

- Windows 10/11
- Python 3.9+（建议 3.10/3.11）
- Android platform-tools（adb）
- 已开启 USB/Wi-Fi 调试的手机或模拟器

## 3. 安装 Python

从 https://www.python.org/downloads/ 安装 3.9+，勾选 **Add python.exe to PATH**。

## 4. 安装 ADB

1. 下载 [platform-tools](https://developer.android.com/tools/releases/platform-tools)
2. 解压后把目录加入 PATH，或在 `config/config.yaml` 写绝对路径：

```yaml
adb:
  executable: "D:/Android/platform-tools/adb.exe"
```

验证：

```bash
adb version
adb devices
```

## 5. 开启 Android 调试

1. 开发者选项 → 打开 **USB 调试**
2. USB 连接，或无线：`adb connect <ip>:5555`
3. `adb devices` 显示 `device` 状态即可

## 6. 安装依赖

```bash
cd ArenaAuto
python -m pip install -U pip
pip install -r requirements.txt
```

> PaddleOCR 首次运行会下载模型，需网络。若 Windows/Python 组合不兼容，程序会自动回退到其他 OCR 或进入安全空 OCR 模式（战力识别失败并走恢复流程，**不会**误点）。

仅开发测试：

```bash
pip install -r requirements-dev.txt
```

## 7. 配置 YAML

配置文件：`config/config.yaml`（首次启动若不存在会自动创建默认值）。

关键项：

| 键 | 说明 |
|---|---|
| `automation.power_threshold` | 战力阈值 |
| `automation.dry_run` | true = 只打日志不点击 |
| `battle.poll_interval` | 战斗中检测间隔（秒） |
| `purchase.*` | 是否自动购买挑战次数；`buy_button`/`confirm_button` 为标定几何 |
| `opponents[].power_region` | 战力 ROI `[x,y,width,height]` |
| `opponents[].click` | 对手点击坐标 `[x,y]` |
| `opponents[].source` | 来源：`default/ocr/structure/manual/calibrated` |
| `opponents[].confidence` | 标定置信度 0~1 |
| `challenge_count.region` | 挑战次数 ROI（与旧 `challenge_region` 兼容） |
| `screen.reference_*` | 参考分辨率（用于缩放） |
| `templates.*` | 按钮模板路径与阈值 |
| `version` | 配置版本（当前 2，旧版自动迁移） |
| `calibration.*` | 标定参数：padding、weights、阈值（见下节） |
| `buttons.*` | 标定按钮几何：`region/click/confidence/source` |
| `result.*` | 结算页 `victory/defeat/exit` 几何 |

**当前仓库不包含真实游戏坐标**，默认 `power_region` / `click` 为占位值 `[0,0,100,50]` / `[0,0]`，必须按你的设备截图修改或使用自动标定。点击坐标为 `(0,0)` 时程序会拒绝点击并进入恢复流程（防误触）。

配置来源优先级：

```text
manual（手动修改，confidence=1.0）
  > calibrated（自动标定）
  > default（内置默认）
```

手动修改过的区域（`source: manual`）在重新标定时不会被覆盖。

## 8. 自动标定（自动配置）

三种配置模式，最终生成相同的数据结构：

| 模式 | 入口 | 说明 |
|---|---|---|
| 自动标定 | 主界面 `[自动配置]` | 截图 → 自动分析 → 用户确认 → 保存 |
| 半自动修正 | 向导内 `[调整]` | 拖动 ROI / 点击点 / 重新分析后保存（`source=manual`） |
| 手动配置 | 主界面 `[手动配置]` | 直接编辑 YAML，程序打开 `config/config.yaml` |

### 使用方法

1. 连接 Android，打开游戏**竞技场页面**
2. 点击 `[自动配置]`（或 `[重新标定]` 直接进入并允许替换）
3. 按向导 7 步确认：

```text
步骤 1：连接设备（单设备自动选中）
步骤 2：确认竞技场页面（可手动点「我已确认」）
步骤 3：分析对手列表（1~5 战力 ROI + 点击点 + 置信度）
步骤 4：确认战力区域（可拖动调整）
步骤 5：分步检测按钮（按页面依次检测，结果合并）
        ① 竞技场页 → 去获胜/退出/购买挑战次数
        ② 购买弹窗（点对手后自动弹出）→ 购买/确认
        ③ 胜利结算 → 胜利/退出
        ④ 失败结算 → 失败/退出（可选）
步骤 6：测试识别（重新截图校验，禁止点击）
步骤 7：保存配置
```

按钮列表三态：`✓ 已检出` / `○ 已检测未找到（可到其它页面重试）` / `· 未检测（需切换到对应页面）`。每页只检测该页应有的按钮，跨页结果自动合并，不会互相覆盖。

4. 置信度 ≥0.90 绿色、≥0.75 黄色、否则红色 ⚠ 并要求 `[调整]`
5. 保存时若已存在 `config.yaml`，默认 **另存为** `config/config.calibrated.yaml`（可选覆盖/取消，不静默覆盖）

### 主界面按钮

```text
[自动配置]   打开标定向导
[重新标定]   直接进入向导，允许替换配置
[测试配置]   重新截图识别竞技场/1~5 战力/挑战次数/去获胜/退出，绝不点击
[手动配置]   用系统编辑器打开 config.yaml
```

### 自动生成的 YAML（示例）

```yaml
version: 2

screen:
  reference_width: 1920
  reference_height: 1080

opponents:
  - id: 1
    power_region: [1200, 300, 220, 70]
    click: [1050, 335]
    source: calibrated
    confidence: 0.96

challenge_count:
  region: [500, 800, 150, 50]

buttons:
  go_win:
    region: [1600, 900, 250, 100]
    click: [1725, 950]
    confidence: 0.98
    source: calibrated

purchase:
  buy_button: { region: [1000, 500, 300, 100], click: [1150, 550] }
  confirm_button: { region: [1000, 600, 300, 100], click: [1150, 650] }

result:
  victory: { region: [500, 400, 300, 150] }
  defeat: { region: [500, 400, 300, 150] }
  exit: { region: [1600, 900, 250, 100], click: [1725, 950] }
```

坐标一律先按**实际截图**检测，保存时换算到 `screen.reference_*` 参考分辨率，运行时再按设备实际分辨率缩放（支持 720p/1080p/1440p 等同比例分辨率）。

### 重新标定

- 游戏 UI 变化、换分辨率、换设备 → 点 `[重新标定]`
- 单个区域微调：向导步骤 3/4/5 点 `[调整]`，拖动后该项 `source=manual`（优先级最高）
- `[全部重新标定]` 才会覆盖手动区域

### 调试文件

标定过程写入 `debug/calibration/`：

```text
original.png / ocr_overlay.png / opponent_detection.png
power_regions.png / buttons.png / challenge_count.png / final_calibration.png
```

自动生成的模板写入 `resources/templates/generated/`（不覆盖 `resources/templates/` 原模板）；运行时模板匹配失败会回退 OCR → 结构检测 → 安全停止，**绝不**因模板失败而固定坐标点击。

## 9. 准备模板

将按钮截图裁剪为 PNG，放到：

```text
resources/templates/
├── go_win.png
├── exit.png
├── buy_challenge.png
├── confirm_buy.png
├── victory.png
└── defeat.png
```

或使用 GUI「开发/调试 → 从当前截图创建模板」。模板缺失时程序仍可启动，GUI 会提示；按钮识别会尝试 OCR 文案 fallback。

## 10. 启动程序

任选其一（推荐直接双击 `start.bat`）：

```bat
:: Windows 双击或命令行
start.bat
```

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

```bash
python start.py
```

开发环境手动启动：

```bash
python main.py
```

或：

```bash
set PYTHONPATH=src
python -m arena_auto
```

或：

```bash
python scripts/run.py
```

启动脚本会：定位 Python（含 `.venv`）→ 创建 `logs/debug/data` 目录 → 检查依赖（缺失则自动 `pip install`）→ 设置 `PYTHONPATH` → 运行程序。

## 11. 第一次运行流程

1. 连接设备 → 打开程序 → 「刷新设备」选择 serial  
2. 截图确认游戏停留在**竞技场页面**  
3. 点击 **[自动配置]**，按向导确认对手/战力/按钮并保存（见第 8 节）  
4. 或手动：用「测试竞技场检测 / 测试OCR / 测试战力ROI」校验识别，修改 `power_region`、`click`、`challenge_count.region`  
5. 点 **[测试配置]** 校验当前配置（不点击）  
6. 建议先勾选 **测试模式（dry_run）** 跑一轮，确认日志里的点击坐标  
7. 取消 dry_run → 「开始运行」

## 12. 调试方法

- **实时截图**：刷新截图 / 1 FPS 预览  
- **开发/调试**：OCR、竞技场、战力 ROI、挑战次数、各按钮模板测试；绿框为识别框  
- **自动标定**：`debug/calibration/` 下 7 张过程截图（见第 8 节）  
- **日志**：GUI 日志页 + `logs/arena_auto.log`（标定日志前缀 `[CALIBRATION]`）  
- **调试截图**：`debug/arena|prepare|battle|result|purchase|failed|recovery/calibration/`  
- **OCR 失败**：`debug/failed/power_*.png` 等  
- 统计：`data/statistics.json`

## 13. OCR 调整

```yaml
ocr:
  provider: paddle   # paddle | tesseract | easyocr
  device: auto       # auto | cpu | cuda（EasyOCR）
  scale: 3           # ROI 放大倍数
  threshold: true    # 二值化
  whitelist: "0123456789,"
  timeout_ms: 1500   # 软超时预算（超时仅记日志，不中断线程）
  cache_enabled: true
  cache_difference_threshold: 3.0
  power_allowlist: "0123456789,."
  challenge_allowlist: "0123456789/"
recognition:
  stable_reads: 2    # 连续一致才确认
  retry_count: 3     # 整帧重试次数
logging:
  performance: false # true → [PERF] + logs/performance.log
```

识别失败返回 `OCR_FAILED`（`None`），**不会**当成 0。

### 13.1 OCR 性能要点

- **共享 Reader**：进程内只创建一次 EasyOCR Reader（`OcrEngine` / `EasyOCRProvider`）。
- **懒加载**：`create_ocr_provider` 立即返回 facade，模型在自动化工作线程首次识别时加载（GUI 显示 `OCR 初始化中...`）。
- **ROI + allowlist**：战力/挑战次数优先裁剪 ROI + 数字白名单，避免整图 OCR。
- **ROI 缓存**：同一 ROI 几乎不变时直接命中缓存（约 0.1ms）。
- **模板优先**：页面/按钮先走 OpenCV 模板（同分辨率 conf≈1.0 时跳过多尺度金字塔）；失败再 OCR。
- **单帧扇出**：`PowerReader.read_all` 一张截图读全部 5 个战力 ROI；仅不稳定/失败槽位重试截图。
- **合并识别**：`detect_screen_and_outcome` 一次整图 OCR 同时得到页面 + 胜负。
- **测速**（本机实测，勿编造）：

```bash
python scripts/benchmark_ocr.py --repeat 3
```

  典型结果（本机 CPU / 1600×720，仅供对照）：缓存命中 ~0.1ms；命中模板 ~40–60ms；战力 ROI 单次约数百 ms；整图 EasyOCR CPU 数秒（无 CUDA 时 `device: auto`→cpu）。

`timeout_ms` 为**软**预算：超出只打 `OCR timeout budget exceeded` 并计入 stats，不杀线程（避免死锁）。

## 14. 坐标调整

1. 参考分辨率写进 `screen.reference_width/height`（例如 1920x1080）  
2. 实际设备分辨率不同时，ROI/点击会按比例缩放  
3. 模板匹配默认多尺度 0.8~1.2  
4. 优先用 **自动标定** 生成坐标（第 8 节）  
5. 或用调试页框选功能读取 ROI，再写入 YAML

## 15. PyInstaller 打包

```bash
pip install pyinstaller
python scripts/build.py
# 或
pyinstaller ArenaAuto.spec
```

产物：`dist/ArenaAuto/`（含 `ArenaAuto.exe`）。  
`resources/` 与 `config/` 会打进包内；用户可改的 `config/config.yaml`、模板、日志、debug、data 在 exe 同级外部目录生成。

## 16. 测试

```bash
pip install pytest
pytest tests -q
```

覆盖：战力解析、坐标缩放、配置校验、状态机、对手选择、模板匹配与页面识别、购买流程（次数为 0 → 选对手 → 弹窗购买），以及自动标定（ROI 扩展、坐标映射、1~5 对手检测、结构推导、YAML v2 迁移、按钮检测、验证流程不点击）。

## 17. 安全设计（重要）

```text
识别失败 ≠ 默认成功
识别失败 ≠ 默认失败
识别失败 ≠ 默认点击
```

- 对手战力 OCR 失败 → 不选择、进恢复  
- 未找到按钮 → 不点击固定坐标  
- 未知页面 → 尝试 BACK（可配置），超过 `recovery.max_attempts` → ERROR 停止  
- 每个状态有 timeout  
- 停止按钮使用 `threading.Event`，长等待可中断  
- 购买有 `max_per_session` 上限，购买后必须 OCR 确认次数恢复  
- 挑战次数为 0 时：**先读战力并点击对手** → 游戏弹出购买弹窗 → 点「购买」→（如有）点「确认」→ OCR 校验次数恢复；未弹窗则超时进恢复，不会在竞技场页盲找购买按钮  
- 自动标定 / 测试配置全程**只读**（截图+识别），绝不点击  
- 标定失败不会写入错误坐标；低置信度必须提示用户调整  

## 18. 常见问题

### ADB 连接失败

1. `adb kill-server` → `adb start-server`  
2. 确认 `adb devices` 为 `device` 而非 `unauthorized/offline`  
3. 配置里写 adb 绝对路径  
4. 模拟器多开时选对 serial  
5. 程序内置 `reconnect_attempts` 自动重连  

### OCR 识别失败

1. 看 `debug/failed/power_*.png` 是否裁剪正确  
2. 调整 `power_region` 与 `ocr.scale`  
3. 切换 `ocr.provider`  
4. 确认文字清晰、无动画遮挡  
5. 保持 `stable_reads>=2` 防抖  

### 模板匹配失败

1. 模板应从**同一设备同一分辨率**截取  
2. 降低 `threshold`（如 0.75）  
3. 重新制作更干净的按钮 PNG  
4. 依赖 OCR fallback（文案在 `profile.*_text`）  

### 自动标定不准

1. 看 `debug/calibration/*.png` 定位是 OCR 还是结构问题  
2. 向导内对低置信度项 `[调整]` 拖动修正（记为 `manual`）  
3. 确认停留在竞技场页面、无动画遮挡后 `[重新分析]`  
4. 换分辨率/换 UI 后点 `[重新标定]`  
5. 调整 `calibration.*`（padding、weights、阈值）后重跑  

### 程序停在 RECOVERY

1. 看 `debug/recovery/` 当时画面  
2. 手动回到竞技场再启动  
3. 增大 `timeouts.*`  
4. 确认游戏语言与 `profile` 关键词一致  

### 打包后找不到配置/模板

- 用户数据一律在 **exe 同级** `config/`、`resources/templates/`、`logs/`、`debug/`、`data/`  
- 不要只拷贝 exe，保留同级生成的目录  

## 19. 目录结构

```text
ArenaAuto/
├── main.py
├── ArenaAuto.spec
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── LICENSE
├── config/config.yaml            # 手动/运行配置
├── config/config.calibrated.yaml # 自动标定默认另存路径
├── resources/templates/          # 手工模板
├── resources/templates/generated/ # 自动标定生成的模板
├── debug/calibration/            # 标定过程截图
├── scripts/run.py
├── scripts/build.py
├── src/arena_auto/
│   ├── app.py
│   ├── paths.py
│   ├── adb/controller.py
│   ├── automation/{controller,state_machine,selection,states}.py
│   ├── recognition/{ocr,template,screen,detector}.py
│   ├── config/manager.py         # 含 ConfigMigrator v1→v2
│   ├── calibration/              # 自动标定（新增）
│   │   ├── wizard.py             # 7 步向导 + CalibrationCanvas
│   │   ├── analyzer.py           # 分析编排 + 验证 + 可视化
│   │   ├── screen_analyzer.py    # 页面/OCR 分析
│   │   ├── opponent_detector.py  # 1~5 对手多特征评分检测
│   │   ├── button_detector.py    # 按钮 OCR 关键词检测
│   │   ├── roi_detector.py       # 战力 ROI / 挑战次数
│   │   ├── template_generator.py # generated/ 模板生成
│   │   ├── coordinate_mapper.py  # 实际像素 ↔ 参考分辨率
│   │   └── models.py             # Calibration* 数据模型
│   ├── logging/logger.py
│   ├── debug/recorder.py
│   ├── models/{config,recognition,state}.py
│   └── gui/{main_window,widgets,workers}.py
├── tests/                        # 含 test_calibration.py
└── build/
```

## 20. 后续扩展

架构预留：`ArenaProfile`（多游戏关键词）、多设备、多竞技场配置、更多 OCR Provider、定时运行、统计导出等。请勿把单一游戏文案硬编码进底层模块。

## 21. 免责声明

本工具仅供学习与个人效率研究。使用自动化脚本可能违反游戏服务条款，请自行承担风险。
