# 项目名称

**ArenaAuto — 基于 ADB + OpenCV + OCR 的《胜利者的竞技场》自动化工具**

---

# 一、项目目标

请从零生成一个**完整、可运行、可编译、可使用 PyInstaller 打包成 Windows EXE** 的桌面自动化工具。

工具用于通过 **ADB 控制 Android 设备**，自动完成游戏《胜利者的竞技场》的竞技场挑战流程。

核心功能：

1. 通过 ADB 连接 Android 设备。
2. 获取 Android 当前屏幕截图。
3. 识别竞技场页面中的 1~5 名对手战斗力。
4. 根据用户设置的战力阈值，从 1~5 中选择第一个战力不超过阈值的对手。
5. 点击对应对手。
6. 进入攻击编队页面后识别并点击「去获胜」。
7. 开始战斗后每 5 秒检测一次是否进入结算页面。
8. 识别胜利/失败结果。
9. 点击「退出」返回竞技场。
10. 检查剩余挑战次数。
11. 挑战次数不足时，根据配置自动购买挑战次数。
12. 循环执行。
13. 如果页面识别失败、ADB 断开、游戏卡死、页面异常等情况，不允许无脑继续点击，必须进入安全恢复流程。
14. 所有关键行为写入日志。
15. 支持调试截图。
16. 支持 YAML 配置。
17. 支持 GUI 修改主要配置。
18. 支持 Windows PyInstaller 打包。
19. 项目必须是真正可运行的工程，而不是示例代码或伪代码。

---

# 二、重要原则

## 1. 不要把所有逻辑写进一个 Python 文件

必须采用模块化设计。

推荐：

```text
ArenaAuto/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── config/
│   ├── config.yaml
│   └── templates/
├── src/
│   └── arena_auto/
│       ├── __init__.py
│       ├── app.py
│       │
│       ├── adb/
│       │   ├── __init__.py
│       │   └── controller.py
│       │
│       ├── automation/
│       │   ├── __init__.py
│       │   ├── controller.py
│       │   ├── state_machine.py
│       │   └── states.py
│       │
│       ├── recognition/
│       │   ├── __init__.py
│       │   ├── ocr.py
│       │   ├── template.py
│       │   ├── screen.py
│       │   └── detector.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── manager.py
│       │
│       ├── logging/
│       │   ├── __init__.py
│       │   └── logger.py
│       │
│       ├── debug/
│       │   ├── __init__.py
│       │   └── recorder.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── recognition.py
│       │   └── state.py
│       │
│       └── gui/
│           ├── __init__.py
│           ├── main_window.py
│           ├── widgets.py
│           └── workers.py
│
├── resources/
│   └── templates/
│       ├── go_win.png
│       ├── exit.png
│       ├── buy_challenge.png
│       ├── confirm_buy.png
│       ├── victory.png
│       └── defeat.png
│
├── scripts/
│   ├── run.py
│   └── build.py
│
└── build/
```

可以根据实际实现调整目录，但必须保持类似的模块化结构。

---

# 三、技术栈

必须使用：

* Python 3.9+
* PySide6
* OpenCV
* NumPy
* PyYAML
* ADB
* OCR

OCR 优先选择：

**PaddleOCR**

如果 PaddleOCR 在 Windows/Python 3.9 环境下存在兼容性问题，必须设计 OCR Provider 抽象层，使以后可以替换为：

* PaddleOCR
* Tesseract
* EasyOCR

不要让核心逻辑直接依赖某一个 OCR 实现。

推荐：

```python
class OCRProvider:
    def recognize(self, image):
        raise NotImplementedError
```

然后：

```text
PaddleOCRProvider
TesseractOCRProvider
```

---

# 四、ADB 模块

实现：

```python
class AdbController:
    def list_devices(self):
        ...

    def connect(self, serial=None):
        ...

    def disconnect(self):
        ...

    def is_connected(self):
        ...

    def screenshot(self):
        ...

    def tap(self, x, y):
        ...

    def swipe(self, x1, y1, x2, y2, duration):
        ...

    def keyevent(self, key):
        ...

    def shell(self, command):
        ...
```

ADB 可以通过：

```text
adb.exe
```

执行。

不要假设用户一定把 ADB 加入 PATH。

配置允许：

```yaml
adb:
  executable: "adb"
```

也允许填写：

```yaml
adb:
  executable: "D:/Android/platform-tools/adb.exe"
```

如果找不到 adb：

GUI 必须显示明确错误。

---

# 五、截图

使用：

```text
adb exec-out screencap -p
```

直接获取 PNG。

不要频繁创建临时文件。

优先：

```python
subprocess.run(...)
```

直接读取 stdout 并转换成 OpenCV/Numpy 图像。

统一内部图像格式：

```text
numpy.ndarray
BGR
```

---

# 六、ADB 点击

实现：

```text
adb shell input tap X Y
```

例如：

```python
adb.tap(1000, 500)
```

必须增加：

* 点击前连接检查
* 点击异常处理
* 日志记录
* 可选点击延迟

配置：

```yaml
automation:
  click_delay: 0.3
```

---

# 七、竞技场页面

游戏竞技场主界面结构：

顶部：

```text
胜利者的竞技场
```

左侧：

```text
个人信息
排名
奖励
可挑战次数 4 / 5
防御卡组
```

右侧：

```text
请选择对手
```

下面有 1~5 名对手。

每个对手包含：

* 玩家名称
* 排名
* 战斗力
* 角色阵容
* 点击区域

---

# 八、对手选择逻辑

必须严格按照：

```text
1 → 2 → 3 → 4 → 5
```

的顺序判断。

例如：

```text
阈值 = 2,000,000

1 = 2,500,000
2 = 1,800,000
3 = 1,900,000
4 = 2,100,000
5 = 1,500,000
```

程序必须选择：

```text
2
```

因为 2 是第一个：

```text
power <= threshold
```

的目标。

不要选择战力最低的目标。

不要排序。

不要随机选择。

---

# 九、战力 OCR

每个目标必须支持独立 ROI。

例如：

```yaml
opponents:
  - id: 1
    power_region: [x, y, width, height]
    click: [x, y]

  - id: 2
    power_region: [x, y, width, height]
    click: [x, y]

  - id: 3
    power_region: [x, y, width, height]
    click: [x, y]

  - id: 4
    power_region: [x, y, width, height]
    click: [x, y]

```

注意：

`power_region` 使用：

```text
x, y, width, height
```

而不是：

```text
x1, y1, x2, y2
```

内部自动转换。

---

# 十、OCR 图像预处理

战力 OCR 必须提供预处理：

```text
原图
 ↓
ROI
 ↓
放大
 ↓
灰度
 ↓
对比度增强
 ↓
二值化
 ↓
OCR
```

可以尝试：

* grayscale
* resize 2x/3x
* threshold
* adaptive threshold
* sharpen
* morphology

但是不要盲目处理。

必须允许配置：

```yaml
ocr:
  scale: 3
  threshold: true
  whitelist: "0123456789,"
```

战力最终转换：

```text
1,234,567
```

→

```text
1234567
```

同时处理：

```text
1 234 567
1.234.567
1234567
```

---

# 十一、OCR 容错

OCR 可能出现：

```text
1,234,567
1 234 567
1.234.567
I234567
l234567
```

必须进行合理清洗。

但是：

**不要在 OCR 无法确定时猜一个数字。**

如果识别失败：

```text
OCR_FAILED
```

而不是：

```text
power = 0
```

否则会导致程序错误选择对手。

---

# 十二、连续识别确认

对战力识别必须支持确认机制。

例如：

```text
第一次识别：
2,001,234

第二次识别：
2,001,234
```

一致后才确认。

如果：

```text
2,001,234
20,012,34
```

不一致，则重新截图。

配置：

```yaml
recognition:
  stable_reads: 2
  retry_count: 3
```

---

# 十三、按钮识别

按钮优先使用 OpenCV Template Matching。

实现：

```python
class TemplateRecognizer:
    def find(self, screenshot, template_name, threshold):
        ...
```

返回：

```python
DetectionResult(
    found=True,
    x=...,
    y=...,
    width=...,
    height=...,
    confidence=...
)
```

点击位置默认：

```text
模板中心
```

---

# 十四、按钮模板

支持：

```text
resources/templates/
├── go_win.png
├── exit.png
├── buy_challenge.png
├── confirm_buy.png
├── victory.png
└── defeat.png
```

模板不存在时：

GUI 必须提示。

不要让程序直接崩溃。

---

# 十五、模板匹配阈值

配置：

```yaml
templates:
  go_win:
    path: "resources/templates/go_win.png"
    threshold: 0.82

  exit:
    path: "resources/templates/exit.png"
    threshold: 0.82

  victory:
    path: "resources/templates/victory.png"
    threshold: 0.80

  defeat:
    path: "resources/templates/defeat.png"
    threshold: 0.80
```

---

# 十六、按钮 OCR Fallback

模板匹配失败后，可以使用 OCR 查找：

```text
去获胜
退出
购买挑战次数
确认
```

实现：

```python
find_text("去获胜")
```

返回文字 bounding box。

这样即使 UI 有少量变化，也有 fallback。

---

# 十七、页面状态识别

实现：

```text
ScreenDetector
```

支持：

```python
detect_arena_screen()
detect_prepare_screen()
detect_battle_screen()
detect_result_screen()
detect_purchase_dialog()
```

不要仅依赖一个文字。

例如竞技场页面可以综合：

```text
胜利者的竞技场
+
请选择对手
+
可挑战次数
```

来判断。

---

# 十八、状态机

必须使用明确的状态机。

例如：

```python
class AutomationState(Enum):
    IDLE = auto()
    CHECK_DEVICE = auto()
    DETECT_SCREEN = auto()
    ARENA = auto()
    SELECT_OPPONENT = auto()
    WAIT_PREPARE = auto()
    PREPARE = auto()
    START_BATTLE = auto()
    WAIT_RESULT = auto()
    RESULT = auto()
    EXIT_RESULT = auto()
    CHECK_CHALLENGE_COUNT = auto()
    BUY_CHALLENGE = auto()
    CONFIRM_PURCHASE = auto()
    RECOVERY = auto()
    STOPPED = auto()
    ERROR = auto()
```

---

# 十九、状态机原则

每个状态：

```text
进入
 ↓
执行
 ↓
验证结果
 ↓
成功 → 下一个状态
失败 → 重试/恢复
```

禁止：

```python
while True:
    click()
    sleep()
    click()
```

必须保证程序知道自己当前处于哪个页面。

---

# 二十、状态超时

每个状态必须有 timeout。

例如：

```yaml
timeouts:
  detect_screen: 10
  prepare: 15
  start_battle: 10
  result: 300
  purchase: 15
  recovery: 30
```

如果超时：

```text
进入 RECOVERY
```

而不是无限等待。

---

# 二十一、战斗检测

点击：

```text
去获胜
```

后进入：

```text
WAIT_RESULT
```

默认：

```yaml
battle:
  poll_interval: 5
  max_duration: 300
```

逻辑：

```text
等待 5 秒
 ↓
截图
 ↓
检测胜利
 ↓
检测失败
 ↓
检测结算页面
 ↓
没有结果 → 再等 5 秒
```

---

# 二十二、结算识别

优先检测：

```text
胜利
失败
退出
```

如果识别到：

```text
胜利
```

记录：

```text
WIN
```

如果识别到：

```text
失败
```

记录：

```text
LOSE
```

不要因为检测不到「胜利」就自动认为失败。

---

# 二十三、结算退出

检测：

```text
退出
```

然后点击。

点击后必须等待竞技场页面出现。

如果没有返回：

```text
等待
 ↓
再次检测
 ↓
必要时重新点击退出
```

但必须设置最大重试次数。

---

# 二十四、挑战次数

竞技场页面存在：

```text
可挑战次数 4 / 5
```

需要识别当前次数。

例如：

```text
4 / 5
3 / 5
2 / 5
1 / 5
0 / 5
```

必须提取当前值。

实现：

```python
ChallengeCount(
    current=4,
    maximum=5
)
```

---

# 二十五、购买挑战次数

当：

```text
current == 0
```

根据配置决定是否购买。

配置：

```yaml
purchase:
  enabled: true
  max_per_session: 10
  auto_confirm: true
```

如果：

```yaml
enabled: false
```

则停止自动化，并显示：

```text
挑战次数不足
```

---

# 二十六、购买流程

```text
点击购买挑战次数
 ↓
等待购买窗口
 ↓
检测购买确认按钮
 ↓
点击确认
 ↓
等待挑战次数变化
 ↓
重新进入 ARENA
```

购买后必须重新 OCR 挑战次数确认。

例如：

```text
购买前：0 / 5
购买后：1 / 5
```

如果仍然：

```text
0 / 5
```

不得继续无限点击购买。

---

# 二十七、购买保护

必须记录：

```text
session_purchase_count
```

达到：

```yaml
max_per_session
```

之后停止购买。

同时提供：

```text
累计购买次数
本次运行购买次数
```

---

# 二十八、异常恢复

这是项目的核心功能之一。

如果页面识别失败：

```text
RECOVERY
```

流程：

```text
截图
 ↓
尝试判断当前页面
 ↓
竞技场 → ARENA
备战 → PREPARE
结算 → RESULT
购买窗口 → BUY
未知页面 → 尝试返回
```

未知页面时可以：

```text
ADB BACK
```

但是必须配置：

```yaml
recovery:
  allow_back: true
  max_attempts: 3
```

如果仍然无法恢复：

```text
ERROR
```

停止自动点击。

---

# 二十九、绝对禁止危险点击

程序必须避免：

```text
识别失败
 ↓
默认坐标点击
```

所有自动点击必须满足：

```text
当前状态正确
+
目标识别成功
```

才允许点击。

对于 OCR：

```text
OCR_FAILED
```

不能转换成：

```text
0
```

---

# 三十、GUI

使用：

**PySide6**

界面建议：

```text
┌────────────────────────────────────────────┐
│ ArenaAuto                                  │
├────────────────────────────────────────────┤
│ ADB设备                                    │
│ [ emulator-5554                    ▼ ]     │
│                                            │
│ 战力阈值                                   │
│ [ 2000000 ]                                │
│                                            │
│ 检测间隔                                   │
│ [ 5 ] 秒                                   │
│                                            │
│ ☑ 自动购买挑战次数                         │
│ ☑ 自动确认购买                             │
│ ☑ 调试截图                                 │
│                                            │
│ [开始运行]                 [停止]           │
├────────────────────────────────────────────┤
│ 当前状态                                   │
│ ARENA                                      │
│                                            │
│ 当前目标                                   │
│ #2                                         │
│                                            │
│ 对手战力                                   │
│ #1  2,500,000                              │
│ #2  1,800,000                              │
│ #3  2,100,000                              │
│ #4  2,300,000                              │
│                                            │
│ 挑战次数                                   │
│ 4 / 5                                      │
├────────────────────────────────────────────┤
│ 运行统计                                   │
│ 胜利：23                                   │
│ 失败：2                                    │
│ 购买：3                                    │
│                                            │
├────────────────────────────────────────────┤
│ 日志                                       │
│ 20:41:02 检测到竞技场                     │
│ 20:41:03 #1 战力 2500000                   │
│ 20:41:03 #2 战力 1800000                   │
│ 20:41:03 选择 #2                            │
└────────────────────────────────────────────┘
```

---

# 三十一、GUI 线程

绝对不能把 ADB/OCR/自动化循环直接运行在 GUI 主线程。

必须使用：

```text
QThread
```

或者：

```text
QObject Worker + QThread
```

架构：

```text
GUI
 │
 └── AutomationWorker
       │
       ├── ADB
       ├── OCR
       ├── OpenCV
       └── StateMachine
```

Worker 通过 Qt Signal 更新 GUI：

```python
status_changed
log_message
power_detected
state_changed
statistics_changed
screenshot_updated
error_occurred
```

点击停止时必须能够安全停止。

---

# 三十二、停止机制

提供：

```python
threading.Event
```

或者类似的 cancellation event。

自动化循环中的每一个长等待都必须支持停止。

禁止：

```python
time.sleep(300)
```

导致点击停止后程序需要等 5 分钟。

应该实现可中断等待。

---

# 三十三、日志

日志同时输出：

```text
GUI日志窗口
+
logs/arena_auto.log
```

日志等级：

```text
DEBUG
INFO
WARNING
ERROR
```

格式：

```text
2026-09-22 20:41:03 [INFO] 检测到竞技场页面
2026-09-22 20:41:03 [INFO] #1 战力 = 2500000
2026-09-22 20:41:03 [INFO] #2 战力 = 1800000
2026-09-22 20:41:03 [INFO] 选择对手 #2
```

---

# 三十四、调试截图

配置：

```yaml
debug:
  enabled: true
  save_screenshots: true
  save_failed_recognition: true
  directory: "debug"
```

截图分类：

```text
debug/
├── arena/
├── prepare/
├── battle/
├── result/
├── purchase/
└── failed/
```

文件名：

```text
20260922_204103_arena.png
```

---

# 三十五、OCR 调试

如果 OCR 失败，自动保存：

```text
debug/failed/
```

同时保存 ROI：

```text
power_1.png
power_2.png
...
```

GUI 可以提供：

```text
[打开调试目录]
```

---

# 三十六、实时截图

GUI 提供：

```text
[刷新截图]
```

显示当前 ADB 屏幕。

如果性能允许，可以提供低频：

```text
1 FPS
```

预览。

不要每秒大量保存截图。

---

# 三十七、配置文件

使用：

```text
config/config.yaml
```

示例：

```yaml
adb:
  executable: "adb"
  serial: ""

automation:
  power_threshold: 2000000
  click_delay: 0.3
  poll_interval: 5

recognition:
  stable_reads: 2
  retry_count: 3

battle:
  poll_interval: 5
  max_duration: 300

purchase:
  enabled: true
  auto_confirm: true
  max_per_session: 10

recovery:
  allow_back: true
  max_attempts: 3

debug:
  enabled: true
  save_screenshots: true
  save_failed_recognition: true
  directory: "debug"

opponents:
  - id: 1
    power_region: [0, 0, 100, 50]
    click: [0, 0]

  - id: 2
    power_region: [0, 0, 100, 50]
    click: [0, 0]

  - id: 3
    power_region: [0, 0, 100, 50]
    click: [0, 0]

  - id: 4
    power_region: [0, 0, 100, 50]
    click: [0, 0]

  - id: 5
    power_region: [0, 0, 100, 50]
    click: [0, 0]

challenge_count:
  region: [0, 0, 100, 50]

templates:
  go_win:
    path: "resources/templates/go_win.png"
    threshold: 0.82

  exit:
    path: "resources/templates/exit.png"
    threshold: 0.82

  buy_challenge:
    path: "resources/templates/buy_challenge.png"
    threshold: 0.82

  confirm_buy:
    path: "resources/templates/confirm_buy.png"
    threshold: 0.82

  victory:
    path: "resources/templates/victory.png"
    threshold: 0.80

  defeat:
    path: "resources/templates/defeat.png"
    threshold: 0.80
```

第一次运行如果配置不存在：

自动创建默认配置。

---

# 三十八、配置管理器

实现：

```python
ConfigManager
```

负责：

```text
读取 YAML
保存 YAML
校验配置
提供默认值
```

配置错误必须给出明确提示。

例如：

```text
power_threshold 必须是数字
```

而不是程序启动直接崩溃。

---

# 三十九、分辨率适配

不要完全依赖绝对像素。

必须设计：

```text
reference_width
reference_height
```

例如：

```yaml
screen:
  reference_width: 1920
  reference_height: 1080
```

如果实际设备：

```text
1280x720
```

则：

```text
scale_x = actual_width / reference_width
scale_y = actual_height / reference_height
```

ROI 和点击坐标进行缩放。

但是：

**模板匹配最好支持多尺度匹配。**

---

# 四十、坐标转换

提供：

```python
CoordinateScaler
```

实现：

```python
scale_point()
scale_region()
```

避免项目中到处出现：

```python
x * 0.666
```

---

# 四十一、模板匹配多尺度

如果实现成本允许，模板匹配支持：

```text
0.8x
0.9x
1.0x
1.1x
1.2x
```

寻找最高 confidence。

如果超过阈值：

```text
成功
```

否则：

```text
失败
```

---

# 四十二、统计

程序运行期间记录：

```text
总挑战次数
胜利次数
失败次数
购买次数
识别失败次数
恢复次数
运行时间
```

GUI 实时显示。

---

# 四十三、运行记录

程序关闭前可以保存：

```text
data/statistics.json
```

下次启动可以继续显示历史统计。

---

# 四十四、错误处理

所有底层异常必须捕获。

例如：

```text
ADB disconnected
ADB timeout
OCR exception
OpenCV exception
配置错误
模板不存在
截图失败
```

不能出现：

```text
Python Traceback
```

直接把 GUI 崩掉。

GUI 必须显示：

```text
发生错误：
ADB设备已断开
```

---

# 四十五、ADB 自动重新连接

如果运行过程中：

```text
ADB disconnected
```

尝试：

```text
重新获取设备列表
重新连接
```

最多：

```yaml
adb:
  reconnect_attempts: 3
```

失败后停止。

---

# 四十六、设备选择

GUI 启动后读取：

```text
adb devices
```

显示：

```text
device1
device2
device3
```

允许用户选择。

如果只有一个设备：

自动选择。

如果没有设备：

显示：

```text
未检测到 Android 设备
```

---

# 四十七、模拟器兼容

不要写死：

```text
emulator-5554
```

支持：

```text
USB Android
Wi-Fi ADB
模拟器
```

---

# 四十八、测试模式

必须提供：

```text
模拟/测试模式
```

测试模式不执行真实点击。

例如：

```yaml
automation:
  dry_run: true
```

此时：

```text
识别战力
识别按钮
选择目标
```

都正常运行。

但：

```text
tap()
```

只写日志：

```text
[DRY RUN] tap(1000,500)
```

---

# 四十九、识别测试工具

GUI 增加：

```text
开发/调试
```

提供：

```text
[截图]
[测试OCR]
[测试竞技场检测]
[测试去获胜]
[测试退出]
[测试购买按钮]
```

最好可以在截图上绘制：

```text
ROI
识别结果
模板匹配框
confidence
```

这样以后调坐标非常方便。

---

# 五十、模板创建工具

如果实现成本允许，提供：

```text
[从当前截图创建模板]
```

流程：

```text
获取截图
 ↓
用户框选区域
 ↓
输入模板名称
 ↓
保存 PNG
 ↓
自动加入 config.yaml
```

这是非常有价值的开发辅助功能。

---

# 五十一、程序启动

支持：

```text
python main.py
```

也支持：

```text
python -m arena_auto
```

如果需要：

```text
python scripts/build.py
```

生成：

```text
dist/ArenaAuto.exe
```

---

# 五十二、PyInstaller

必须提供：

```text
ArenaAuto.spec
```

支持：

```text
PyInstaller
```

打包时包含：

```text
resources/
config/
```

如果 OCR 引擎需要额外模型/资源，也必须正确加入。

---

# 五十三、EXE 路径处理

必须考虑：

```text
开发环境
```

和：

```text
PyInstaller EXE
```

路径不同。

统一实现：

```python
get_resource_path()
get_app_data_path()
```

不要直接写：

```python
"./resources"
```

导致 EXE 中找不到资源。

---

# 五十四、配置和用户数据

建议：

```text
程序目录/
├── ArenaAuto.exe
├── resources/
├── config/
├── logs/
├── debug/
└── data/
```

配置文件必须是外部 YAML。

这样用户可以直接修改。

---

# 五十五、依赖

生成：

```text
requirements.txt
```

至少包含：

```text
PySide6
opencv-python
numpy
PyYAML
paddleocr
```

如果 PaddleOCR 实际依赖其他包，必须完整列出。

同时提供：

```text
requirements-dev.txt
```

用于开发测试。

---

# 五十六、README

必须生成完整 README。

包含：

1. 项目介绍
2. 环境要求
3. Python 安装
4. ADB 安装
5. Android 开启 USB/Wi-Fi 调试
6. 连接设备
7. 安装依赖
8. 配置 YAML
9. 准备模板
10. 启动程序
11. 调试方法
12. OCR 调整
13. 坐标调整
14. PyInstaller 打包
15. 常见问题
16. ADB 连接失败解决方法
17. OCR 识别失败解决方法
18. 模板匹配失败解决方法

---

# 五十七、日志示例

正常运行：

```text
20:41:01 [INFO] ADB device: emulator-5554
20:41:02 [INFO] 当前页面：ARENA
20:41:03 [INFO] 挑战次数：4/5
20:41:03 [INFO] #1 战力：2500000
20:41:03 [INFO] #2 战力：1800000
20:41:03 [INFO] #3 战力：2100000
20:41:03 [INFO] #4 战力：2300000
20:41:03 [INFO] #5 战力：2800000
20:41:03 [INFO] 战力阈值：2000000
20:41:03 [INFO] 选择目标：#2
20:41:04 [INFO] 当前页面：PREPARE
20:41:05 [INFO] 检测到「去获胜」
20:41:05 [INFO] 点击「去获胜」
20:41:10 [INFO] 战斗进行中
20:41:15 [INFO] 战斗进行中
20:41:20 [INFO] 检测到胜利
20:41:20 [INFO] 点击「退出」
20:41:22 [INFO] 返回竞技场
20:41:22 [INFO] 本轮完成：WIN
```

---

# 五十八、重要安全要求

这个程序是自动点击程序，所以必须尽量避免误操作。

必须遵守：

```text
识别失败 ≠ 默认成功
识别失败 ≠ 默认失败
识别失败 ≠ 默认点击
```

例如：

```python
if power is None:
    return OCR_FAILED
```

而不是：

```python
power = 0
```

同样：

```python
if not detect_go_win():
    不允许点击固定坐标
```

---

# 五十九、自动化循环

最终核心逻辑应该类似：

```python
while not stop_event.is_set():

    state = detect_current_state()

    if state == ARENA:
        handle_arena()

    elif state == PREPARE:
        handle_prepare()

    elif state == BATTLE:
        handle_battle()

    elif state == RESULT:
        handle_result()

    elif state == PURCHASE:
        handle_purchase()

    elif state == UNKNOWN:
        recover()

    elif state == ERROR:
        stop()
```

但不要简单复制这段伪代码。

请实现真正的状态机类。

---

# 六十、不要过度耦合

例如：

OCR 模块不知道竞技场逻辑。

ADB 模块不知道游戏。

OpenCV 模块不知道战力阈值。

GUI 不直接操作 ADB。

状态机通过接口调用：

```text
ADBController
RecognitionService
TemplateRecognizer
OCRProvider
ConfigManager
```

这样以后更换游戏 UI 或 OCR 引擎时不需要重写整个项目。

---

# 六十一、数据模型

至少实现：

```python
@dataclass
class Opponent:
    id: int
    power: int | None
    power_region: Region
    click_position: Point
```

```python
@dataclass
class ChallengeCount:
    current: int
    maximum: int
```

```python
@dataclass
class DetectionResult:
    found: bool
    confidence: float
    bbox: tuple
```

```python
@dataclass
class SessionStatistics:
    total_battles: int
    victories: int
    defeats: int
    purchases: int
    recognition_failures: int
    recoveries: int
```

---

# 六十二、类型和代码质量

必须：

* 使用类型注解
* 合理使用 dataclass
* 避免魔法数字
* 避免全局变量
* 避免超长函数
* 避免重复代码
* 使用 logging
* 使用 pathlib
* 使用异常类型
* 关键函数添加 docstring

---

# 六十三、测试

至少创建：

```text
tests/
├── test_config.py
├── test_power_parser.py
├── test_coordinate.py
├── test_state_machine.py
└── test_recognition.py
```

测试：

```text
1,234,567 → 1234567
1 234 567 → 1234567
1234567 → 1234567
```

以及：

```text
2500000 > 2000000
1800000 <= 2000000
```

选择：

```text
[2500000, 1800000, 1900000]
```

必须得到：

```text
2
```

全部超过：

```text
[2500000, 2800000, 3000000, 3500000, 4000000]
```

必须得到：

```text
None
```

---

# 六十四、项目生成要求

不要只生成核心代码。

必须一次性生成：

```text
完整源代码
+
requirements.txt
+
README.md
+
config.yaml
+
PyInstaller spec
+
测试
+
资源目录
+
模板目录
+
启动脚本
```

所有 import 必须正确。

所有目录必须有必要的 `__init__.py`。

不要留下：

```text
TODO
pass
...
NotImplemented
```

作为核心功能。

如果某个功能无法真正实现，请明确说明原因并提供可工作的替代实现，而不是伪造实现。

---

# 六十五、首次启动行为

首次运行：

```text
检测 config.yaml
 ↓
不存在
 ↓
创建默认配置
 ↓
检测 ADB
 ↓
显示设备
```

如果模板不存在：

```text
程序仍然可以启动
```

但 GUI 必须显示：

```text
缺少模板文件
请进入调试工具创建模板
```

---

# 六十六、不要硬编码我的实际坐标

因为我目前无法直接提供截图和准确坐标。

因此：

**不要假设具体坐标。**

先生成完整框架，并使用：

```yaml
power_region
click
challenge_count.region
```

作为可配置项。

默认值可以使用明显的占位区域：

```text
[0, 0, 100, 50]
```

并在 GUI/README 中说明需要根据实际设备截图进行配置。

---

# 六十七、最终需要支持的完整流程

必须能够实现：

```text
启动程序
 ↓
连接 Android
 ↓
检测竞技场
 ↓
读取挑战次数
 ↓
判断是否有次数
 ↓
读取 1~5 战力
 ↓
从 1 到 5 顺序寻找
 ↓
第一个 <= 战力阈值
 ↓
点击
 ↓
等待备战页面
 ↓
寻找「去获胜」
 ↓
点击
 ↓
等待战斗
 ↓
每 5 秒检测
 ↓
胜利/失败
 ↓
寻找「退出」
 ↓
点击
 ↓
回到竞技场
 ↓
继续下一轮
```

如果：

```text
挑战次数 = 0
```

则：

```text
点击购买
 ↓
确认购买
 ↓
验证次数恢复
 ↓
继续挑战
```

如果：

```text
无法识别
```

则：

```text
保存截图
 ↓
写入日志
 ↓
Recovery
 ↓
重新识别
```

如果：

```text
超过最大恢复次数
```

则：

```text
安全停止
```

---

# 六十八、GUI 最终必须做到

用户可以直接：

```text
选择 ADB 设备
设置战力阈值
设置检测间隔
开启/关闭自动购买
开启/关闭自动确认
开启/关闭调试截图
修改 OCR 参数
查看当前截图
查看 OCR 结果
查看当前状态
查看运行统计
查看实时日志
开始自动化
停止自动化
```

---

# 六十九、最终验收标准

生成项目后必须自行检查：

### ADB

* [ ] 可以发现设备
* [ ] 可以选择设备
* [ ] 可以截图
* [ ] 可以点击
* [ ] 设备断开不会崩溃

### OCR

* [ ] 可以读取战力
* [ ] 可以清洗数字
* [ ] OCR 失败不会误判
* [ ] 支持重复确认

### OpenCV

* [ ] 可以加载模板
* [ ] 可以模板匹配
* [ ] 返回 confidence
* [ ] 支持阈值

### 状态机

* [ ] Arena
* [ ] Prepare
* [ ] Battle
* [ ] Result
* [ ] Purchase
* [ ] Recovery
* [ ] Error

全部可正常转换。

### GUI

* [ ] 不阻塞
* [ ] 可以开始
* [ ] 可以停止
* [ ] 日志实时更新
* [ ] 状态实时更新

### 配置

* [ ] YAML 可读取
* [ ] YAML 可保存
* [ ] 配置错误有提示
* [ ] 默认配置自动创建

### 调试

* [ ] 可以保存截图
* [ ] 可以保存 OCR ROI
* [ ] 可以保存失败截图
* [ ] 可以查看识别结果

### 打包

必须成功执行：

```bash
pyinstaller ArenaAuto.spec
```

生成：

```text
dist/ArenaAuto/
```

或者：

```text
dist/ArenaAuto.exe
```

并且打包后的程序可以正常启动。

---

# 七十、最终输出要求

完成代码后，请按照以下顺序输出：

1. 项目目录树
2. 所有源代码
3. requirements.txt
4. config.yaml
5. PyInstaller spec
6. README
7. 测试代码
8. 安装步骤
9. 启动步骤
10. 配置方法
11. 模板创建方法
12. 第一次运行流程
13. 常见问题

不要只给核心代码。

目标是：

> **把生成出来的整个项目放入一个目录后，安装依赖即可运行，并且可以进一步使用 PyInstaller 打包。**

特别注意：

**当前没有真实游戏截图和坐标，因此不要伪造识别区域。**

所有和游戏 UI 相关的位置都必须配置化。

---

# 七十一、后续扩展预留

架构必须允许以后添加：

```text
多个竞技场配置
多个游戏配置
多个设备
不同战力策略
不同购买策略
更多页面
更多 OCR Provider
更多模板
自动刷新对手
运行计划
定时启动
统计导出
```

因此不要把：

```text
《胜利者的竞技场》
```

直接硬编码到所有模块中。

建议使用：

```text
ArenaProfile
```

保存游戏相关配置。

---

# 七十二、最终要求

请现在直接开始生成这个项目。

不要只给架构设计。

不要只给伪代码。

不要省略 GUI、ADB、OCR、OpenCV、状态机、配置管理、日志、调试、测试和 PyInstaller。

请确保生成的代码之间可以互相导入，并按照真实 Python 项目的方式组织。

如果一次输出过长，可以按照：

```text
Part 1：项目结构 + 核心模型 + 配置 + ADB
Part 2：OCR + OpenCV + 页面识别
Part 3：状态机 + 自动化逻辑
Part 4：PySide6 GUI
Part 5：测试 + PyInstaller + README
```

依次生成。

但每一部分都必须与前面的代码保持一致，最终形成一个完整可运行项目。


