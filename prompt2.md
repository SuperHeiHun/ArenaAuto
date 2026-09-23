# ArenaAuto 自动配置 / 自动标定系统开发任务

## 一、任务背景

当前项目已经具备：

* PySide6 GUI
* ADB 设备控制
* ADB 截图
* OpenCV
* OCR
* 战力识别
* YAML 配置
* 状态机
* 调试截图
* `resources/templates/`
* `power1.png`、`power2.png` 等战力相关资源
* `power1_region.txt`、`power2_region.txt` 等区域配置

现在不要推倒重写整个项目。

请基于现有 ArenaAuto 项目进行**增量重构和功能扩展**。

本次主要目标：

> 将目前需要人工填写坐标、ROI、模板的配置方式，升级为“自动标定 + 半自动修正 + 手动配置兜底”的完整配置系统。

最终用户第一次使用时应该尽可能做到：

```text
连接 Android
    ↓
打开游戏竞技场
    ↓
点击「自动配置」
    ↓
程序截图并自动分析
    ↓
自动识别 1~5 对手
    ↓
自动识别战力区域
    ↓
自动推导点击区域
    ↓
用户确认
    ↓
程序保存配置
```

之后直接：

```text
开始自动挑战
```

不再需要手工填写大量坐标。

---

# 二、核心设计原则

必须保留现有功能。

不要删除：

* 原来的 YAML 配置
* 原来的手动 ROI
* 原来的模板
* 原来的 ADB
* 原来的 OCR
* 原来的状态机
* 原来的调试截图
* 原来的 GUI

新增自动配置系统必须建立在现有模块上。

禁止为了实现自动配置而重写整个项目。

---

# 三、最终配置模式

程序必须支持三种配置模式：

## 模式 1：自动标定

推荐普通用户使用。

```text
ADB
 ↓
截图
 ↓
自动分析
 ↓
自动生成配置
 ↓
用户确认
 ↓
保存
```

---

## 模式 2：半自动标定

如果自动识别结果不准确：

```text
自动分析
 ↓
显示识别框
 ↓
用户拖动/调整
 ↓
确认
 ↓
保存
```

---

## 模式 3：手动配置

保留现有 YAML / ROI 配置方式。

高级用户可以直接编辑：

```yaml
opponents:
  - id: 1
    power_region: [...]
    click: [...]
```

三种模式最终都生成相同的数据结构。

---

# 四、新增 Calibration 模块

增加：

```text
src/arena_auto/calibration/
├── __init__.py
├── wizard.py
├── analyzer.py
├── screen_analyzer.py
├── opponent_detector.py
├── button_detector.py
├── roi_detector.py
├── template_generator.py
├── coordinate_mapper.py
└── models.py
```

---

# 五、Calibration 数据模型

新增：

```python
@dataclass
class CalibrationPoint:
    x: int
    y: int
```

```python
@dataclass
class CalibrationRegion:
    x: int
    y: int
    width: int
    height: int
```

```python
@dataclass
class CalibrationDetection:
    name: str
    region: CalibrationRegion
    click_point: CalibrationPoint
    confidence: float
    source: str
```

其中：

```text
source:
- ocr
- template
- structure
- manual
```

---

# 六、自动标定向导

GUI 增加：

```text
[自动配置]
```

点击后打开：

```text
CalibrationWizard
```

向导页面：

```text
步骤 1：连接设备
步骤 2：确认竞技场页面
步骤 3：分析对手列表
步骤 4：确认战力区域
步骤 5：分析按钮
步骤 6：测试识别
步骤 7：保存配置
```

---

# 七、步骤 1：设备检测

自动获取：

```text
adb devices
```

如果只有一个设备：

自动选择。

如果多个设备：

显示设备选择框。

如果没有设备：

显示：

```text
未检测到 Android 设备
请确认 ADB 已连接
```

禁止进入下一步。

---

# 八、步骤 2：竞技场页面确认

获取当前截图。

自动判断是否是竞技场页面。

综合检测：

```text
胜利者的竞技场
请选择对手
可挑战次数
```

不要只检测单一文字。

如果识别失败：

允许用户点击：

```text
[我已确认当前是竞技场页面]
```

然后继续。

---

# 九、步骤 3：自动发现 1~5 对手

这是自动配置最重要的部分。

不要要求用户分别框选：

```text
power1
power2
power3
power4
power5
```

程序应该尝试自动发现重复的对手卡片结构。

---

# 十、对手结构检测

优先使用 OCR 获取整屏文字及 bounding box。

例如 OCR 得到：

```text
梦的胜负师
第2664名
1,234,567

智慧的机器
第2710名
1,543,210

Railgun
第2752名
1,872,320
```

程序提取：

```python
OCRText(
    text="1,234,567",
    bbox=(...)
)
```

---

# 十一、战力候选识别

对 OCR 文本进行数字清洗。

支持：

```text
1,234,567
1 234 567
1.234.567
1234567
```

转换：

```text
1234567
```

---

# 十二、不要把所有数字都认为是战力

页面可能存在：

```text
等级 85
排名 2664
挑战次数 4/5
金币 377726
战力 4575674
```

所以必须结合：

## 位置

## 数字长度

## OCR bounding box

## 对手卡片区域

## 相邻文本

进行判断。

例如：

```text
战力数字通常位于对手卡片的固定水平区域。
```

---

# 十三、重复结构分析

假设发现：

```text
候选战力：

(1200, 300)
(1200, 400)
(1200, 500)
(1200, 600)
(1200, 700)
```

程序应该发现：

```text
Y 间距 ≈ 100
```

于是建立：

```text
Opponent #1
Opponent #2
Opponent #3
Opponent #4
Opponent #5
```

即使其中一个 OCR 失败，也可以根据重复结构推导它的位置。

---

# 十四、自动生成 power_region

对于每个对手：

```text
OCR 战力 bbox
 ↓
扩大边界
 ↓
得到 power_region
```

例如：

```text
OCR bbox:
x=1200
y=310
w=180
h=40
```

生成：

```text
power_region:
x=1180
y=290
w=220
h=80
```

扩大范围必须可配置：

```yaml
calibration:
  power_region_padding:
    x: 20
    y: 20
```

---

# 十五、自动生成点击区域

对手卡片点击区域不应该直接使用战力文字位置。

程序应该寻找：

```text
对手卡片整体区域
```

点击点默认：

```text
卡片中心
```

如果无法识别卡片边界：

根据：

```text
战力位置
+
重复卡片间距
```

推导卡片区域。

最终：

```python
click_point = opponent_card.center
```

---

# 十六、自动生成 1~5

最终得到：

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

  - id: 5
    power_region: [x, y, width, height]
    click: [x, y]
```

---

# 十七、识别结果必须可视化

自动分析后不要直接保存。

先显示：

```text
┌───────────────────────────────────┐
│           当前游戏截图             │
│                                   │
│  ┌─────────────────────┐          │
│  │ #1                  │          │
│  │         1,234,567   │ ← ROI    │
│  └─────────────────────┘          │
│                                   │
│  ┌─────────────────────┐          │
│  │ #2                  │          │
│  │         1,543,210   │          │
│  └─────────────────────┘          │
│                                   │
└───────────────────────────────────┘
```

使用 OpenCV 在截图上绘制：

* 对手区域
* power ROI
* click point
* OCR bbox
* confidence
* 对手编号

---

# 十八、识别结果列表

右侧显示：

```text
#1
战力：1,234,567
ROI：自动
点击：自动
Confidence：0.94

#2
战力：1,543,210
ROI：自动
点击：自动
Confidence：0.91
```

每一项提供：

```text
[调整]
```

---

# 十九、半自动调整

点击：

```text
[调整 #1]
```

允许用户拖动：

```text
power ROI
click point
```

使用 PySide6 自定义：

```text
CalibrationCanvas
```

支持：

* 鼠标拖动
* 缩放
* 框选
* 调整 ROI
* 设置点击位置

---

# 二十、坐标系统

必须统一使用：

```text
Reference Resolution
```

例如：

```yaml
screen:
  reference_width: 1920
  reference_height: 1080
```

保存的坐标必须基于参考分辨率。

实际设备运行时：

```text
reference → actual
```

自动缩放。

---

# 二十一、自动按钮发现

自动配置不仅需要对手。

还要尝试自动发现：

```text
去获胜
退出
购买挑战次数
确认
胜利
失败
```

---

# 二十二、按钮 OCR

优先：

```text
OCR
 ↓
查找指定文字
 ↓
bbox
 ↓
扩大 bbox
 ↓
生成点击区域
```

例如：

```text
OCR:
去获胜
bbox=(1600,950,200,70)
```

自动保存：

```yaml
buttons:
  go_win:
    region: [...]
    click: [...]
```

---

# 二十三、按钮模板自动生成

找到按钮文字后：

```text
截图
 ↓
根据 bbox 扩展
 ↓
裁剪按钮
 ↓
保存模板
```

例如：

```text
resources/templates/
├── go_win.png
├── exit.png
├── buy_challenge.png
└── confirm_buy.png
```

这样用户不需要自己截图制作模板。

---

# 二十四、模板生成安全机制

不能把 OCR 文字 bbox 原样当成模板。

应该向四周扩展：

```yaml
calibration:
  template_padding:
    x: 20
    y: 15
```

并进行边界裁剪。

---

# 二十五、结算页面自动配置

用户进入一次胜利结算页面后：

程序自动寻找：

```text
胜利
退出
```

生成：

```yaml
result:
  victory:
    region: [...]
  exit:
    region: [...]
    click: [...]
```

---

# 二十六、失败页面

同样寻找：

```text
失败
退出
```

生成：

```yaml
result:
  defeat:
    region: [...]
```

不要要求用户手工配置。

---

# 二十七、购买页面

进入购买界面后：

自动寻找：

```text
购买挑战次数
确认
```

生成：

```yaml
purchase:
  buy_button:
    region: [...]
    click: [...]

  confirm_button:
    region: [...]
    click: [...]
```

---

# 二十八、挑战次数自动发现

竞技场页面 OCR 全屏。

寻找：

```text
4 / 5
3 / 5
2 / 5
1 / 5
0 / 5
```

程序自动发现该区域。

不要要求用户手工填写：

```yaml
challenge_count:
  region:
```

自动生成。

---

# 二十九、配置文件最终结构

自动配置后生成类似：

```yaml
version: 2

screen:
  reference_width: 1920
  reference_height: 1080

opponents:
  - id: 1
    power_region: [1200, 300, 220, 70]
    click: [1050, 335]

  - id: 2
    power_region: [1200, 400, 220, 70]
    click: [1050, 435]

  - id: 3
    power_region: [1200, 500, 220, 70]
    click: [1050, 535]

  - id: 4
    power_region: [1200, 600, 220, 70]
    click: [1050, 635]

  - id: 5
    power_region: [1200, 700, 220, 70]
    click: [1050, 735]

challenge_count:
  region: [500, 800, 150, 50]

buttons:
  go_win:
    region: [1600, 900, 250, 100]
    click: [1725, 950]

  exit:
    region: [1600, 900, 250, 100]
    click: [1725, 950]

purchase:
  buy_button:
    region: [1000, 500, 300, 100]
    click: [1150, 550]

  confirm_button:
    region: [1000, 600, 300, 100]
    click: [1150, 650]

result:
  victory:
    region: [500, 400, 300, 150]

  defeat:
    region: [500, 400, 300, 150]
```

实际数值由自动标定生成。

---

# 三十、不要覆盖用户手工配置

如果用户已经存在：

```text
config.yaml
```

自动标定时：

不要直接覆盖。

先生成：

```text
config.calibrated.yaml
```

GUI 显示：

```text
发现已有配置。

[覆盖现有配置]
[另存为新配置]
[取消]
```

默认选择：

```text
另存为新配置
```

---

# 三十一、配置版本

新增：

```yaml
version: 2
```

ConfigManager 必须支持版本迁移。

例如：

```text
version 1
 ↓
ConfigMigrator
 ↓
version 2
```

以后配置结构变化时不要直接导致旧配置失效。

---

# 三十二、自动标定置信度

所有自动发现结果都必须有：

```text
confidence
```

例如：

```text
#1 power
confidence = 0.96

#2 power
confidence = 0.93

go_win
confidence = 0.98
```

定义：

```text
0.90+
高可信
0.75~0.90
需要确认
<0.75
建议手动调整
```

---

# 三十三、自动标定不要盲目保存低置信度结果

例如：

```text
#3 power confidence = 0.41
```

则 GUI 必须显示：

```text
⚠ #3 识别可信度较低
```

并要求：

```text
[调整]
```

而不是直接认为成功。

---

# 三十四、自动标定测试

保存之前必须进行测试。

程序自动：

```text
重新截图
 ↓
读取 1~5 战力
 ↓
读取挑战次数
 ↓
寻找按钮
 ↓
对比刚刚标定的数据
```

如果一致：

```text
✓ 配置验证成功
```

如果不一致：

```text
⚠ 配置可能存在问题
```

---

# 三十五、测试模式

增加：

```text
[测试当前配置]
```

测试过程中：

**禁止真正点击。**

执行：

```text
截图
 ↓
识别竞技场
 ↓
读取 1~5 战力
 ↓
读取挑战次数
 ↓
识别去获胜
 ↓
识别退出
```

最终显示：

```text
配置测试结果

✓ 竞技场
✓ #1
✓ #2
✓ #3
✓ #4
✓ #5
✓ 挑战次数
✓ 去获胜
✓ 退出
```

---

# 三十六、自动配置日志

例如：

```text
[CALIBRATION] 开始自动标定
[CALIBRATION] 屏幕：1920x1080
[CALIBRATION] 检测竞技场页面
[CALIBRATION] OCR detected 42 text regions
[CALIBRATION] 找到 7 个数字候选
[CALIBRATION] 找到 5 个对手战力候选
[CALIBRATION] 对手结构分析成功
[CALIBRATION] #1 power ROI detected
[CALIBRATION] #2 power ROI detected
[CALIBRATION] #3 power ROI detected
[CALIBRATION] #4 power ROI detected
[CALIBRATION] #5 power ROI detected
[CALIBRATION] 找到「去获胜」
[CALIBRATION] 找到「退出」
[CALIBRATION] 找到挑战次数
[CALIBRATION] 自动标定完成
```

---

# 三十七、调试文件

自动标定过程中保存：

```text
debug/calibration/
├── original.png
├── ocr_overlay.png
├── opponent_detection.png
├── power_regions.png
├── buttons.png
├── challenge_count.png
└── final_calibration.png
```

这样如果自动标定失败，可以直接查看原因。

---

# 三十八、自动模板生成

自动生成的模板：

```text
resources/templates/generated/
```

不要直接覆盖用户原来的：

```text
resources/templates/
```

例如：

```text
generated/
├── go_win.png
├── exit.png
├── buy_challenge.png
└── confirm_buy.png
```

配置引用：

```yaml
templates:
  go_win:
    path: resources/templates/generated/go_win.png
```

---

# 三十九、不要强依赖模板

自动标定生成模板只是第一层。

运行时优先：

```text
模板匹配
```

失败后：

```text
OCR fallback
```

再失败：

```text
结构检测
```

最后：

```text
安全停止
```

禁止：

```text
模板失败 → 固定坐标强制点击
```

---

# 四十、自动配置按钮

主界面增加：

```text
[自动配置]
[重新标定]
[测试配置]
[手动配置]
```

其中：

```text
自动配置
```

打开向导。

```text
重新标定
```

直接进入向导并允许替换配置。

```text
测试配置
```

不执行点击。

```text
手动配置
```

打开 YAML 编辑/配置界面。

---

# 四十一、GUI 自动标定界面

建议：

```text
┌───────────────────────────────────────────────┐
│ ArenaAuto - 自动配置                         │
├───────────────────────────────────────────────┤
│ 步骤 3 / 7                                    │
│                                               │
│ 对手检测                                      │
│                                               │
│ ┌───────────────────────────────┐             │
│ │                               │             │
│ │      游戏实时截图              │             │
│ │                               │             │
│ │  [#1 ROI]                     │             │
│ │  [#2 ROI]                     │             │
│ │  [#3 ROI]                     │             │
│ │  [#4 ROI]                     │             │
│ │  [#5 ROI]                     │             │
│ │                               │             │
│ └───────────────────────────────┘             │
│                                               │
│ #1  1,234,567    96%    ✓                    │
│ #2  1,543,210    94%    ✓                    │
│ #3  1,872,320    91%    ✓                    │
│ #4  2,123,456    88%    ✓                    │
│ #5  2,345,678    72%    ⚠                    │
│                                               │
│ [重新分析] [调整]                             │
│                                               │
│ [上一步]                       [下一步]       │
└───────────────────────────────────────────────┘
```

---

# 四十二、截图缩放

因为手机截图可能非常大。

CalibrationCanvas 必须支持：

* 缩放
* 平移
* 鼠标滚轮缩放
* ROI 拖动
* 点击点拖动

不能简单 QLabel 显示一张图片。

---

# 四十三、自动标定的识别算法要求

不要把算法写死成：

```python
if len(numbers) == 5:
```

必须使用多个特征：

```text
OCR 数字
+
数字长度
+
位置
+
Y 间距
+
重复结构
+
邻近玩家信息
+
候选区域
```

综合评分。

---

# 四十四、对手检测评分

例如：

```python
score = (
    numeric_score * 0.30
    + position_score * 0.25
    + spacing_score * 0.20
    + structure_score * 0.15
    + ocr_confidence * 0.10
)
```

权重不要散落在代码中。

使用配置：

```yaml
calibration:
  weights:
    numeric: 0.30
    position: 0.25
    spacing: 0.20
    structure: 0.15
    ocr: 0.10
```

---

# 四十五、适配不同分辨率

自动标定必须使用：

```text
实际截图坐标
```

生成：

```text
reference coordinates
```

不要假设永远是：

```text
1920x1080
```

支持：

```text
1280x720
1920x1080
2560x1440
```

至少支持任意宽高比例一致的设备。

---

# 四十六、适配不同 UI 缩放

如果游戏 UI 整体缩放：

自动标定应该重新分析，而不是依赖固定坐标。

所以：

```text
重新标定
```

必须是正常功能，而不是错误恢复。

---

# 四十七、手动修正结果必须保存

用户调整：

```text
#3 power ROI
```

后：

```text
source = manual
confidence = 1.0
```

这样下次自动标定不会错误覆盖用户手动调整的数据。

---

# 四十八、配置来源优先级

最终运行时：

```text
manual
  >
calibrated
  >
default
```

如果用户明确手动修改了某个区域：

自动重新标定时不要覆盖它。

除非用户明确选择：

```text
[全部重新标定]
```

---

# 四十九、自动配置完成后

显示：

```text
自动配置完成

竞技场：
✓

对手：
5 / 5

战力区域：
5 / 5

挑战次数：
✓

去获胜：
✓

胜利：
✓

失败：
✓

退出：
✓

购买：
✓

配置验证：
✓

[保存配置]
[重新标定]
```

---

# 五十、验收标准

完成后必须验证：

## 自动发现

* [ ] 自动发现竞技场
* [ ] 自动发现 1~5 对手
* [ ] 自动发现战力区域
* [ ] 自动发现点击区域
* [ ] 自动发现挑战次数
* [ ] 自动发现「去获胜」
* [ ] 自动发现「退出」
* [ ] 自动发现购买按钮
* [ ] 自动发现确认按钮
* [ ] 自动发现胜利
* [ ] 自动发现失败

## 半自动

* [ ] 可以拖动 ROI
* [ ] 可以移动点击位置
* [ ] 可以重新分析
* [ ] 可以保存修改

## 配置

* [ ] 自动生成 YAML
* [ ] 不覆盖原配置
* [ ] 支持配置版本
* [ ] 支持旧配置迁移

## 调试

* [ ] 保存原始截图
* [ ] 保存 OCR Overlay
* [ ] 保存对手检测结果
* [ ] 保存 ROI
* [ ] 保存按钮检测结果
* [ ] 保存最终配置截图

## 安全

* [ ] OCR 失败不会默认点击
* [ ] 自动配置失败不会生成错误坐标
* [ ] 低 confidence 必须提示
* [ ] 测试配置不会真正点击

---

# 五十一、非常重要：不要制造“看起来自动、实际上不可靠”的方案

不要实现：

```text
截图
↓
随便找 5 个数字
↓
认为是战力
```

也不要实现：

```text
找到「去获胜」
↓
固定向右/向下偏移 100px
```

所有自动配置结果必须有：

```text
检测依据
+
confidence
+
可视化
+
用户确认
```

---

# 五十二、最终目标

最终用户体验应该是：

第一次：

```text
安装 ArenaAuto
 ↓
连接手机
 ↓
打开游戏
 ↓
进入竞技场
 ↓
点击「自动配置」
 ↓
程序自动分析
 ↓
用户确认一次
 ↓
保存
```

以后：

```text
打开游戏
 ↓
ArenaAuto
 ↓
选择设备
 ↓
开始
```

即可。

---

# 五十三、开发要求

请直接修改当前 ArenaAuto 项目。

不要重新创建一个与现有项目完全无关的 Demo。

首先分析现有项目：

```text
项目结构
现有 ADB
现有 OCR
现有 OpenCV
现有状态机
现有 ConfigManager
现有 GUI
现有 power1~4 资源
现有 region 配置
```

然后在现有架构上增量实现。

如果现有代码结构存在明显问题，可以进行必要的小范围重构，但：

**不得删除已有功能。**

---

# 五十四、最终输出

完成后输出：

1. 修改后的项目目录树
2. 新增文件
3. 修改文件
4. 每个核心模块的作用
5. 自动标定工作流程
6. 自动配置 YAML 示例
7. 启动方式
8. 测试方式
9. 自动标定使用方法
10. 如何重新标定
11. 如何手动调整
12. 如何查看 debug/calibration
13. PyInstaller 打包方式

同时检查：

```text
所有 import
所有路径
所有资源路径
所有 YAML 字段
所有 Signal
所有线程
所有模块依赖
```

确保项目可以直接运行。

如果发现现有代码与本提示词存在冲突：

**优先保留已有功能，然后通过兼容层或适配器实现新功能。**

不要为了“代码看起来更干净”而删除已有功能。
