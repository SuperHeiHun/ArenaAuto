"""Editor for game-specific OCR keywords (no OCR or device actions)."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from arena_auto.config.manager import parse_arena_profile
from arena_auto.exceptions import ConfigError
from arena_auto.models.config import ArenaProfile


class OcrTextDialog(QDialog):
    """Changes are returned only after confirmation; cancel leaves config intact."""

    TEXT_FIELDS = (
        ("display_name", "配置名称"),
        ("go_win_text", "去获胜 / 开始按钮"),
        ("exit_text", "退出按钮"),
        ("victory_text", "胜利文案"),
        ("defeat_text", "失败文案"),
        ("buy_text", "购买按钮 / 弹窗文案"),
        ("confirm_text", "确认按钮"),
    )
    KEYWORD_FIELDS = (
        ("arena_title_keywords", "竞技场标题关键词"),
        ("arena_select_keywords", "选择对手关键词"),
        ("challenge_keywords", "挑战次数关键词"),
    )

    def __init__(self, profile: ArenaProfile, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("自定义 OCR 文本")
        self.resize(560, 620)
        self.profile = profile
        layout = QVBoxLayout(self)
        hint = QLabel(
            "填写游戏画面上的实际文字。页面关键词和按钮文案均可用逗号 , 或 ，"
            "分隔多个文本，匹配任意一个即可；页面关键词也支持换行。"
            "留空可禁用该项 OCR 文本匹配（模板匹配仍然有效）。\n"
            "模板优先；修改文字不改变坐标，也不会执行点击。保存后可在开发/调试中测试。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.text_edits = {}
        self.keyword_edits = {}
        for name, label in self.TEXT_FIELDS:
            edit = QLineEdit()
            if name != "display_name":
                edit.setPlaceholderText("例如：去获胜,开始挑战")
            self.text_edits[name] = edit
            form.addRow(label, edit)
        for name, label in self.KEYWORD_FIELDS:
            edit = QPlainTextEdit()
            edit.setMaximumHeight(65)
            edit.setPlaceholderText("逗号分隔或每行一个关键词")
            self.keyword_edits[name] = edit
            form.addRow(label, edit)
        self.min_hits_spin = QSpinBox()
        self.min_hits_spin.setRange(1, 1000)
        form.addRow("竞技场最低关键词命中数", self.min_hits_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存并应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        reset = QPushButton("恢复默认文本")
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        reset.clicked.connect(lambda: self.load_profile(ArenaProfile()))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.load_profile(profile)

    def load_profile(self, profile: ArenaProfile) -> None:
        for name, edit in self.text_edits.items():
            edit.setText(getattr(profile, name))
        for name, edit in self.keyword_edits.items():
            edit.setPlainText("\n".join(getattr(profile, name)))
        self.min_hits_spin.setValue(profile.min_keyword_matches)

    def edited_profile(self) -> ArenaProfile:
        values = asdict(self.profile)
        values.update({name: edit.text() for name, edit in self.text_edits.items()})
        values.update({name: edit.toPlainText() for name, edit in self.keyword_edits.items()})
        values["min_keyword_matches"] = self.min_hits_spin.value()
        return parse_arena_profile(values)

    def accept(self) -> None:
        try:
            self.profile = self.edited_profile()
        except ConfigError as exc:
            QMessageBox.warning(self, "OCR 文本配置错误", str(exc))
            return
        super().accept()
