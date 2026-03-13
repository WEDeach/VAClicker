# VAClicker

用於辟邪除妖steam版本的輕量級腳本

## 先決條件

1. 一個被封也沒差的帳號

2. 安裝程序依賴 (測試於`python 3.10`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 運行腳本

腳本位於 scripts 下, 一些參數請自行微調 (開箱角色等)

```powershell
python -m scripts.sa_event
```

## 疑難排解

1. 遊戲僅測試在 __1920x1080__ 的解析度, 地下城亮度 __最亮__
2. 自動戰鬥模式請固定為 __模式1__ 並為角色配置相對行為(遊戲內)
3. 默認情況下, __角色意志力過低仍會強制進入地下城__
4. 程序有使用OCR, 目前僅針對 __繁體中文__ 做支持

## 免責聲明

本專案僅供學習與研究用途。

使用腳本化工具可能違反遊戲服務條款，並導致帳號遭到封禁。作者不對任何因使用本專案而產生的損失或後果負責。
