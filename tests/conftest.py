import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 `src` 패키지를 import 가능하도록 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
