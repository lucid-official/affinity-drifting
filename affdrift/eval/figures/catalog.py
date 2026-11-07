"""Figure IDs and their pre-registered reading criteria, verbatim from docs/plan.md §10.

The ``.md`` next to every figure quotes its entry here.  Do not paraphrase:
if the plan changes, copy the new text.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class FigureSpec:
    id: str
    experiments: str
    figure: str
    reading: str


CATALOG = MappingProxyType({s.id: s for s in (
    FigureSpec("FC.1", "모든 run", "항별 ‖V‖ 학습 곡선 (V⁺, V⁻, negative별)",
               "항 간 스케일이 [0.5, 2] 안. 한 항이 죽거나 폭주하면 비교 무효 (G2)"),
    FigureSpec("FC.2", "모든 run", "조건별 N_eff 히스토그램",
               "대부분 N* 근처. N_eff = 1에 몰린 조건은 사실상 single"),
    FigureSpec("FC.3", "모든 run", "probe set 샘플 그리드 (체크포인트별)",
               "학습 중 언제 다양성이 생기거나 사라지는지"),
    FigureSpec("FC.4", "모든 run", "ε 보간 strip: ε₀ → ε₁을 16단계로 보간한 출력",
               "mixture면 mode 사이를 **건너뛰고**, blob이면 매끄럽게 **변형**된다. Prop B를 가장 직접 보여 주는 판독"),
    FigureSpec("F0.1", "E0.2–E0.3", "particle 궤적 스냅샷 (t = 0, 중간, 끝), 조건별 색",
               "대칭은 자기 y_k로 모이고, h_q = 0은 이웃 y_j들에 나뉘어 모임"),
    FigureSpec("F0.2", "E0.3", "배정 행렬(조건 k × positive j에 모인 비율)과 w̃ 행렬을 나란히 그린 heatmap",
               "두 행렬의 띠 구조가 같음"),
    FigureSpec("F0.3", "E0.4", "h_q 보간 strip: h_q 4개 값의 최종 배정 행렬",
               "띠가 점점 대각선으로 좁아짐"),
    FigureSpec("F0.4", "E0.5", "λ별 평형 히스토그램 + 닫힌 형태 Gaussian 겹쳐 그리기, λ = 1.1은 발산 궤적",
               "λ < 1은 곡선이 거의 겹침. 평균이 μ₁에서 멀어지는 방향"),
    FigureSpec("F0.5", "E0.2", "V field quiver (조건 하나, 대칭 ↔ h_q = 0)",
               "h_q = 0에서 여러 basin이 보임"),
    FigureSpec('F1.1', 'E1.2–E1.3', '산점도 그리드: 행 = arm, 열 = 조건 4개. 참 $p(x\\mid c)$ 등고선을 배경에, 조건부 평균을 별표로',
               'single·K1은 별표 근처에 뭉침, K2는 엉뚱한 mode, K3는 퍼진 구름, AffD·K4는 3개 mode를 덮음. **Figure 1 후보**'),
    FigureSpec('F1.2', 'E1.3', '$V^+$ quiver: single ↔ AffD ↔ K1, 같은 조건',
               'single과 K1은 모든 화살표가 한 점을 향함 ($x$ 무관). AffD는 mode별 basin으로 나뉨. Prop B의 메커니즘 그림'),
    FigureSpec('F1.3', 'E1.3', '$\\epsilon \\to x$ 배정 그림: $\\epsilon$ 공간을 출력이 떨어진 mode 색으로 칠함',
               'AffD는 $\\epsilon$ 공간이 영역으로 분할되고, single은 한 색 또는 경계가 흐릿함'),
    FigureSpec('F1.4', 'E1.4', '$\\delta$ × $h_p$ phase diagram heatmap (E1.4b는 $1/L \\times h_p$로 한 장 더, 같은 색 척도) (값 = W2, 셀에 MCI 숫자)',
               '$\\delta$가 커질수록 single 열과 AffD 열의 색 차이가 커짐. **Figure 2 후보**'),
    FigureSpec('F1.5', 'E1.4', 'MCI–$\\delta$ 선 그래프 (arm별 선, 시드 CI 띠)',
               'single의 선만 $\\delta$와 함께 상승'),
    FigureSpec('F1.6', 'E1.5', '최근접 학습점 거리 히스토그램 (대칭 ↔ $h_q = 0$) + 배정 heatmap',
               '대칭은 0 근처에 몰림 (암기)'),
    FigureSpec('F1.7', 'E1.6', '$h_p$에 대한 W2 U자 곡선과 $\\lVert V\\rVert^2$ 곡선을 두 축으로, $N_{\\text{eff}}$ 규칙이 고른 $h$ 표시',
               '$\\lVert V\\rVert^2$는 $h$에 대해 단조 감소, W2는 U자. $\\lVert V\\rVert^2$로 튜닝하면 안 되는 이유'),
)})
