import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import os
import tarfile
import tempfile
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import io
import platform

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
import matplotlib
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False

# ────────────────────────────────────────────────────────────────────────────────
# EfficientAD 핵심 구성 요소
# ────────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """EfficientNet-B0 사전학습 Teacher (가중치 고정)"""
    def __init__(self, out_channels=256):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.features)[:5])  # stride=8, ch=80
        self.proj = nn.Conv2d(80, out_channels, 1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.proj(self.encoder(x))


class StudentNet(nn.Module):
    """Teacher를 모사하는 Student"""
    def __init__(self, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class AENet(nn.Module):
    """오토인코더 - 정상 패턴만 재구성"""
    def __init__(self, out_channels=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class EfficientAD:
    """
    올바른 EfficientAD 구현
    - Teacher: 사전학습 EfficientNet (고정)
    - Student: Teacher 모사를 정상 이미지로 학습
    - AE: Teacher 특징 재구성을 정상 이미지로 학습
    - 이상 점수: 정상 학습 분포의 95 퍼센타일 기준으로 자동 임계값 계산
    """
    def __init__(self, image_size=256, out_channels=256, device='cpu'):
        self.image_size = image_size
        self.out_channels = out_channels
        self.device = device

        self.teacher = FeatureExtractor(out_channels).to(device)
        self.student = StudentNet(out_channels).to(device)
        self.ae      = AENet(out_channels).to(device)
        self.teacher.eval()

        self.t_mean = None
        self.t_std  = None
        self.threshold_auto = 0.5   # 학습 후 정상 분포에서 자동 계산
        self.score_lo = 0.0
        self.score_hi = 1.0

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _prep(self, img):
        return self.transform(img.convert('RGB')).unsqueeze(0).to(self.device)

    # ── 학습 ──────────────────────────────────────────────────────────────────
    def fit(self, train_images, epochs=10, lr=2e-4, progress_callback=None):
        # 1) Teacher 특징 통계 (정규화용)
        self.teacher.eval()
        with torch.no_grad():
            feats = torch.cat([self.teacher(self._prep(img)) for img in train_images])
        self.t_mean = feats.mean(dim=(0,2,3), keepdim=True)
        self.t_std  = feats.std (dim=(0,2,3), keepdim=True).clamp(min=1e-6)

        # 2) Student + AE 학습
        opt_s = torch.optim.Adam(self.student.parameters(), lr=lr, weight_decay=1e-5)
        opt_a = torch.optim.Adam(self.ae.parameters(),      lr=lr, weight_decay=1e-5)
        sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=epochs)
        sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=epochs)

        self.student.train()
        self.ae.train()
        total = len(train_images) * epochs
        step  = 0

        for ep in range(epochs):
            for img in train_images:
                x = self._prep(img)
                with torch.no_grad():
                    t_feat = self.teacher(x)
                    t_norm = (t_feat - self.t_mean) / self.t_std

                # Student loss
                s_out = self.student(x)
                s_out = F.interpolate(s_out, size=t_norm.shape[2:],
                                      mode='bilinear', align_corners=False)
                loss_s = F.mse_loss(s_out, t_norm)
                opt_s.zero_grad(); loss_s.backward(); opt_s.step()

                # AE loss
                ae_out = self.ae(x)
                ae_out = F.interpolate(ae_out, size=t_norm.shape[2:],
                                       mode='bilinear', align_corners=False)
                loss_a = F.mse_loss(ae_out, t_norm)
                opt_a.zero_grad(); loss_a.backward(); opt_a.step()

                step += 1
                if progress_callback:
                    progress_callback(step, total, loss_s.item(), loss_a.item())

            sch_s.step()
            sch_a.step()

        self.student.eval()
        self.ae.eval()

        # 3) 정상 이미지 점수 분포로 임계값 자동 계산
        #    정상의 95퍼센타일을 임계값으로 → FPR ≈ 5%
        raw_scores = []
        with torch.no_grad():
            for img in train_images:
                score, _ = self._raw_predict(img)
                raw_scores.append(score)

        raw_scores = torch.tensor(raw_scores)
        self.score_lo = raw_scores.quantile(0.05).item()
        self.score_hi = raw_scores.quantile(0.95).item()
        # 정상 95퍼센타일을 임계값으로 설정 (정규화 후 기준)
        p99_raw = raw_scores.quantile(0.99).item()
        self.threshold_auto = (p99_raw - self.score_lo) / max(self.score_hi - self.score_lo, 1e-6)

    @torch.no_grad()
    def _raw_predict(self, img):
        """정규화 전 원시 점수 반환"""
        x = self._prep(img)
        t_feat = self.teacher(x)
        t_norm = (t_feat - self.t_mean) / self.t_std

        s_out  = self.student(x)
        s_out  = F.interpolate(s_out,  size=t_norm.shape[2:], mode='bilinear', align_corners=False)
        ae_out = self.ae(x)
        ae_out = F.interpolate(ae_out, size=t_norm.shape[2:], mode='bilinear', align_corners=False)

        st_map = torch.mean((t_norm - s_out)  ** 2, dim=1, keepdim=True)
        ae_map = torch.mean((t_norm - ae_out) ** 2, dim=1, keepdim=True)
        combined = 0.5 * st_map + 0.5 * ae_map

        up = F.interpolate(combined, size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False)
        # max 대신 상위 1% 평균 사용 (노이즈에 강건)
        flat = up.flatten()
        k = max(1, int(len(flat) * 0.005))
        score = flat.topk(k).values.mean().item()
        heatmap = up.squeeze().cpu().numpy()
        return score, heatmap

    @torch.no_grad()
    def predict(self, img, weight_ae=0.5):
        """정규화된 점수 (0=완전 정상, 1=임계값 근처, >1=이상)"""
        x = self._prep(img)
        t_feat = self.teacher(x)
        t_norm = (t_feat - self.t_mean) / self.t_std

        s_out  = self.student(x)
        s_out  = F.interpolate(s_out,  size=t_norm.shape[2:], mode='bilinear', align_corners=False)
        ae_out = self.ae(x)
        ae_out = F.interpolate(ae_out, size=t_norm.shape[2:], mode='bilinear', align_corners=False)

        st_map = torch.mean((t_norm - s_out)  ** 2, dim=1, keepdim=True)
        ae_map = torch.mean((t_norm - ae_out) ** 2, dim=1, keepdim=True)
        combined = (1 - weight_ae) * st_map + weight_ae * ae_map

        up = F.interpolate(combined, size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False)

        # 상위 1% 평균으로 점수 계산 (max보다 안정적)
        flat = up.flatten()
        k = max(1, int(len(flat) * 0.005))
        raw_score = flat.topk(k).values.mean().item()

        # 정상 분포 기준으로 정규화 (0~1 범위, 1이 자동 임계값)
        norm_score = (raw_score - self.score_lo) / max(self.score_hi - self.score_lo, 1e-6)
        heatmap = up.squeeze().cpu().numpy()
        return norm_score, heatmap


# ────────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ────────────────────────────────────────────────────────────────────────────────

def load_images_from_tar(tar_path, prefix, max_images=None):
    images = []
    with tarfile.open(tar_path, 'r:xz') as tf:
        members = [m for m in tf.getmembers()
                   if m.name.startswith(prefix) and m.name.endswith('.png') and not m.isdir()]
        members = sorted(members, key=lambda x: x.name)
        if max_images:
            members = members[:max_images]
        for m in members:
            f = tf.extractfile(m)
            if f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                images.append((m.name, img))
    return images


def make_overlay(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img_arr = np.array(img.resize((heatmap.shape[1], heatmap.shape[0])))
    hmap_norm = np.clip(heatmap, 0, 1)
    colormap = cm.get_cmap('jet')
    hmap_colored = (colormap(hmap_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = (img_arr * (1 - alpha) + hmap_colored * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EfficientAD 이상 감지",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .anomaly-high { color: #ff4b4b; font-size: 2rem; font-weight: bold; }
    .anomaly-low  { color: #00cc88; font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🔩 EfficientAD 나사 이상 감지 시스템")
st.caption("MVTec Screw 데이터셋 · Teacher-Student + AutoEncoder 기반 비지도 이상 감지")

# ── 사이드바 파라미터 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 모델 파라미터")

    st.subheader("🏗️ 아키텍처")
    out_channels = st.select_slider("출력 채널 수", [128, 256, 384], value=256,
                                    help="Teacher/Student 특징 벡터 차원")
    image_size = st.select_slider("입력 이미지 크기", [128, 192, 256, 320], value=256,
                                  help="클수록 정밀하지만 느림")

    st.subheader("🎓 학습")
    max_train = st.slider("학습 이미지 수", 10, 320, 80, 10,
                          help="많을수록 정확하지만 느림")
    epochs = st.slider("에폭 수", 1, 30, 10, 1,
                       help="많을수록 정확하지만 느림")
    lr = st.select_slider("학습률", [1e-4, 2e-4, 5e-4, 1e-3], value=2e-4,
                          format_func=lambda x: f"{x:.0e}")

    st.subheader("🔍 추론")
    weight_ae = st.slider("AE 가중치", 0.0, 1.0, 0.5, 0.05,
                          help="0=Teacher-Student만, 1=AutoEncoder만 사용")
    use_auto_threshold = st.checkbox("자동 임계값 사용 (권장)", value=True,
                                      help="정상 학습 이미지의 95퍼센타일을 임계값으로 자동 설정")
    manual_threshold = st.slider("수동 임계값", 0.1, 3.0, 1.0, 0.05,
                                 help="자동 임계값 해제 시 사용")
    overlay_alpha = st.slider("히트맵 투명도", 0.1, 0.9, 0.5, 0.05)

    st.subheader("📂 데이터")
    tar_path_input = st.text_input(
        "TAR 파일 경로",
        value=r"C:\Users\KDS21\Desktop\기업프로젝트\screw.tar.xz",
        help="screw_tar.xz 파일의 절대 경로"
    )

    device_opt = st.radio("연산 장치", ["auto", "cpu"],
                          help="auto: GPU 있으면 CUDA 사용")
    device = "cuda" if (device_opt == "auto" and torch.cuda.is_available()) else "cpu"
    st.caption(f"현재 장치: **{device}**")

# ── 세션 상태 ──────────────────────────────────────────────────────────────────
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'train_config' not in st.session_state:
    st.session_state.train_config = {}

# ── 탭 ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎓 모델 학습", "🔍 이상 감지", "📊 배치 평가"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 학습
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("모델 초기화 및 정규화 학습")
    st.info("""
    EfficientAD는 **정상 이미지만**으로 Teacher 특징의 통계를 학습합니다.
    별도 라벨 없이 분포를 추정하는 비지도 학습 방식입니다.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        **현재 설정:**
        - 채널: `{out_channels}` | 이미지: `{image_size}px` | 에폭: `{epochs}` | LR: `{lr}`
        - 학습 이미지: 최대 `{max_train}`장 (train/good 폴더)
        - 장치: `{device}`
        """)

    with col2:
        train_btn = st.button("🚀 학습 시작", type="primary", use_container_width=True)

    if train_btn:
        if not os.path.exists(tar_path_input):
            st.error(f"파일을 찾을 수 없습니다: `{tar_path_input}`")
        else:
            progress_bar = st.progress(0)
            status = st.empty()

            status.info("📦 정상 학습 이미지 로딩 중...")
            train_data = load_images_from_tar(tar_path_input, "screw/train/good/", max_train)
            train_imgs = [img for _, img in train_data]
            status.info(f"✅ {len(train_imgs)}장 로딩 완료. 모델 초기화 및 학습 시작...")

            model = EfficientAD(image_size, out_channels, device)

            loss_placeholder = st.empty()
            def prog_cb(step, total, ls, la):
                progress_bar.progress(min(step / total, 1.0))
                loss_placeholder.caption(f"Step {step}/{total} | Loss S: {ls:.4f} | Loss AE: {la:.4f}")

            status.info("🎓 Student + AE 학습 중...")
            model.fit(train_imgs, epochs=epochs, lr=lr, progress_callback=prog_cb)

            st.session_state.model = model
            st.session_state.trained = True
            st.session_state.train_config = {
                'out_channels': out_channels,
                'image_size': image_size, 'n_train': len(train_imgs), 'epochs': epochs
            }

            progress_bar.progress(1.0)
            auto_thr = model.threshold_auto
            status.success(f"🎉 학습 완료! {len(train_imgs)}장 학습 · 자동 임계값: {auto_thr:.3f}")

    if st.session_state.trained:
        cfg = st.session_state.train_config
        st.success(f"✅ 모델 준비 완료 — 채널: `{cfg['out_channels']}`, 학습: `{cfg['n_train']}`장, 에폭: `{cfg['epochs']}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 단일 이미지 이상 감지
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("단일 이미지 이상 감지")

    if not st.session_state.trained:
        st.warning("⚠️ 먼저 [모델 학습] 탭에서 학습을 완료해주세요.")
    else:
        source = st.radio("이미지 소스", ["데이터셋에서 선택", "직접 업로드"])

        img_to_test = None
        true_label = None

        if source == "데이터셋에서 선택":
            if not os.path.exists(tar_path_input):
                st.error("TAR 파일을 찾을 수 없습니다.")
            else:
                split = st.selectbox("분류", ["test/good", "test/manipulated_front",
                                              "test/scratch_head", "test/scratch_neck",
                                              "test/thread_side", "test/thread_top"])
                true_label = "정상" if split == "test/good" else f"이상 ({split.split('/')[1]})"

                items = load_images_from_tar(tar_path_input, f"screw/{split}/", max_images=50)
                if items:
                    names = [n.split('/')[-1] for n, _ in items]
                    sel = st.selectbox("이미지 선택", names)
                    idx = names.index(sel)
                    img_to_test = items[idx][1]

        else:
            uploaded = st.file_uploader("이미지 업로드 (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
            if uploaded:
                img_to_test = Image.open(uploaded).convert('RGB')
                true_label = "알 수 없음"

        if img_to_test and st.button("🔍 분석 실행", type="primary"):
            model = st.session_state.model
            with st.spinner("이상 점수 계산 중..."):
                score, heatmap = model.predict(img_to_test, weight_ae)

            threshold = model.threshold_auto if use_auto_threshold else manual_threshold
            is_anomaly = score >= threshold
            verdict = "🔴 이상 감지" if is_anomaly else "🟢 정상"

            # 결과 표시
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("원본 이미지")
                st.image(img_to_test.resize((256, 256)), use_container_width=True)
            with c2:
                st.subheader("이상 히트맵")
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with c3:
                st.subheader("오버레이")
                overlay = make_overlay(img_to_test, heatmap, overlay_alpha)
                st.image(overlay, use_container_width=True)

            st.divider()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("이상 점수", f"{score:.4f}", delta=f"임계값: {threshold}")
            mc2.metric("판정", verdict)
            if true_label:
                mc3.metric("실제 레이블", true_label)

            if is_anomaly:
                st.error(f"**{verdict}** — 점수 {score:.4f} ≥ 임계값 {threshold}")
            else:
                st.success(f"**{verdict}** — 점수 {score:.4f} < 임계값 {threshold}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: 배치 평가
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("테스트셋 배치 평가")

    if not st.session_state.trained:
        st.warning("⚠️ 먼저 [모델 학습] 탭에서 학습을 완료해주세요.")
    else:
        defect_types = st.multiselect(
            "평가할 결함 유형",
            ["good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
            default=["good", "scratch_head", "manipulated_front"]
        )
        max_per_class = st.slider("클래스당 최대 이미지 수", 5, 50, 20)

        if st.button("📊 배치 평가 시작", type="primary"):
            if not os.path.exists(tar_path_input):
                st.error("TAR 파일을 찾을 수 없습니다.")
            else:
                model = st.session_state.model
                results = []
                prog = st.progress(0)
                status = st.empty()
                total_types = len(defect_types)

                for ti, dtype in enumerate(defect_types):
                    status.info(f"처리 중: test/{dtype}")
                    items = load_images_from_tar(tar_path_input, f"screw/test/{dtype}/",
                                                 max_images=max_per_class)
                    for name, img in items:
                        score, _ = model.predict(img, weight_ae)
                        threshold = model.threshold_auto if use_auto_threshold else manual_threshold
                        results.append({
                            '파일': name.split('/')[-1],
                            '유형': dtype,
                            '실제': '정상' if dtype == 'good' else '이상',
                            '이상점수': round(score, 4),
                            '판정': '이상' if score >= threshold else '정상',
                            '정답': (dtype == 'good') == (score < threshold),
                        })
                    prog.progress((ti + 1) / total_types)

                status.success("✅ 평가 완료!")

                import pandas as pd
                df = pd.DataFrame(results)

                # 성능 지표
                tp = len(df[(df['실제'] == '이상') & (df['판정'] == '이상')])
                tn = len(df[(df['실제'] == '정상') & (df['판정'] == '정상')])
                fp = len(df[(df['실제'] == '정상') & (df['판정'] == '이상')])
                fn = len(df[(df['실제'] == '이상') & (df['판정'] == '정상')])
                acc = (tp + tn) / max(len(df), 1)
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-6)

                st.subheader("📈 성능 지표")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("정확도", f"{acc:.1%}")
                m2.metric("정밀도", f"{prec:.1%}")
                m3.metric("재현율", f"{rec:.1%}")
                m4.metric("F1 점수", f"{f1:.3f}")

                st.subheader("📋 유형별 평균 이상 점수")
                summary = df.groupby('유형')['이상점수'].agg(['mean', 'max', 'min']).round(4)
                summary.columns = ['평균', '최대', '최소']
                st.dataframe(summary, use_container_width=True)

                st.subheader("🔬 점수 분포")
                fig, ax = plt.subplots(figsize=(10, 4))
                for dtype in df['유형'].unique():
                    sub = df[df['유형'] == dtype]['이상점수']
                    ax.hist(sub, bins=15, alpha=0.6, label=dtype)
                ax.axvline(threshold, color='red', linestyle='--', label=f'임계값={threshold}')
                ax.set_xlabel('이상 점수')
                ax.set_ylabel('빈도')
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.subheader("📄 상세 결과")
                st.dataframe(
                    df.style.applymap(
                        lambda v: 'color: red' if v == '이상' else 'color: green',
                        subset=['판정']
                    ),
                    use_container_width=True
                )