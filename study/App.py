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
import tarfile
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import io
import platform
import threading
import datetime
import json
import traceback
import random

# ── 한글 폰트 설정
import matplotlib
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False


# ────────────────────────────────────────────────────────────────────────────────
# 파일 I/O 유틸리티
# ────────────────────────────────────────────────────────────────────────────────

def _save_yaml(data: dict, path: str):
    """YAML 저장 (PyYAML 없으면 JSON으로 fallback)"""
    try:
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
    except ImportError:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _save_checkpoint(model, opt_s, opt_a, sch_s, sch_a,
                     epoch: int, step: int, config: dict,
                     checkpoint_dir: str) -> str:
    """학습 체크포인트를 .ckpt 파일로 저장"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(checkpoint_dir, f'checkpoint_{ts}.ckpt')
    torch.save({
        'epoch':         epoch,
        'step':          step,
        'teacher_state': model.teacher.state_dict(),
        'student_state': model.student.state_dict(),
        'ae_state':      model.ae.state_dict(),
        'opt_s_state':   opt_s.state_dict(),
        'opt_a_state':   opt_a.state_dict(),
        'sch_s_state':   sch_s.state_dict(),
        'sch_a_state':   sch_a.state_dict(),
        't_mean':         model.t_mean,
        't_std':          model.t_std,
        'train_scores':   model._train_scores,
        'map_q001':       model._map_q001,
        'map_q999':       model._map_q999,
        'threshold_p95':  model._threshold_p95,
        'threshold_p99':  model._threshold_p99,
        'st_mean_g':        model._st_mean_g,
        'st_std_g':         model._st_std_g,
        'ae_mean_g':        model._ae_mean_g,
        'ae_std_g':         model._ae_std_g,
        'feat_mean_loc':    model._feat_mean_loc,
        'feat_std_loc':     model._feat_std_loc,
        'direct_mean_g':    model._direct_mean_g,
        'direct_std_g':     model._direct_std_g,
        'direct_map_q001':  model._direct_map_q001,
        'direct_map_q999':  model._direct_map_q999,
        'config':           config,
    }, path)
    return path


def _save_final_model(model, model_root: str, config: dict) -> str:
    """학습 완료 후 모델을 {model_root}/models/efficientad_{timestamp}/ 에 저장"""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(model_root, 'models', f'efficientad_{ts}')
    os.makedirs(model_dir, exist_ok=True)

    # model.pth
    torch.save({
        'teacher':   model.teacher.state_dict(),
        'student':   model.student.state_dict(),
        'ae':        model.ae.state_dict(),
        't_mean':    model.t_mean,
        't_std':     model.t_std,
        'map_q001':  model._map_q001,
        'map_q999':  model._map_q999,
        'st_mean_g': model._st_mean_g,
        'st_std_g':  model._st_std_g,
        'ae_mean_g':       model._ae_mean_g,
        'ae_std_g':        model._ae_std_g,
        'feat_mean_loc':   model._feat_mean_loc,
        'feat_std_loc':    model._feat_std_loc,
        'direct_mean_g':   model._direct_mean_g,
        'direct_std_g':    model._direct_std_g,
        'direct_map_q001': model._direct_map_q001,
        'direct_map_q999': model._direct_map_q999,
    }, os.path.join(model_dir, 'model.pth'))

    # config.yaml
    _save_yaml(config, os.path.join(model_dir, 'config.yaml'))

    # thresholds.json
    scores = model._train_scores or []
    if scores:
        arr = torch.tensor(scores)
        mu, sigma = arr.mean().item(), arr.std().item()
    else:
        mu = sigma = 0.0
    p95 = model._threshold_p95 if model._threshold_p95 is not None else (
        float(np.percentile(scores, 95)) if scores else 0.0)
    p99 = model._threshold_p99 if model._threshold_p99 is not None else (
        float(np.percentile(scores, 99)) if scores else 0.0)
    thresholds = {
        'threshold_auto': round(mu + 2.0 * sigma, 6),
        'threshold_p95':  round(p95, 6),
        'threshold_p99':  round(p99, 6),
        'mu':             round(mu, 6),
        'sigma':          round(sigma, 6),
        'train_scores':   [round(s, 6) for s in scores],
    }
    with open(os.path.join(model_dir, 'thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    # 루트 config.yaml 갱신
    os.makedirs(model_root, exist_ok=True)
    _save_yaml(config, os.path.join(model_root, 'config.yaml'))

    return model_dir


# ────────────────────────────────────────────────────────────────────────────────
# 학습 스레드
# ────────────────────────────────────────────────────────────────────────────────

def _run_training_thread(model, train_images, epochs, lr,
                          training_state: dict,
                          pause_event: threading.Event,
                          stop_event:  threading.Event,
                          checkpoint_dir: str, model_root: str, config: dict,
                          start_epoch: int = 0, start_step: int = 0,
                          opt_s_state=None, opt_a_state=None,
                          sch_s_state=None, sch_a_state=None):
    """백그라운드 스레드에서 model.fit() 실행"""

    def progress_cb(step, total, ls, la):
        training_state['step']   = step
        training_state['total']  = total
        training_state['loss_s'] = ls
        training_state['loss_a'] = la
        training_state['status'] = 'training'   # 재개 후 상태 복원

    def checkpoint_cb(epoch, step_in_ep, opt_s, opt_a, sch_s, sch_a):
        try:
            path = _save_checkpoint(
                model, opt_s, opt_a, sch_s, sch_a,
                epoch, step_in_ep, config, checkpoint_dir)
            training_state['last_checkpoint'] = path
            training_state['status'] = 'paused'
        except Exception as e:
            training_state['checkpoint_error'] = str(e)

    try:
        result = model.fit(
            train_images, epochs=epochs, lr=lr,
            progress_callback=progress_cb,
            pause_event=pause_event,
            stop_event=stop_event,
            start_epoch=start_epoch,
            start_step_in_epoch=start_step,
            opt_s_state=opt_s_state, opt_a_state=opt_a_state,
            sch_s_state=sch_s_state, sch_a_state=sch_a_state,
            checkpoint_callback=checkpoint_cb,
        )

        if result == 'stopped':
            training_state['status'] = 'stopped'
        else:
            try:
                model_dir = _save_final_model(model, model_root, config)
                training_state['model_dir'] = model_dir
            except Exception as e:
                training_state['save_error'] = str(e)
            training_state['status'] = 'done'

    except Exception as e:
        training_state['status'] = 'error'
        training_state['error']  = f"{e}\n{traceback.format_exc()}"


# ────────────────────────────────────────────────────────────────────────────────
# EfficientAD 핵심 구성 요소
# ────────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """EfficientNet-B0 2-scale Teacher (완전 고정)
      scale1: features[:3] → stride-4, out_ch  → Student/AE 매칭 타깃
      scale2: features[:2] → stride-2, 16ch 원본 → Direct 비교 전용 (고해상도)

    scale2 선택 이유:
      - stride-2 → 256px 입력 시 128×128 특징 맵
      - 3px 스크래치 = 128×(3/256) ≈ 1.5 특징 셀 → 명확히 감지 가능
      - 이전 stride-8(32×32): 3px → 0.4 특징 셀 → 신호 완전 소실
      - 투영 없이 raw 16ch 사용 → 임의 선형조합이 신호를 뭉개지 않음
    """
    def __init__(self, out_channels=256):
        super().__init__()
        base  = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        feats = list(base.features)
        self.enc_12 = nn.Sequential(*feats[:2])   # stride-2, 16ch (직접 비교용 고해상도)
        self.enc_3  = feats[2]                    # stride-2→4, 24ch
        self.proj   = nn.Conv2d(24, out_channels, 1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """반환: (scale1, scale2) = (out_ch H/4, 16ch H/2)"""
        with torch.no_grad():
            f2 = self.enc_12(x)   # (B, 16, H/2, W/2) — raw 직접 비교용
            f3 = self.enc_3(f2)   # (B, 24, H/4, W/4)
            return self.proj(f3), f2


class StudentNet(nn.Module):
    """Teacher 출력 해상도(H/4)에 맞춰 설계된 Student.
    stride-2 레이어 2개 → H/4 출력 (Teacher의 새로운 해상도와 일치).
    GroupNorm 사용: 배치=1 단일 이미지 처리 시에도 안정적으로 동작.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   64,  3, stride=1, padding=1), nn.GroupNorm(8,   64),  nn.ReLU(True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
            nn.Conv2d(256, out_channels, 1),
        )
        # stride: 1→2→2 = H/4 출력. 마지막 stride=1 conv로 용량 확보(수용야 확장 없이)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class AENet(nn.Module):
    """입력 이미지 → Teacher 특징 공간으로 재구성하는 오토인코더.

    병목 설계 기준:
      - 너무 타이트 → 정상 이미지도 재구성 실패 → 정상/이상 구분 불가
      - 너무 여유 → 결함도 재구성 성공 → 이상 탐지 불가
      - 적정: 병목 H/16 (16×16×128 = 32,768값), Teacher 출력 H/4 대비 압축비 ~32:1
      - 3px 스크래치 → 각 병목 셀이 16×16 입력픽셀 커버 → 결함 정보 탈락 → 재구성 실패
    """
    def __init__(self, out_channels=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(True),    # H/2
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(True),    # H/4
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(True),    # H/8
            nn.Conv2d(128, 128, 4, 2, 1), nn.ReLU(True),    # H/16  ← 병목 (압축비 ~32:1)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, 2, 1), nn.ReLU(True),   # H/8
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.ReLU(True),   # H/4  ← Teacher 해상도
            nn.Conv2d(256, out_channels, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.dec(self.enc(x))


class EfficientAD:
    """
    수정된 EfficientAD (일시정지/재시작 지원)
    """
    def __init__(self, image_size=256, out_channels=256, device='cpu'):
        self.image_size   = image_size
        self.out_channels = out_channels
        self.device       = device

        self.teacher = FeatureExtractor(out_channels).to(device)
        self.student = StudentNet(out_channels).to(device)
        self.ae      = AENet(out_channels).to(device)

        self.t_mean         = None
        self.t_std          = None
        self.threshold_auto = None
        self._train_scores  = []
        self._map_q001      = None   # 정상 이미지 픽셀맵 1퍼센타일 (히트맵 정규화용)
        self._map_q999      = None   # 정상 이미지 픽셀맵 99퍼센타일
        self._threshold_p95 = None   # 학습 점수 95퍼센타일 임계값
        self._threshold_p99 = None   # 학습 점수 99퍼센타일 임계값
        self._st_mean_g     = None   # 학습 정상 이미지 ST 맵 전체 평균 (Z-정규화용)
        self._st_std_g      = None   # 학습 정상 이미지 ST 맵 전체 표준편차
        self._ae_mean_g     = None   # 학습 정상 이미지 AE 맵 전체 평균
        self._ae_std_g      = None   # 학습 정상 이미지 AE 맵 전체 표준편차
        self._feat_mean_loc   = None   # 위치별 Teacher 특징 평균 (1,C,H/4,W/4) CPU
        self._feat_std_loc    = None   # 위치별 Teacher 특징 표준편차 (1,C,H/4,W/4) CPU
        self._direct_mean_g   = None   # Direct 맵 전역 평균 (점수 Z-정규화용)
        self._direct_std_g    = None   # Direct 맵 전역 표준편차
        self._direct_map_q001 = None   # Direct 맵 0.1 퍼센타일 (히트맵 정규화용)
        self._direct_map_q999 = None   # Direct 맵 99.9 퍼센타일

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # 학습용 증강: 색상 지터만 사용 (공간 변환 제거)
        # RandomCrop / RandomHorizontalFlip은 Student 입력과 Teacher 목표의
        # 공간 위치 대응을 깨뜨려 이상 부위 국소화를 불가능하게 만드므로 제거.
        # 색상 지터는 공간 구조를 유지하므로 안전하게 사용 가능.
        self.aug_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _prep(self, img):
        return self.transform(img.convert('RGB')).unsqueeze(0).to(self.device)

    def _aug_prep(self, img):
        return self.aug_transform(img.convert('RGB')).unsqueeze(0).to(self.device)

    def fit(self, train_images, epochs=10, lr=2e-4, progress_callback=None,
            pause_event: threading.Event = None,
            stop_event:  threading.Event = None,
            start_epoch: int = 0,
            start_step_in_epoch: int = 0,
            opt_s_state=None, opt_a_state=None,
            sch_s_state=None, sch_a_state=None,
            checkpoint_callback=None):
        """
        학습 메인 루프.
        - pause_event가 set되면 체크포인트 저장 후 대기
        - stop_event가 set되면 즉시 'stopped' 반환
        - start_epoch / start_step_in_epoch 로 체크포인트에서 재개 가능
        """

        # ── 1. Teacher 특징 통계 계산
        if self.t_mean is None:
            f3_list, f4_list = [], []
            with torch.no_grad():
                for img in train_images:
                    f3, f4 = self.teacher(self._prep(img))
                    f3_list.append(f3)
                    f4_list.append(f4)
            feats3 = torch.cat(f3_list)   # (N, out_ch, H/4, W/4)
            feats4 = torch.cat(f4_list)   # (N, 16,     H/2, W/2) — raw stride-2 특징
            self.t_mean = feats3.mean(dim=(0, 2, 3), keepdim=True).to(self.device)
            self.t_std  = feats3.std(dim=(0, 2, 3),  keepdim=True).clamp(min=1e-6).to(self.device)
            # scale2(stride-2) 위치별 통계 — 3px 스크래치도 1~2셀에 반영되는 고해상도
            self._feat_mean_loc = feats4.mean(dim=0, keepdim=True).cpu()   # (1,16,H/2,W/2)
            self._feat_std_loc  = feats4.std(dim=0,  keepdim=True).clamp(min=1e-4).cpu()

        # ── 2. 옵티마이저 / 스케줄러 (Student + AE만 학습, Teacher는 동결)
        opt_s = torch.optim.Adam(list(self.student.parameters()), lr=lr, weight_decay=1e-5)
        opt_a = torch.optim.Adam(list(self.ae.parameters()),      lr=lr, weight_decay=1e-5)
        sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=epochs)
        sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=epochs)

        if opt_s_state: opt_s.load_state_dict(opt_s_state)
        if opt_a_state: opt_a.load_state_dict(opt_a_state)
        if sch_s_state: sch_s.load_state_dict(sch_s_state)
        if sch_a_state: sch_a.load_state_dict(sch_a_state)

        # Teacher는 항상 eval() — BN running stats 오염 방지
        self.teacher.eval()
        self.student.train()
        self.ae.train()

        total       = len(train_images) * epochs
        global_step = start_epoch * len(train_images) + start_step_in_epoch
        _pause_ckpt_saved = False

        for ep in range(start_epoch, epochs):
            # 에폭마다 순서 셔플 (에폭 번호를 시드로 고정 → 재개 시 재현 가능)
            order = list(range(len(train_images)))
            random.Random(ep).shuffle(order)

            for si, img_idx in enumerate(order):

                # 재개 시 이미 완료된 step 건너뜀
                if ep == start_epoch and si < start_step_in_epoch:
                    global_step += 1
                    continue

                # ── 중지 체크
                if stop_event and stop_event.is_set():
                    return 'stopped'

                # ── 일시정지 체크
                if pause_event and pause_event.is_set():
                    if not _pause_ckpt_saved and checkpoint_callback:
                        checkpoint_callback(ep, si, opt_s, opt_a, sch_s, sch_a)
                        _pause_ckpt_saved = True
                    while pause_event.is_set():
                        time.sleep(0.1)
                        if stop_event and stop_event.is_set():
                            return 'stopped'
                    _pause_ckpt_saved = False

                img = train_images[img_idx]

                # ── 학습 스텝: 증강 이미지로 Student/AE, Teacher는 clean 이미지
                x_aug   = self._aug_prep(img)      # 증강 적용
                x_clean = self._prep(img)           # Teacher용 원본

                with torch.no_grad():
                    t_feat, _ = self.teacher(x_clean)   # scale1만 사용 (student/AE 타깃)
                    t_norm = (t_feat - self.t_mean) / self.t_std

                # Student
                s_out = self.student(x_aug)
                if s_out.shape[2:] != t_norm.shape[2:]:
                    s_out = F.interpolate(s_out, t_norm.shape[2:],
                                          mode='bilinear', align_corners=False)
                loss_s = F.mse_loss(s_out, t_norm)
                opt_s.zero_grad()
                loss_s.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                opt_s.step()

                # AE
                ae_out = self.ae(x_aug)
                if ae_out.shape[2:] != t_norm.shape[2:]:
                    ae_out = F.interpolate(ae_out, t_norm.shape[2:],
                                           mode='bilinear', align_corners=False)
                loss_a = F.mse_loss(ae_out, t_norm)
                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(self.ae.parameters(), max_norm=1.0)
                opt_a.step()

                global_step += 1
                if progress_callback:
                    progress_callback(global_step, total, loss_s.item(), loss_a.item())

            sch_s.step(); sch_a.step()

        # Teacher가 frozen이므로 통계 재계산 불필요 — eval 모드 전환만
        self.student.eval()
        self.ae.eval()

        # ── 1단계: ST / AE / Direct 맵 한 번에 수집
        train_st_np     = []
        train_ae_np     = []
        train_direct_np = []   # PaDiM-style 직접 비교 맵
        use_direct = self._feat_mean_loc is not None
        with torch.no_grad():
            for img in train_images:
                x          = self._prep(img)
                t_feat, t_feat4 = self.teacher(x)   # scale1(H/4), scale2(H/8)
                t_norm = (t_feat - self.t_mean) / self.t_std
                s_out  = self.student(x)
                ae_out = self.ae(x)
                if s_out.shape[2:]  != t_norm.shape[2:]:
                    s_out  = F.interpolate(s_out,  t_norm.shape[2:], mode='bilinear', align_corners=False)
                if ae_out.shape[2:] != t_norm.shape[2:]:
                    ae_out = F.interpolate(ae_out, t_norm.shape[2:], mode='bilinear', align_corners=False)
                st_map = torch.mean((t_norm - s_out)  ** 2, dim=1, keepdim=True)
                ae_map = torch.mean((t_norm - ae_out) ** 2, dim=1, keepdim=True)
                up_st  = F.interpolate(st_map, (self.image_size, self.image_size),
                                       mode='bilinear', align_corners=False)
                up_ae  = F.interpolate(ae_map, (self.image_size, self.image_size),
                                       mode='bilinear', align_corners=False)
                train_st_np.append(up_st.squeeze().cpu().numpy())
                train_ae_np.append(up_ae.squeeze().cpu().numpy())
                if use_direct:
                    # scale2(stride-8) 특징으로 직접 비교
                    diff = (t_feat4.cpu() - self._feat_mean_loc) / self._feat_std_loc
                    dist = (diff ** 2).mean(dim=1, keepdim=True)
                    up_d = F.interpolate(dist, (self.image_size, self.image_size),
                                         mode='bilinear', align_corners=False)
                    train_direct_np.append(up_d.squeeze().numpy())

        # ── 2단계: 전역 Z-정규화 통계 (ST / AE / Direct 각각)
        all_st = np.concatenate([m.flatten() for m in train_st_np])
        all_ae = np.concatenate([m.flatten() for m in train_ae_np])
        self._st_mean_g = float(all_st.mean())
        self._st_std_g  = float(max(all_st.std(), 1e-8))
        self._ae_mean_g = float(all_ae.mean())
        self._ae_std_g  = float(max(all_ae.std(), 1e-8))
        if train_direct_np:
            all_direct = np.concatenate([m.flatten() for m in train_direct_np])
            self._direct_mean_g = float(all_direct.mean())
            self._direct_std_g  = float(max(all_direct.std(), 1e-8))

        # ── 3단계: 점수 계산 (ST + AE + Direct 3-way Z-정규화 결합)
        # Direct 맵 = 위치별 정상 분포와의 직접 거리 → 이상 위치에서 크게 이탈
        # 정상 이미지의 Direct Z-score ≈ 0 → 이상 이미지에서만 크게 양수
        scores   = []
        raw_maps = []
        for i, (st_np, ae_np) in enumerate(zip(train_st_np, train_ae_np)):
            st_z = (st_np - self._st_mean_g) / self._st_std_g
            ae_z = (ae_np - self._ae_mean_g) / self._ae_std_g
            if train_direct_np:
                direct_z = (train_direct_np[i] - self._direct_mean_g) / self._direct_std_g
                combined = 0.25 * st_z + 0.25 * ae_z + 0.5 * direct_z
            else:
                combined = 0.5 * st_z + 0.5 * ae_z
            flat  = combined.flatten()
            k     = max(1, int(len(flat) * 0.005))
            score = float(np.partition(flat, -k)[-k:].mean())
            scores.append(score)
            raw_maps.append(combined)

        self._train_scores  = scores
        arr                 = torch.tensor(scores, dtype=torch.float32)
        mu, sigma           = arr.mean().item(), arr.std().item()
        self.threshold_auto = mu + 2.0 * sigma

        all_maps = np.stack(raw_maps)
        self._map_q001 = float(np.percentile(all_maps, 0.1))
        self._map_q999 = float(np.percentile(all_maps, 99.9))

        self._threshold_p95 = float(np.percentile(scores, 95))
        self._threshold_p99 = float(np.percentile(scores, 99))

        # Direct 맵 히트맵 정규화 기준
        if train_direct_np:
            all_dm = np.stack(train_direct_np)
            self._direct_map_q001 = float(np.percentile(all_dm, 0.1))
            self._direct_map_q999 = float(np.percentile(all_dm, 99.9))

        return 'done'

    def _raw_maps(self, img):
        """ST/AE raw 맵을 numpy 배열로 반환 (full-res, Z-정규화 전)."""
        x      = self._prep(img)
        t_feat, _ = self.teacher(x)   # scale1만 사용 (student/AE 타깃)
        t_norm = (t_feat - self.t_mean) / self.t_std
        s_out  = self.student(x)
        ae_out = self.ae(x)
        if s_out.shape[2:]  != t_norm.shape[2:]:
            s_out  = F.interpolate(s_out,  t_norm.shape[2:], mode='bilinear', align_corners=False)
        if ae_out.shape[2:] != t_norm.shape[2:]:
            ae_out = F.interpolate(ae_out, t_norm.shape[2:], mode='bilinear', align_corners=False)
        st_map = torch.mean((t_norm - s_out)  ** 2, dim=1, keepdim=True)
        ae_map = torch.mean((t_norm - ae_out) ** 2, dim=1, keepdim=True)
        up_st  = F.interpolate(st_map, (self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        up_ae  = F.interpolate(ae_map, (self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        return up_st.squeeze().cpu().numpy(), up_ae.squeeze().cpu().numpy()

    def _direct_map_np(self, img):
        """PaDiM-style: scale2(stride-8) 특징으로 위치별 정상 분포와의 거리 맵.
        scale2는 더 깊은 의미 특징 → 구조 변형 / 나사 결함 탐지에 유리.
        """
        x = self._prep(img)
        with torch.no_grad():
            _, t_feat4 = self.teacher(x)         # scale2(H/8,W/8)
        diff = (t_feat4.cpu() - self._feat_mean_loc) / self._feat_std_loc
        dist = (diff ** 2).mean(dim=1, keepdim=True)   # (1,1,H/8,W/8)
        up   = F.interpolate(dist, (self.image_size, self.image_size),
                             mode='bilinear', align_corners=False)
        return up.squeeze().numpy()   # (H,W)

    def _combine_maps(self, st_np, ae_np, weight_ae: float):
        """ST/AE 맵을 Z-정규화 후 weight_ae 비율로 결합."""
        if self._st_mean_g is not None:
            st_z = (st_np - self._st_mean_g) / self._st_std_g
            ae_z = (ae_np - self._ae_mean_g) / self._ae_std_g
        else:
            st_z, ae_z = st_np, ae_np   # 학습 미완료 시 raw 값 사용
        return (1 - weight_ae) * st_z + weight_ae * ae_z

    def _normalize_heatmap(self, hmap_raw: np.ndarray) -> np.ndarray:
        """학습 정상 이미지 픽셀맵 [0.1~99.9 퍼센타일]로 [0,1] 정규화.
        99.9 퍼센타일을 상한으로 사용하여 정상 이미지의 자연스러운 최대값이
        색상 범위 하단(파란색)에 머물게 하고, 이상 영역만 빨간색으로 표시.
        """
        if self._map_q001 is not None and self._map_q999 is not None:
            span = max(self._map_q999 - self._map_q001, 1e-8)
            return np.clip((hmap_raw - self._map_q001) / span, 0.0, 1.0)
        # 학습 미완료 시 이미지 자체 범위로 정규화
        lo, hi = hmap_raw.min(), hmap_raw.max()
        return np.clip((hmap_raw - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    @torch.no_grad()
    def predict(self, img, weight_ae=0.5):
        """점수: ST + AE + Direct 3-way Z-정규화 결합 (학습 보정과 일관성 유지).
        히트맵: PaDiM-style 직접 비교 맵 (정상 이미지와 다른 위치 표시).
        """
        st_np, ae_np = self._raw_maps(img)
        if self._st_mean_g is not None:
            st_z = (st_np - self._st_mean_g) / self._st_std_g
            ae_z = (ae_np - self._ae_mean_g) / self._ae_std_g
        else:
            st_z, ae_z = st_np, ae_np

        # Direct 맵 (teacher 호출 1회 추가, 판별력 대폭 향상)
        direct_np = None
        if self._feat_mean_loc is not None and self._direct_mean_g is not None:
            direct_np = self._direct_map_np(img)
            direct_z  = (direct_np - self._direct_mean_g) / self._direct_std_g
            combined  = 0.25 * st_z + 0.25 * ae_z + 0.5 * direct_z
        else:
            combined = 0.5 * st_z + 0.5 * ae_z

        flat  = combined.flatten()
        k     = max(1, int(len(flat) * 0.005))
        score = float(np.partition(flat, -k)[-k:].mean())

        # 히트맵: direct 맵 우선 (정상 분포와 위치별 차이)
        if direct_np is not None and self._direct_map_q001 is not None:
            span = max(self._direct_map_q999 - self._direct_map_q001, 1e-8)
            heatmap = np.clip((direct_np - self._direct_map_q001) / span, 0.0, 1.0)
        else:
            hmap_raw = (1 - weight_ae) * st_z + weight_ae * ae_z
            heatmap  = self._normalize_heatmap(hmap_raw)
        return score, heatmap


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
    img_arr   = np.array(img.resize((heatmap.shape[1], heatmap.shape[0])))
    hmap_norm = np.clip(heatmap, 0, 1)
    colormap  = cm.get_cmap('jet')
    hmap_colored = (colormap(hmap_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay      = (img_arr * (1 - alpha) + hmap_colored * alpha).astype(np.uint8)
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
    .metric-card { background: #1e1e2e; border-radius: 10px; padding: 16px; text-align: center; }
    .anomaly-high { color: #ff4b4b; font-size: 2rem; font-weight: bold; }
    .anomaly-low  { color: #00cc88; font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🔩 EfficientAD 나사 이상 감지 시스템")
st.caption("MVTec Screw 데이터셋 · Teacher-Student + AutoEncoder 기반 비지도 이상 감지")

# ── 사이드바 파라미터
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
    weight_ae = st.slider("AE 가중치 (히트맵 시각화)", 0.0, 1.0, 0.5, 0.05,
                          help="히트맵 시각화에만 영향. 이상 점수 계산은 항상 0.5 고정.")
    threshold_mode = st.selectbox(
        "임계값 방식",
        ["μ + k×σ", "퍼센타일 95%", "퍼센타일 99%", "수동"],
        index=0,
        help="μ+k×σ: 평균+k×표준편차 / 퍼센타일: 학습 정상 점수 상위 n% / 수동: 직접 지정",
    )
    k_sigma = st.slider("임계값 민감도 (k)", 1.0, 5.0, 2.0, 0.5,
                        help="낮출수록 민감(미탐↓, 오탐↑) / 높일수록 보수적(오탐↓, 미탐↑)",
                        disabled=(threshold_mode != "μ + k×σ"))
    manual_threshold = st.slider("수동 임계값", 0.0, 5.0, 1.0, 0.01,
                                 help="임계값 방식이 '수동'일 때 사용",
                                 disabled=(threshold_mode != "수동"))
    overlay_alpha = st.slider("히트맵 투명도", 0.1, 0.9, 0.5, 0.05)

    st.subheader("📂 데이터 / 저장")
    tar_path_input = st.text_input(
        "TAR 파일 경로",
        value=r"C:\Users\KDS21\Desktop\기업프로젝트\screw.tar.xz",
        help="screw_tar.xz 파일의 절대 경로"
    )
    model_root = st.text_input(
        "모델 저장 루트 디렉토리",
        value=str(Path(__file__).parent / "output"),
        help="config.yaml 및 models/ 폴더가 생성될 루트 경로"
    )

    st.subheader("💻 연산 장치")
    _cuda_ok = torch.cuda.is_available()
    _mps_ok  = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    _dev_opts = []
    if _cuda_ok:
        _dev_opts.append("GPU (CUDA)")
    if _mps_ok:
        _dev_opts.append("GPU (MPS)")
    _dev_opts.append("CPU")

    device_opt = st.radio("장치 선택", _dev_opts, index=0,
                          help="GPU가 있으면 CUDA가 기본 선택됩니다")
    if device_opt == "GPU (CUDA)":
        device = "cuda"
    elif device_opt == "GPU (MPS)":
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        _gpu_name   = torch.cuda.get_device_name(0)
        _vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _vram_free  = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_allocated(0)) / 1024**3
        st.success(f"**{_gpu_name}**")
        st.caption(f"VRAM: {_vram_free:.1f} GB 사용 가능 / {_vram_total:.1f} GB 전체")
    elif device == "mps":
        st.success("Apple Silicon GPU (MPS)")
    else:
        st.info("CPU 모드로 실행 중")

    # GPU를 선택할 수 없는 경우 진단 정보 표시
    if not _cuda_ok and not _mps_ok:
        with st.expander("⚠️ GPU를 사용할 수 없는 이유"):
            st.markdown(f"""
**PyTorch 버전:** `{torch.__version__}`
**CUDA 빌드 버전:** `{torch.version.cuda if torch.version.cuda else '없음 (CPU 전용 빌드)'}`

**원인 및 해결 방법:**

CUDA가 `None`이면 CPU 전용 PyTorch가 설치된 것입니다.
아래 명령으로 CUDA 버전을 확인 후 재설치하세요.

```
# CUDA 버전 확인 (cmd)
nvidia-smi

# CUDA 12.1 기준 재설치 예시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

[PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서
본인 CUDA 버전에 맞는 명령어를 확인하세요.
            """)
            if torch.version.cuda is None:
                st.error("현재 설치된 PyTorch는 CUDA를 지원하지 않습니다.")
            else:
                st.warning(f"CUDA {torch.version.cuda} 빌드이지만 GPU를 찾을 수 없습니다. 드라이버를 확인하세요.")

# ── 체크포인트 디렉토리 경로
checkpoint_dir = os.path.join(model_root, "models", "checkpoints")

# ── 세션 상태 초기화
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'train_config' not in st.session_state:
    st.session_state.train_config = {}
if 'training_state' not in st.session_state:
    st.session_state.training_state = {
        'status': 'idle', 'step': 0, 'total': 1,
        'loss_s': 0.0, 'loss_a': 0.0,
        'last_checkpoint': None, 'model_dir': None, 'error': None,
    }
if 'pause_event' not in st.session_state:
    st.session_state.pause_event = threading.Event()
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'train_images_cache' not in st.session_state:
    st.session_state.train_images_cache = None

# ── 탭
tab1, tab2, tab3 = st.tabs(["🎓 모델 학습", "🔍 이상 감지", "📊 배치 평가"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 학습
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("모델 초기화 및 정규화 학습")

    ts     = st.session_state.training_state
    status = ts['status']

    # ── 완료/오류/중지 상태 → trained 플래그 동기화
    if status == 'done' and not st.session_state.trained:
        st.session_state.trained = True

    # ══ 학습 시작 섹션 (idle / done / stopped / error 상태)
    if status in ('idle', 'done', 'stopped', 'error'):
        st.info("""
        EfficientAD는 **정상 이미지만**으로 Teacher 특징의 통계를 학습합니다.
        별도 라벨 없이 분포를 추정하는 비지도 학습 방식입니다.
        """)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **현재 설정:**
            - 채널: `{out_channels}` | 이미지: `{image_size}px` | 에폭: `{epochs}` | LR: `{lr}`
            - 학습 이미지: 최대 `{max_train}`장 | 장치: `{device}`
            - 저장 경로: `{model_root}`
            """)
        with col2:
            train_btn = st.button("🚀 학습 시작", type="primary", use_container_width=True)

        # ── 체크포인트에서 재시작 섹션
        ckpt_files = sorted(Path(checkpoint_dir).glob('*.ckpt'),
                            key=lambda p: p.stat().st_mtime, reverse=True) \
                     if os.path.exists(checkpoint_dir) else []

        if ckpt_files:
            with st.expander(f"⏩ 체크포인트에서 재시작 ({len(ckpt_files)}개 파일)"):
                ckpt_names = [p.name for p in ckpt_files]
                sel_ckpt   = st.selectbox("체크포인트 선택", ckpt_names, key='sel_ckpt')
                ckpt_path  = os.path.join(checkpoint_dir, sel_ckpt)

                # 선택된 체크포인트 정보 표시
                try:
                    ckpt_info = torch.load(ckpt_path, map_location='cpu')
                    cfg_info  = ckpt_info.get('config', {})
                    st.caption(
                        f"에폭: {ckpt_info['epoch']+1}/{cfg_info.get('epochs','?')} | "
                        f"스텝: {ckpt_info['step']} | "
                        f"채널: {cfg_info.get('out_channels','?')} | "
                        f"이미지크기: {cfg_info.get('image_size','?')}"
                    )
                except Exception:
                    st.caption("체크포인트 정보를 읽을 수 없습니다.")

                resume_btn = st.button("⏩ 이 체크포인트에서 재시작", use_container_width=True)

                if resume_btn:
                    if not os.path.exists(tar_path_input):
                        st.error(f"TAR 파일을 찾을 수 없습니다: `{tar_path_input}`")
                    else:
                        with st.spinner("체크포인트 로딩 및 이미지 준비 중..."):
                            ckpt = torch.load(ckpt_path, map_location=device)
                            cfg  = ckpt['config']

                            # 체크포인트 설정으로 모델 복원
                            model = EfficientAD(cfg['image_size'], cfg['out_channels'], device)
                            model.teacher.load_state_dict(ckpt['teacher_state'])
                            model.student.load_state_dict(ckpt['student_state'])
                            model.ae.load_state_dict(ckpt['ae_state'])
                            model.t_mean        = ckpt['t_mean']
                            model.t_std         = ckpt['t_std']
                            model._train_scores  = ckpt.get('train_scores', [])
                            model._map_q001      = ckpt.get('map_q001')
                            model._map_q999      = ckpt.get('map_q999')
                            model._threshold_p95 = ckpt.get('threshold_p95')
                            model._threshold_p99 = ckpt.get('threshold_p99')
                            model._st_mean_g      = ckpt.get('st_mean_g')
                            model._st_std_g       = ckpt.get('st_std_g')
                            model._ae_mean_g      = ckpt.get('ae_mean_g')
                            model._ae_std_g       = ckpt.get('ae_std_g')
                            model._feat_mean_loc   = ckpt.get('feat_mean_loc')
                            model._feat_std_loc    = ckpt.get('feat_std_loc')
                            model._direct_mean_g   = ckpt.get('direct_mean_g')
                            model._direct_std_g    = ckpt.get('direct_std_g')
                            model._direct_map_q001 = ckpt.get('direct_map_q001')
                            model._direct_map_q999 = ckpt.get('direct_map_q999')

                            # 학습 이미지 재로딩
                            train_data = load_images_from_tar(
                                tar_path_input, "screw/train/good/",
                                cfg.get('max_train', max_train))
                            train_imgs = [img for _, img in train_data]

                        st.session_state.model             = model
                        st.session_state.train_images_cache = train_imgs
                        st.session_state.train_config      = cfg

                        # 새 이벤트 생성
                        pause_ev = threading.Event()
                        stop_ev  = threading.Event()
                        st.session_state.pause_event = pause_ev
                        st.session_state.stop_event  = stop_ev

                        start_ep   = ckpt['epoch']
                        start_step = ckpt['step']
                        total_steps = len(train_imgs) * cfg['epochs']

                        new_ts = {
                            'status': 'training',
                            'step':   start_ep * len(train_imgs) + start_step,
                            'total':  total_steps,
                            'loss_s': 0.0, 'loss_a': 0.0,
                            'last_checkpoint': ckpt_path,
                            'model_dir': None, 'error': None,
                        }
                        st.session_state.training_state = new_ts

                        thread = threading.Thread(
                            target=_run_training_thread,
                            args=(model, train_imgs, cfg['epochs'], cfg['lr'],
                                  new_ts, pause_ev, stop_ev,
                                  checkpoint_dir, model_root, cfg,
                                  start_ep, start_step,
                                  ckpt['opt_s_state'], ckpt['opt_a_state'],
                                  ckpt['sch_s_state'], ckpt['sch_a_state']),
                            daemon=True,
                        )
                        thread.start()
                        st.session_state.training_thread = thread
                        st.rerun()

        # ── 학습 시작 버튼 처리
        if train_btn:
            if not os.path.exists(tar_path_input):
                st.error(f"파일을 찾을 수 없습니다: `{tar_path_input}`")
            else:
                with st.spinner("📦 정상 학습 이미지 로딩 중..."):
                    train_data = load_images_from_tar(tar_path_input, "screw/train/good/", max_train)
                    train_imgs = [img for _, img in train_data]

                st.session_state.train_images_cache = train_imgs

                if device == "cuda":
                    torch.cuda.empty_cache()
                model = EfficientAD(image_size, out_channels, device)
                st.session_state.model = model

                # 새 이벤트 (이전 학습과 분리)
                pause_ev = threading.Event()
                stop_ev  = threading.Event()
                st.session_state.pause_event = pause_ev
                st.session_state.stop_event  = stop_ev

                config = {
                    'model_type':   'efficientad',
                    'out_channels': out_channels,
                    'image_size':   image_size,
                    'epochs':       epochs,
                    'lr':           lr,
                    'max_train':    max_train,
                    'device':       device,
                    'created_at':   datetime.datetime.now().isoformat(),
                }
                st.session_state.train_config = config

                new_ts = {
                    'status': 'training',
                    'step':   0,
                    'total':  len(train_imgs) * epochs,
                    'loss_s': 0.0, 'loss_a': 0.0,
                    'last_checkpoint': None,
                    'model_dir': None, 'error': None,
                }
                st.session_state.training_state = new_ts
                st.session_state.trained = False

                thread = threading.Thread(
                    target=_run_training_thread,
                    args=(model, train_imgs, epochs, lr,
                          new_ts, pause_ev, stop_ev,
                          checkpoint_dir, model_root, config),
                    daemon=True,
                )
                thread.start()
                st.session_state.training_thread = thread
                st.rerun()

    # ══ 학습 진행 중 / 일시정지 상태 UI
    if status in ('training', 'paused'):
        step     = ts['step']
        total    = ts['total']
        progress = step / max(total, 1)

        # 진행 표시
        st.progress(progress)
        _ncols = 4 if device == "cuda" else 3
        _cols  = st.columns(_ncols)
        _cols[0].metric("진행", f"{step}/{total} steps")
        _cols[1].metric("Loss S",  f"{ts['loss_s']:.4f}")
        _cols[2].metric("Loss AE", f"{ts['loss_a']:.4f}")
        if device == "cuda":
            _vram_used = torch.cuda.memory_allocated(0) / 1024**3
            _vram_tot  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            _cols[3].metric("VRAM", f"{_vram_used:.1f} / {_vram_tot:.1f} GB")

        # 상태 배지
        if status == 'training':
            st.success("🟢 학습 진행 중...")
        else:
            st.warning("⏸ 일시정지됨 — 체크포인트 저장 완료")
            if ts.get('last_checkpoint'):
                st.caption(f"저장 위치: `{ts['last_checkpoint']}`")

        # 제어 버튼
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("⏸ 일시정지", disabled=(status == 'paused'),
                         use_container_width=True):
                st.session_state.pause_event.set()
                ts['status'] = 'paused'
                st.rerun()
        with btn_c2:
            if st.button("▶ 재시작", disabled=(status == 'training'),
                         use_container_width=True):
                st.session_state.pause_event.clear()
                ts['status'] = 'training'
                st.rerun()
        with btn_c3:
            if st.button("⏹ 학습 중지", use_container_width=True, type="secondary"):
                st.session_state.stop_event.set()

        # 스레드가 살아있는 동안 자동 새로고침
        thr = st.session_state.get('training_thread')
        if thr and thr.is_alive():
            time.sleep(0.5)
            st.rerun()

    # ══ 완료 상태
    if status == 'done':
        model = st.session_state.model
        if model and model._train_scores:
            arr       = torch.tensor(model._train_scores)
            mu, sigma = arr.mean().item(), arr.std().item()
            p95 = model._threshold_p95
            p99 = model._threshold_p99
            st.success(
                f"🎉 학습 완료! · μ={mu:.4f} σ={sigma:.4f} · "
                f"임계값(k=2): {mu+2*sigma:.4f} | P95: {p95:.4f} | P99: {p99:.4f}"
            )

            # 학습 점수 분포 히스토그램
            with st.expander("📊 정상 학습 점수 분포 (임계값 참고용)", expanded=True):
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(model._train_scores, bins=30, color='steelblue', alpha=0.75, label='정상 점수')
                ax.axvline(mu + 2 * sigma, color='orange', linestyle='--', linewidth=1.5,
                           label=f'μ+2σ = {mu+2*sigma:.4f}')
                ax.axvline(mu + 3 * sigma, color='red',    linestyle='--', linewidth=1.5,
                           label=f'μ+3σ = {mu+3*sigma:.4f}')
                if p95 is not None:
                    ax.axvline(p95, color='green',  linestyle=':',  linewidth=1.5,
                               label=f'P95 = {p95:.4f}')
                if p99 is not None:
                    ax.axvline(p99, color='purple', linestyle=':',  linewidth=1.5,
                               label=f'P99 = {p99:.4f}')
                ax.set_xlabel('이상 점수')
                ax.set_ylabel('빈도')
                ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=True)
                plt.close()
                st.caption("임계값이 정상 점수 분포 오른쪽 끝에 위치할수록 미탐(FN)이 줄어듭니다.")

        if ts.get('model_dir'):
            st.info(f"모델 저장 위치: `{ts['model_dir']}`")
        if ts.get('save_error'):
            st.warning(f"모델 저장 오류: {ts['save_error']}")

    if status == 'stopped':
        st.warning("⏹ 학습이 중지되었습니다.")

    if status == 'error':
        st.error(f"학습 중 오류 발생:\n```\n{ts.get('error', '')}\n```")

    # ── 모델 준비 완료 배너
    if st.session_state.trained:
        cfg = st.session_state.train_config
        st.success(
            f"✅ 모델 준비 완료 — 채널: `{cfg.get('out_channels')}`, "
            f"학습: `{cfg.get('max_train')}`장, 에폭: `{cfg.get('epochs')}`"
        )


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
        true_label  = None

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
                    sel   = st.selectbox("이미지 선택", names)
                    idx   = names.index(sel)
                    img_to_test = items[idx][1]
        else:
            uploaded = st.file_uploader("이미지 업로드 (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
            if uploaded:
                img_to_test = Image.open(uploaded).convert('RGB')
                true_label  = "알 수 없음"

        if img_to_test and st.button("🔍 분석 실행", type="primary"):
            model = st.session_state.model
            with st.spinner("이상 점수 계산 중..."):
                score, heatmap = model.predict(img_to_test, weight_ae)

            _sc = model._train_scores
            if threshold_mode == "μ + k×σ" and _sc:
                threshold = torch.tensor(_sc).mean().item() + k_sigma * torch.tensor(_sc).std().item()
            elif threshold_mode == "퍼센타일 95%" and model._threshold_p95 is not None:
                threshold = model._threshold_p95
            elif threshold_mode == "퍼센타일 99%" and model._threshold_p99 is not None:
                threshold = model._threshold_p99
            else:
                threshold = manual_threshold
            is_anomaly = score >= threshold
            verdict    = "🔴 이상 감지" if is_anomaly else "🟢 정상"

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
            mc1.metric("이상 점수", f"{score:.4f}", delta=f"임계값: {threshold:.4f}")
            mc2.metric("판정", verdict)
            if true_label:
                mc3.metric("실제 레이블", true_label)

            if is_anomaly:
                st.error(f"**{verdict}** — 점수 {score:.4f} ≥ 임계값 {threshold:.4f}")
            else:
                st.success(f"**{verdict}** — 점수 {score:.4f} < 임계값 {threshold:.4f}")


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
                model   = st.session_state.model
                results = []
                prog    = st.progress(0)
                status_el    = st.empty()
                total_types  = len(defect_types)

                # 임계값 1회 계산 (루프 밖)
                _sc = model._train_scores
                if threshold_mode == "μ + k×σ" and _sc:
                    threshold = torch.tensor(_sc).mean().item() + k_sigma * torch.tensor(_sc).std().item()
                elif threshold_mode == "퍼센타일 95%" and model._threshold_p95 is not None:
                    threshold = model._threshold_p95
                elif threshold_mode == "퍼센타일 99%" and model._threshold_p99 is not None:
                    threshold = model._threshold_p99
                else:
                    threshold = manual_threshold

                for ti, dtype in enumerate(defect_types):
                    status_el.info(f"처리 중: test/{dtype}")
                    items = load_images_from_tar(tar_path_input, f"screw/test/{dtype}/",
                                                 max_images=max_per_class)
                    for name, img in items:
                        score, _ = model.predict(img, weight_ae)
                        results.append({
                            '파일':    name.split('/')[-1],
                            '유형':    dtype,
                            '실제':    '정상' if dtype == 'good' else '이상',
                            '이상점수': round(score, 4),
                            '판정':    '이상' if score >= threshold else '정상',
                            '정답':    (dtype == 'good') == (score < threshold),
                        })
                    prog.progress((ti + 1) / total_types)

                status_el.success("✅ 평가 완료!")

                import pandas as pd
                df = pd.DataFrame(results)

                tp  = len(df[(df['실제'] == '이상') & (df['판정'] == '이상')])
                tn  = len(df[(df['실제'] == '정상') & (df['판정'] == '정상')])
                fp  = len(df[(df['실제'] == '정상') & (df['판정'] == '이상')])
                fn  = len(df[(df['실제'] == '이상') & (df['판정'] == '정상')])
                acc  = (tp + tn) / max(len(df), 1)
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                f1   = 2 * prec * rec / max(prec + rec, 1e-6)

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
                ax.axvline(threshold, color='red', linestyle='--', label=f'임계값={threshold:.4f}')
                ax.set_xlabel('이상 점수')
                ax.set_ylabel('빈도')
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.subheader("📄 상세 결과")
                st.dataframe(
                    df.style.map(
                        lambda v: 'color: red' if v == '이상' else 'color: green',
                        subset=['판정']
                    ),
                    use_container_width=True
                )
