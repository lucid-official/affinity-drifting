"""Drifting Models × CLIP 텍스트 조건 통합 (Lucid 구현)."""

import os

import certifi

# python.org 빌드의 Python은 시스템 CA를 쓰지 않아 가중치 다운로드에서 인증서 오류가 난다.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
