# common.py — 모든 복습 스크립트가 같이 쓰는 경로와 장치 설정 (빈칸 없음)
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent                 # review_nfcvae/
DATA = HERE.parent / 'NFCVAE' / 'classifier'           # train/test_voxel.npy, energies, dataset_2_2.hdf5가 있는 곳
OUT  = HERE                                            # 복습에서 새로 만드는 .pt/.npy/.png는 전부 여기에 저장 (원본을 덮어쓰지 않음)

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')