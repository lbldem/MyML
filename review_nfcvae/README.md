# NF + CVAE 복습 (β = 1, temperature 없음)

빈칸 `___` 38개([B1]–[B38])를 채운 뒤, 이 폴더에서 순서대로 실행한다.

| 순서 | 파일 | 하는 일 | 만드는 파일 |
|---|---|---|---|
| 1 | step1_model.py | Hurdle CVAE 설계도 (`python step1_model.py`로 shape 확인) | 없음 |
| 2 | step2_train.py | VAE 학습, 30 epoch | calo_cvae_hurdle.pt, norm_params_voxel.pt |
| 3 | step3_collect_z.py | encoder로 z 10만 개 수집 | train_z.npy, train_c.npy |
| 4 | step4_flow_model.py | NF 설계도 (`python step4_flow_model.py`로 확인) | 없음 |
| 5 | step5_train_nf.py | NF prior 학습, 50 epoch | z_prior_nf.pt |
| 6 | step6_generate.py | NF prior → decoder 생성, 실제와 비교 | review_nf_prior.png |

- 데이터는 `../NFCVAE/classifier/`에서 읽고, 새로 만드는 파일은 전부 이 폴더에 저장한다(원본 .pt를 덮어쓰지 않음).
- step2가 오래 걸리면 EPOCHS를 줄여 먼저 끝까지 돌아가는지 확인하자. 학습을 건너뛰려면 step3, step6의 MODEL_PATH와 NORM_PATH를 `DATA / ...`로 바꾸면 된다.
- 기대값(이전 β = 1 결과): step6에서 hit voxels/event가 실제 약 1,617, NF prior 약 1,644.
- 정답은 ANSWERS.md. 다 채운 뒤에 열어 보자.
