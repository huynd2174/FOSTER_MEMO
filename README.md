# MEMO + FOSTER: Class-Incremental Learning with Feature Boosting, Compression, and Exemplar Memory

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.8-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

Kho lưu trữ mã cho phương pháp CIL kết hợp MEMO và FOSTER. Triển khai bao gồm:

- MEMO: Đóng băng lớp nông (shallow) để chia sẻ đặc trưng tổng quát và mở rộng module lớp sâu cho các tác vụ mới.
- FOSTER: Tăng cường (feature boosting) theo nguyên lý học phần dư và nén (feature compression) bằng tri thức chưng cất.
- Quản lý exemplar: Chọn mẫu đại diện (herding) và duy trì bộ đệm ví dụ để giảm quên thảm khốc.

Nếu bạn sử dụng mã trong kho này, vui lòng trích dẫn bài báo FOSTER:

```
@article{wang2022foster,
  title={FOSTER: Feature Boosting and Compression for Class-Incremental Learning},
  author={Wang, Fu-Yun and Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
  journal={arXiv preprint arXiv:2204.04662},
  year={2022}
}
```

## 1. Tổng quan phương pháp

- "Gradient Boosting" cho CIL: Huấn luyện module mới để học phần dư so với mô hình hiện tại nhằm hội tụ tới mục tiêu.

<p align="center"><img src='imgs/gradientboosting.png' width='900'></p>

- Feature Boosting (FOSTER): Nối nhánh mới và tối ưu hóa để giảm sai khác với mục tiêu; giảm lệch phân lớp bằng weight alignment và KD.

<p align="center"><img src='imgs/boosting.png' width='900'></p>

- Feature Compression (FOSTER): Chưng cất tri thức từ mô hình giáo viên (mô hình sau boosting) sang mô hình sinh viên nhỏ gọn.

<p align="center"><img src='imgs/compression.png' width='900'></p>

- MEMO: Đóng băng lớp nông và chỉ huấn luyện module mở rộng (lớp sâu) cho tác vụ mới; kết hợp với bộ đệm exemplar.

## 2. Kiến trúc và tích hợp

- Đóng băng lớp nông: `freeze_until` đã được hỗ trợ cho CIFAR (`stage_2`,...) và ImageNet (`layer2`,...). Có thể điều khiển qua tham số cấu hình `memo_freeze_until`.
- Mạng đa nhánh (FOSTERNet): Ghép đặc trưng từ các nhánh, dùng `oldfc` cho đặc trưng cũ và `fe_fc` cho nhánh mới. Logits tổng hợp tương ứng với F_current + F_new.
- Boosting: Kết hợp Cross-Entropy (CE) + KD tới logits cũ để học phần dư ổn định trên cả lớp cũ và mới.
- Compression: Chưng cất (KD temperature) + CE để huấn luyện mô hình sinh viên nhỏ gọn thay thế mô hình sau boosting.
- Exemplar: Chiến lược herding greedy và tính `class_means` để duy trì hiệu quả bộ nhớ.

## 3. Cài đặt phụ thuộc

- torch, torchvision
- tqdm, numpy

Khuyến nghị Python 3.8+, PyTorch 1.8+.

## 4. Chuẩn bị dữ liệu

- CIFAR-100: Tải về theo cấu trúc mặc định (thư mục `data/cifar-100-python` đã có ví dụ).
- ImageNet-100: Chuẩn bị list train/test trong `imagenet-sub/train.txt`, `imagenet-sub/eval.txt` (đã cung cấp ví dụ).

Lưu ý: Nếu bạn có biến đường dẫn dữ liệu tùy chỉnh, hãy chỉnh trong configs hoặc nơi đọc dữ liệu tương ứng (xem `utils/data_manager.py`).

## 5. Cấu hình quan trọng (MEMO + FOSTER)

- `model_name`: "memo-foster" để bật tích hợp MEMO+FOSTER.
- `memo_freeze_until`: Tên độ sâu cần đóng băng. CIFAR ResNet-32: `"stage_2"`; ImageNet ResNet-18: `"layer2"`.
- `kd_temperature` (compression), `kd_alpha` (trọng số KD trong compression).
- `T`: temperature cho KD ở boosting.
- `device`: Danh sách GPU/CUDA như `["0","1"]`. Nếu chạy CPU, dùng `[-1]`.
- Bộ nhớ exemplar: `memory_size`, `fixed_memory`, `memory_per_class`.

Ví dụ cấu hình CIFAR-100 (đã có sẵn): `configs/cifar/b0inc10.json`.

## 6. Huấn luyện

- CIFAR-100 (MEMO+FOSTER):

```
python main.py --config configs/cifar/b0inc10.json
```

- ImageNet-100 (FOSTER baseline):

```
python main.py --config configs/foster-imagenet100.json
```

- FOSTER-RMM (tùy chọn):

```
python main.py --config configs/foster-rmm.json
```

Gợi ý Windows/CPU: nếu không có CUDA, chỉnh `"device": [-1]` trong file cấu hình.

## 7. Ghi log, checkpoint và kết quả

- Log được ghi vào `logs/{model_name}/{dataset}/{init_cls}/{increment}/...`.
- Sau mỗi tác vụ (task), mã sẽ báo cáo Top-1/Top-5 theo CNN và (nếu có) NME.
- Kết quả mẫu (CIFAR-100):

| Protocols    | Reproduced Avg | Reported Avg |
| ------------ | -------------- | ------------ |
| B0 5 steps   | 73.88          | 72.54        |
| B0 10 steps  | 73.10          | 72.90        |
| B0 20 steps  | 70.59          | 70.65        |
| B50 5 steps  | 71.08          | 70.10        |
| B50 10 steps | 68.61          | 67.95        |
| B50 25 steps | 64.95          | 63.83        |
| B50 50 steps | 59.96          | -            |

<p align="center"><img src='imgs/performance.png' width='900'></p>

<p align="center"><img src='imgs/vis.png' width='900'></p>

## 8. Tùy chỉnh & mẹo thực nghiệm

- Điều chỉnh `memo_freeze_until` để cân bằng stability–plasticity. Với backbone ImageNet-Style, phổ biến là `layer2`.
- `kd_alpha` và `kd_temperature` trong compression giúp ổn định nén mà không giảm hiệu năng.
- `is_teacher_wa` / `is_student_wa`: tùy chọn weight alignment.
- Bộ đệm exemplar có thể tăng khi model được nén hiệu quả (nếu ngân sách cho phép).

## 9. Trích dẫn

Vui lòng trích dẫn FOSTER nếu bạn sử dụng mã này:

```
@article{wang2022foster,
  title={FOSTER: Feature Boosting and Compression for Class-Incremental Learning},
  author={Wang, Fu-Yun and Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
  journal={arXiv preprint arXiv:2204.04662},
  year={2022}
}
```

## 10. Ghi nhận và giấy phép

- Tham khảo các repo: [PyCIL](https://github.com/G-U-N/PyCIL), [Proser](https://github.com/zhoudw-zdw/CVPR21-Proser), [AutoAugment](https://github.com/DeepVoltaire/AutoAugment).
- Giấy phép: MIT (xem `LICENSE`).

## 11. Liên hệ

Nếu có câu hỏi, vui lòng liên hệ tác giả gốc bài FOSTER: Fu-Yun Wang (wangfuyun@smail.nju.edu.cn). Enjoy the code.
