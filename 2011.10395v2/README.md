# DQC differential-equation solver

Project Python mô phỏng trên CPU các thành phần chính của **Solving nonlinear differential equations with differentiable quantum circuits**, Kyriienko, Paine và Elfving (2021), [arXiv:2011.10395v2](https://arxiv.org/abs/2011.10395v2).

Bạn có thể import từng module để thử ý tưởng, hoặc chạy CLI để huấn luyện và xuất nghiệm, residual, tham số, đồ thị cùng cấu hình. Backend hiện tại là **noiseless statevector bằng NumPy/SciPy**. Đây là implementation nghiên cứu/giảng giải; các preset sáu qubit lấy cấu hình chính từ paper nhưng không phải cam kết tái lập đúng từng đường cong của tác giả.

## Chạy ngay trên máy này

Môi trường `.venv` đã được tạo và package đã được cài editable trong quá trình triển khai. Từ thư mục project, chạy bằng PowerShell:

```powershell
.venv\Scripts\python.exe -m dqc --config configs/quick_decay.json
```

Bài mặc định là ví dụ dễ kiểm tra $u'+u^2=0$, $u(0)=1$, có nghiệm $1/(1+x)$. Đây là bài minh họa bổ sung, không phải một thí nghiệm được đánh số trong paper. Lệnh trên huấn luyện mạch ba qubit và tạo kết quả trong `outputs/decay/`.

Để chọn interpreter trong VS Code, dùng `.venv\Scripts\python.exe`.

## Cài lại trên môi trường mới

Yêu cầu Python 3.10 trở lên. Ví dụ với `uv`:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

Hoặc dùng Python/pip nếu đã có Python phù hợp:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Package `dqc` chỉ cần NumPy, SciPy và Matplotlib. Pytest nằm trong extra `dev`. Không cần Qiskit, PennyLane, tài khoản quantum cloud hay tải dataset để chạy các ví dụ cơ bản.

## Cấu trúc

```text
.
├── pyproject.toml
├── README.md
├── configs/
│   ├── quick_decay.json
│   ├── paper_damped.json
│   ├── paper_nonlinear.json
│   ├── paper_coupled.json
│   └── paper_nozzle_upstream.json
├── src/dqc/
│   ├── gates.py               # R_x/y/z, CNOT, Pauli strings
│   ├── feature_maps.py        # Prod, Cheb, ChebT, fixed evolution, domain mapping
│   ├── ansatz.py              # HEA và alternating blocks
│   ├── observables.py         # Z, magnetization, Ising, spin-glass, weighted readout
│   ├── boundaries.py          # pinned / floating / optimized
│   ├── model.py               # hàm, đạo hàm, parameter Jacobian; shared / parallel
│   ├── losses.py              # MSE, MAE, KL và JS
│   ├── schedules.py           # constant / linear / tanh regularization weights
│   ├── objective.py           # differential + boundary + regularization loss
│   ├── optimizers.py          # Adam và analytic-Jacobian least squares
│   ├── grids.py               # uniform, Chebyshev, random, split nozzle grids
│   ├── experiments.py         # config và factory dựng bài toán/mô hình
│   ├── reporting.py           # metrics, CSV, NPZ, JSON và PNG
│   ├── cli.py
│   └── problems/
│       ├── base.py
│       ├── scalar.py          # Eq. (30), Eq. (32), nonlinear decay
│       ├── coupled.py         # Eqs. (33)-(34)
│       └── nozzle.py          # Eqs. (40)-(50), geometry và PDE RHS
├── examples/
│   ├── compare_feature_maps.py
│   ├── nonlinear_staged.py
│   └── nozzle_staged.py
├── tests/
│   ├── test_quantum.py
│   ├── test_problems.py
│   └── test_training.py
├── docs/paper_mapping.md
├── literature_review_DQC_2011.10395v2.md
└── understanding_differentiable_quantum_circuits.ipynb
```

## Những phần đã triển khai

- Feature maps Prod/Cheb/ChebT, affine domain mapping và evolution-enhanced map với Hamiltonian cố định, không phụ thuộc $x$.
- HEA với chuỗi quay `ZXZ` theo paper; có thể chọn `YZ` cho ví dụ CPU nhỏ. ABA gồm các block HEA xếp xen kẽ, có block ngắn ở biên.
- Readout một hoặc nhiều observable Hermitian; trọng số readout cố định hoặc được học; các observable Z, total magnetization, Ising và spin-glass.
- Đạo hàm theo $x$ bậc một và hai bằng parameter-shift đầy đủ chain rule, bao gồm các hạng chéo giữa những góc mã hóa.
- Jacobian theo tham số cho đầu ra và các đạo hàm; có cả gradient của trọng số cổ điển, offset và anchor floating boundary.
- Một register với nhiều đầu ra, hoặc nhiều register độc lập cùng tham gia một loss ghép cặp.
- Objective MSE/MAE, boundary penalty, regularization với lịch fading; optimizer Adam và least squares.
- Các bài toán scalar, coupled và steady nozzle; hàm đánh giá RHS của PDE nozzle theo thời gian khi đã biết đạo hàm không gian.

KL/JS được cung cấp cho **phân phối xác suất hợp lệ**, không dùng trực tiếp trên residual có dấu. Backend không mô phỏng shot noise/gate noise. Phần PDE nozzle chỉ đánh giá RHS, chưa có bộ tích phân PDE theo thời gian. Chưa triển khai digital feature maps, biến đầu vào nhiều chiều, hay evolution có Hamiltonian/thời gian phụ thuộc $x$.

## Lệnh chạy các bài toán

Ví dụ nhỏ, thích hợp để học và kiểm tra code:

```powershell
.venv\Scripts\python.exe -m dqc --problem nonlinear --depth 3 --points 80 --max-nfev 220 --output outputs/nonlinear_quick
.venv\Scripts\python.exe -m dqc --problem coupled --optimizer adam --steps 100 --output outputs/coupled_quick
.venv\Scripts\python.exe -m dqc --problem nozzle --optimizer adam --steps 100 --points 20 --output outputs/nozzle_upstream_quick
```

So sánh feature maps hoặc thêm evolution-enhanced stage:

```powershell
.venv\Scripts\python.exe examples/compare_feature_maps.py
.venv\Scripts\python.exe examples/nonlinear_staged.py --steps 100
```

Preset lớn hơn theo cấu hình chính của paper:

```powershell
.venv\Scripts\python.exe -m dqc --config configs/paper_damped.json
.venv\Scripts\python.exe -m dqc --config configs/paper_nonlinear.json
.venv\Scripts\python.exe -m dqc --config configs/paper_coupled.json
.venv\Scripts\python.exe -m dqc --config configs/paper_nozzle_upstream.json
```

Các preset sáu qubit dùng `ZXZ`, fixed total-magnetization readout và Adam; chúng có thể chạy lâu trên CPU do phải đánh giá nhiều mạch dịch góc cho mỗi gradient. Seed, góc khởi tạo và một số lựa chọn thực thi là của project này. Learning rate 0.01 trong preset damped là lựa chọn mặc định của implementation. Preset nonlinear tương ứng giai đoạn ChebT; để thêm evolution dùng ví dụ staged hoặc đổi `evolution_time`.

CLI ghi đè các trường config khi có tùy chọn tương ứng. Ví dụ:

```powershell
.venv\Scripts\python.exe -m dqc --config configs/paper_coupled.json --qubits 2 --depth 2 --steps 20 --output outputs/coupled_smoke
.venv\Scripts\python.exe -m dqc --help
```

## Cách dùng bằng Python

```python
import numpy as np
from dqc import (
    FeatureMap, HardwareEfficientAnsatz, Readout, Boundary,
    QuantumModel, Objective, fit_adam,
)
from dqc.problems import NonlinearDecay

model = QuantumModel(
    FeatureMap(n_qubits=3, kind="chebt"),
    HardwareEfficientAnsatz(n_qubits=3, depth=2, axes="YZ"),
    Readout.weighted_z(n_qubits=3, trainable=True),
    Boundary(kind="floating", x0=0, values=[1]),
)
problem = NonlinearDecay()
x_train = np.linspace(0, 0.9, 31)
objective = Objective(model, problem, x_train)
fit = fit_adam(objective, steps=100, learning_rate=0.01, seed=7)

x_test = np.linspace(0, 0.9, 200)
e = model.predict(fit.parameters, x_test, order=2, jacobian=True)
print(e.value.shape)          # (200, 1)
print(e.dx.shape)             # (200, 1): df/dx
print(e.dxx.shape)            # (200, 1): d²f/dx²
print(e.jacobians[0].shape)   # (200, 1, number_of_parameters)
print(e.jacobians[1].shape)   # mixed parameter/input derivative
```

Đầu ra luôn có shape `(số điểm, số hàm)`, kể cả bài một hàm. `order` là bậc đạo hàm đầu vào cần tính; dùng `order=0` khi chỉ cần giá trị để tiết kiệm chi phí. `prepare(x, order)` có thể dùng lại khi chỉ thay tham số mạch.

`x` là tọa độ vật lý; `theta`/vector parameters là các tham số được học. Qubit 0 là bit bên trái của statevector. Mỗi tham số góc thuộc đúng một cổng quay. Không áp parameter-shift trực tiếp lên nonlinear loss; project lấy Jacobian của expectation rồi dùng chain rule cổ điển.

### Chuyển miền tọa độ

```python
from dqc import AffineDomain, FeatureMap

mapping = AffineDomain.interval(2.0, 6.0, interior=0.8)
feature_map = FeatureMap(3, "chebt", domain=mapping)
# x in [2,6] -> z in [-0.8,0.8]. Derivatives are still w.r.t. physical x.
```

Đạo hàm cần `|z| < 1`. Giá trị hàm ở `z=±1` vẫn được hỗ trợ với `order=0`. Không chuẩn hóa miền lại giữa các stage nếu muốn tái sử dụng ý nghĩa của tham số đã học; ví dụ nozzle giữ identity mapping xuyên suốt.

### Shared register và các register riêng

`Readout.weighted_z(n_qubits=3, n_outputs=2)` tạo hai đầu ra từ cùng register/ansatz. Tạo `Boundary(..., values=[u10, u20])` tương ứng. Khi các trọng số fixed và có nhiều đầu ra, mỗi đầu ra dùng một tổ hợp Z được chỉ định lúc tạo model.

`ParallelModel([model1, model2])` ghép hai mô hình có tham số độc lập. `Objective(..., CoupledSystem(), ...)` vẫn tối ưu chúng cùng nhau vì residual chứa cả hai hàm.

### Regularization và lịch huấn luyện

```python
from dqc import Regularization
from dqc.schedules import TanhDecay

reg = Regularization(
    x=np.array([0.2, 0.7]),
    values=np.array([[0.83], [0.59]]),  # supplied guidance; not generated by trainer
    schedule=TanhDecay(total_steps=300, drop_step=100, width=0.05),
)
objective = Objective(model, problem, x_train, regularization=reg)
```

Theo Eq. (29), `TanhDecay` bắt đầu gần **hai lần** `weight`, không chuẩn hóa lại thành một. Objective dùng trung bình theo số điểm collocation, cộng các phương trình; boundary và regularization dùng tổng có trọng số như paper. Vì thế metric không phải trung bình trên cả điểm lẫn số phương trình.

Adam dùng schedule tại mỗi iteration và trả về **iterate cuối**, có cả cập nhật cuối cùng. `success=False` biểu thị chưa có tiêu chí chứng nhận hội tụ, không phải lỗi thực thi; xem message, history và validation metrics. Least squares phù hợp với objective MSE cố định, có Jacobian giải tích; schedule thay đổi theo iteration phải dùng Adam. Status hết ngân sách least squares cũng không phải chứng nhận nghiệm đã hội tụ.

## Nozzle hai giai đoạn

```powershell
.venv\Scripts\python.exe examples/nozzle_staged.py --steps 100
```

Lệnh trên chỉ học vùng upstream `[0,0.4]`. Muốn chạy stage 2, truyền CSV có **đúng năm** điểm downstream riêng biệt trong `[0.6,0.9]`:

```powershell
.venv\Scripts\python.exe examples/nozzle_staged.py --steps 100 --guidance path/to/downstream.csv
```

Header CSV cần là `x,density,temperature,velocity`; dữ liệu hữu hạn và các trường dương. Các giá trị này phải do bạn cung cấp từ nguồn hướng dẫn nghiệm thích hợp. Paper có mô tả năm điểm định hướng downstream nhưng không tabulate các nhãn cụ thể; project không tạo nhãn giả để gọi là dữ liệu tác giả.

Stage 2 dùng upstream prediction từ stage 1 làm regularization, thêm downstream labels, fading schedule và lưới tách `[0,0.4] ∪ [0.6,0.9]`. Ví dụ nhỏ dùng ba qubit với weighted readout; đây là triển khai workflow, không phải tái lập đúng Fig. 10. Cả regularization CSV và schedule được lưu cùng kết quả.

Mặc định `NozzleSteady` dùng conservation residual của Eqs. (40)-(42) khi bỏ time derivatives. Chọn `nozzle_residual="divided"` trong config để dùng Eqs. (48)-(50); dạng này kiểm tra mẫu số sonic. Hai residual có weighting khác nhau. Chưa có positivity constraint tự động: xem `physical_diagnostics` và residual toàn miền trước khi diễn giải profile là nghiệm vật lý.

## File kết quả và tải lại

Mỗi lệnh CLI tạo:

- `run.json`: config, phiên bản thư viện, số tham số, thời gian và validation metrics.
- `parameters.npz`: tham số cùng các điểm collocation.
- `history.json`: loss theo iteration/residual evaluation; không dùng chung ý nghĩa trục x cho Adam và least squares.
- `predictions.csv`: nghiệm, residual trên 401 điểm độc lập và reference nếu có.
- `solution.png`: nghiệm, loss và residual; tắt bằng `--no-plot`.

```python
import json
from pathlib import Path
import numpy as np
from dqc.experiments import ExperimentConfig, make_experiment

folder = Path("outputs/decay")
record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
model, problem, x_train = make_experiment(ExperimentConfig(**record["config"]))
parameters = np.load(folder / "parameters.npz")["parameters"]
prediction = model.predict(parameters, [0.15, 0.55], order=1)
```

Reference chỉ dùng trong báo cáo sau huấn luyện; `Objective` không gọi reference để lấy nhãn nghiệm. Với nozzle, không cung cấp reference toàn miền mặc định. Việc CSV có prediction trong khoảng throat không chứng minh residual đã được huấn luyện ở đó.

## Thêm bài toán mới

Kế thừa `dqc.problems.base.Problem` và khai báo `n_outputs`, `order`, `initial_values`, `labels`, `domain`. Hiện thực:

1. `residual(x, evaluation)` → array `(M, equations)`.
2. `partials(x, evaluation)` → tuple đạo hàm residual theo `f`, `f'`, ..., mỗi array `(M, equations, outputs)`.
3. `reference(x)` nếu có → array `(M, outputs)`; nếu không có, giữ mặc định `None`.

Các problem bundled có số phương trình bằng số outputs. Objective hiện kiểm tra shape theo quy ước hệ vuông này. Có thể dùng `order=2` cho ODE bậc hai; việc áp đủ điều kiện biên/đạo hàm đầu cần được thiết kế bổ sung. Boundary handler hiện cung cấp điều kiện giá trị tại một điểm cho mỗi output.

## Kiểm thử và đọc code theo paper

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Nếu môi trường giới hạn quyền ghi thư mục Temp, dùng thư mục tạm riêng trong project (chọn đường dẫn chưa chứa dữ liệu cần giữ):

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest-local
```

Pytest quản lý/xóa nội dung thư mục `basetemp`; không trỏ tùy chọn này vào thư mục dữ liệu của bạn. Các test kiểm tra quy ước cổng, đạo hàm theo công thức độc lập và sai phân, gradient toàn objective, residual các bài toán, boundary handling, training và lưu/tải kết quả.

Xem [bảng ánh xạ công thức sang module](docs/paper_mapping.md) để biết đọc file nào cho từng phần của paper. Ghi chú literature review và notebook cũ vẫn nằm ở thư mục gốc.

