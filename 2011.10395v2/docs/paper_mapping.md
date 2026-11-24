# Ánh xạ paper sang implementation

Nguồn: [arXiv:2011.10395v2](https://arxiv.org/abs/2011.10395v2), [PDF](https://arxiv.org/pdf/2011.10395v2). Số phương trình/hình theo bản v2. Backend là classical noiseless statevector; không phải code gốc Yao.jl của tác giả.

| Paper | API/file | Chi tiết |
|---|---|---|
| Eq. (1), Fig. 2 | `model.QuantumModel`, `observables.Readout` | Expectation-valued functions, một hoặc nhiều đầu ra |
| Eqs. (2), (8)-(10), Fig. 3 | `feature_maps.FeatureMap.prepare`, `QuantumModel.evaluate` | Dịch từng góc, chain rule, Jacobian theo tham số |
| Eq. (6) | `FeatureMap(..., kind="prod")` | Góc arcsin |
| Eq. (14) | `FeatureMap(..., kind="cheb")` | Góc 2 arccos cho mỗi qubit |
| Eq. (15) | `FeatureMap(..., kind="chebt")` | Góc 2j arccos, j đánh số từ 1 |
| Evolution-enhanced, Fig. 3(c), Fig. 7 | `FeatureMap(..., hamiltonian=H, time=tau)` | Unitary cố định sau product encoding; `examples/nonlinear_staged.py` |
| Fig. 5(a) | `ansatz.HardwareEfficientAnsatz` | `axes="ZXZ"` là chuỗi quay của paper |
| Fig. 5(b) | `ansatz.AlternatingBlocksAnsatz` | Block HEA xen kẽ, xử lý block ngắn tại biên |
| Sec. III C | `single_pauli`, `total_magnetization` | Readout một qubit hoặc tổng magnetization |
| Eq. (16) | `ising_hamiltonian` | Open-chain ZZ, X, Z; hệ số được truyền tường minh |
| Eq. (17) | `spin_glass` | Ma trận J đối xứng, lấy mỗi cặp i<j đúng một lần |
| Eq. (18) | `Readout(..., trainable=True)` | Trọng số tuyến tính cổ điển và gradient của chúng |
| Eqs. (19)-(23) | `objective.Objective` | Residual DE, penalty biên và cách tính trọng số |
| MSE, MAE | `losses.mse`, `losses.mae`, `Objective(distance=...)` | Dùng được để huấn luyện |
| KL, Jensen-Shannon | `losses.kl_divergence`, `losses.js_divergence` | API riêng cho phân phối hợp lệ; không gán KL cho residual có dấu |
| Eq. (24) | `Boundary("pinned", ...)` | Điều kiện biên nằm trong loss |
| Eqs. (25)-(26) | `Boundary("floating", ...)` | f = u0 + q(x) - q(x0); gradient anchor được giữ |
| Eq. (27) | `Boundary("optimized", ...)` | Thêm offset được học; vẫn có boundary loss |
| Eq. (28) | `objective.Regularization` | Điểm hướng dẫn và giá trị được cung cấp tường minh |
| Eq. (29) | `schedules.TanhDecay` | Giữ nguyên prefactor 1-tanh, bắt đầu gần 2 |
| Sec. III G | `QuantumModel` + nhiều readout, hoặc `ParallelModel` | Shared register / independent parameter blocks |
| Sec. III H, Fig. 1 | `Objective`, `fit_adam`, CLI | Workflow quantum-classical |
| Eq. (30) | `problems.DampedOscillator` | Tách rõ original residual và cosine-weighted residual |
| Eq. (32) | `problems.NonlinearOscillator` | Phi tuyến/dao động nhanh; reference bằng DOP853 chỉ phục vụ evaluation |
| Eqs. (33)-(34) | `problems.CoupledSystem` | Residual ghép cặp và analytic reference bằng matrix exponential |
| Eqs. (40)-(42) | `nozzle.transient_rhs`, `NozzleSteady(..., residual_form="conservation")` | PDE RHS và steady residual; chưa có transient PDE integrator |
| Eq. (43) | `nozzle.area`, `nozzle.log_area_derivative` | Geometry và derivative |
| Eq. (44) | `NozzleSteady.initial_values` | Inlet (1,1,0.1) |
| Eqs. (45)-(47) | `nozzle.transient_initial_conditions` | Điều kiện đầu theo không gian cho bài toán động |
| Eqs. (48)-(50) | `NozzleSteady.explicit_rhs`, `residual_form="divided"` | Kiểm tra singular denominator T-V² |
| Nozzle staged workflow, pp. 16-17 | `examples/nozzle_staged.py`, `grids.nozzle_stage_grid` | Stage 2 cần năm downstream labels do người dùng cung cấp |

## Quy ước đạo hàm

Với gate $R_P(\alpha)=\exp(-i\alpha P/2)$, parameter-shift expectation dùng hệ số $1/2$. Eq. (9) có hệ số in khác; project theo quy ước cổng trên và kiểm tra bằng nghiệm một qubit.

Đạo hàm bậc hai theo biến vật lý được tính:

$$
\frac{d^2q}{dx^2}
=\sum_i\varphi_i''\frac{\partial q}{\partial\varphi_i}
+\sum_{i,j}\varphi_i'\varphi_j'\frac{\partial^2q}{\partial\varphi_i\partial\varphi_j}.
$$

Code chứa cả hạng diagonal, cross Hessian và derivative của ánh xạ miền. Việc dịch tham số mạch để lấy Jacobian của f, f', f'' không dịch các trọng số classical hoặc nonlinear loss.

## Những khác biệt cần ghi khi tái hiện

1. Backend NumPy/SciPy thay cho Yao.jl. Toàn bộ expectation trong training được tính trực tiếp từ statevector, không lấy shots.
2. Ví dụ quick dùng mạch nhỏ, chuỗi `YZ` và trainable readout để chạy nhanh. Preset paper dùng sáu qubit, `ZXZ` và fixed magnetization; init và seed do project chọn.
3. Least squares là tiện ích bổ sung; Adam là optimizer được dùng trong các thí nghiệm paper. Trục history của hai optimizer khác nghĩa.
4. Công thức nghiệm Eq. (31) dùng hằng số nhân: u0 exp(-lambda*kappa*x) cos(lambda*x). Không dùng hằng số cộng in trong paper.
5. `DampedOscillator` mặc định giữ original tangent residual. `cosine_weighted` là lựa chọn tường minh làm thay đổi weighting và không phải sửa ngầm original equation tại cực.
6. Nozzle mặc định dùng conservation residual. Việc chọn dạng này thay đổi weighting so với explicit divided equations; không gọi kết quả là tái lập chính xác Fig. 10.
7. Stage 2 nozzle cần dữ liệu hướng dẫn downstream. Không cung cấp một profile downstream tự tạo rồi gán cho tác giả.
8. Chưa triển khai multivariable feature maps, digital encodings, overlap/SWAP-test differentiation cho generator tổng quát, noisy backend hoặc đầy đủ điều kiện biên nhiều điểm.

## Tái lập một run

Giữ `run.json`, `parameters.npz`, `history.json` và `predictions.csv`. Với staged nozzle cần thêm `regularization.csv` và `schedule.json`. `run.json` lưu cấu hình tạo model, seed và phiên bản NumPy/SciPy; các tham số khôi phục prediction nhưng không chứa trạng thái Adam để tiếp tục chính xác từng bước optimizer.

Các test là kiểm tra tính đúng của implementation; kết quả test không chứng minh quantum advantage hoặc accuracy của toàn bộ cấu hình paper.
