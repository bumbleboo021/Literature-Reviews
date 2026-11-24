# Đánh giá Lagrange parameter-shift cho bộ giải DQC

Ngày đánh giá: **02/11/2026**.

## 1. Kết luận chính

**Có thể khai thác ý tưởng generalized parameter-shift để mở rộng hoặc tối ưu một số bước đạo hàm của DQC, nhưng chưa có cơ sở kết luận thay trực tiếp bằng công thức trong tài liệu mới sẽ làm các thí nghiệm của paper gốc chạy nhanh hơn.**

| Trường hợp | Đánh giá |
|---|---|
| Giữ nguyên các cổng quay Pauli và chỉ tính đạo hàm bậc một | Không có cải thiện trực tiếp về tính chính xác; vẫn dùng quy tắc hai-shift |
| Thêm cổng native có generator nhiều trị riêng | Có thể dùng generalized rule để lấy đạo hàm trực tiếp, nếu biết phổ và xây dựng được recipe ổn định |
| Tính đồng thời giá trị, đạo hàm bậc một và hai của Chebyshev encoding | Có hướng giảm số cấu hình đánh giá nhờ dùng chung biến mã hóa và tái sử dụng các mẫu |
| Giảm tổng shots, thời gian training hoặc sai số nghiệm ODE/PDE | Chưa được chứng minh bởi tài liệu được gửi hoặc kiểm tra số nhỏ trong ghi chú này |
| Chép nguyên các công thức multiple-term từ bản PDF vào code | Chưa nên: cần làm rõ hệ số, pha phức, số term và bài toán xác định coefficients |

**Khuyến nghị:** ưu tiên một thí nghiệm kiểm chứng về *shared-parameter differentiation* cho Chebyshev encoding, đồng thời giữ two-term rule làm baseline. Không gán mọi lợi ích của generalized shift rules riêng cho cách suy dẫn Lagrange; cần so sánh với generalized Fourier parameter-shift đã có.

## 2. Tài liệu đã đọc và phạm vi đánh giá

### [S1] Paper DQC gốc

Oleksandr Kyriienko, Annie E. Paine, Vincent E. Elfving, *Solving nonlinear differential equations with differentiable quantum circuits*, Physical Review A 103, 052416 (2021), arXiv:2011.10395v2.

- [PDF trong repo](../2011.10395v2.pdf).
- [Hồ sơ arXiv](https://arxiv.org/abs/2011.10395v2).
- Phần liên quan trực tiếp: Sec. II; Sec. III A–B; Figs. 2–3; Eqs. (5)–(15); Sec. IV–V.

### [S2] Tài liệu người dùng cung cấp

Vu Tuan Hai và Le Bin Ho, *Lagrange interpolation approach for parameter-shift rule of general quantum gates*.

- [PDF trong repo](../_QCCSAA__Parameter_shift_rule.pdf), 8 trang.
- Phần liên quan: Sec. 3, Eqs. (4)–(20), Algorithm 1, Table 1; Sec. 4, Eqs. (23)–(27).
- Bản PDF được gửi có hai mục **Benchmarking** và **Applications** chưa có nội dung; một phiên bản Algorithm 1 còn ghi “Remove later”. Vì vậy, các tuyên bố giảm chi phí cần được đọc như những tuyên bố cần kiểm chứng trong chính bản tài liệu này, chưa phải kết quả benchmark DQC đã có. Nhận xét này không khẳng định các phiên bản khác hoặc toàn bộ repository của tác giả đều thiếu benchmark.

### [S3] Baseline generalized parameter-shift để đối chiếu

David Wierichs et al., *General parameter-shift rules for quantum gradients*, Quantum 6, 677 (2022), [DOI](https://doi.org/10.22331/q-2022-03-30-677), [arXiv:2107.12390v3](https://arxiv.org/abs/2107.12390v3).

S3 phân biệt việc tái dựng toàn bộ hàm với việc chỉ lấy đạo hàm: trong thiết lập phổ tần số phù hợp, generalized first-derivative rule dùng **2R** evaluations, trong khi full reconstruction thường dùng **2R+1**. Vì thế, không nên so một gradient recipe của S2 với chi phí tái dựng toàn bộ Fourier coefficients rồi kết luận nó luôn ít evaluations hơn generalized Fourier shift rules. [S3, phần Related work và Sec. 3]

### Cách đọc ghi chú này

- Nội dung ghi rõ “S1/S2 trình bày” là diễn giải tài liệu.
- Các phép đếm cho Cheb/ChebT, kiểm tra chuẩn hóa và đề xuất thí nghiệm là **phân tích của ghi chú này**.
- Kiểm tra số chỉ đánh giá derivative recipes trên mạch nhỏ không nhiễu. Không huấn luyện lại các bài toán trong paper và không sửa source solver.

## 3. DQC gốc đang tính đạo hàm gì?

DQC biểu diễn một hàm thử qua expectation:

$$
f_\theta(x)
=\langle 0|U_\varphi(x)^\dagger U_\theta^\dagger C
U_\theta U_\varphi(x)|0\rangle.
$$

Ở đây:

- $x$ là biến độc lập của phương trình, ví dụ thời gian hoặc vị trí.
- $\theta$ là các tham số được học.
- $\varphi_j(x)$ là góc của cổng mã hóa thứ $j$.

Trong S2, biến được gọi là $x$ trong $U(x)=e^{-ixG}$ là **tham số của một unitary**. Nó chỉ trùng với biến vật lý của DQC khi cách mã hóa thực sự có dạng đó. Để tránh nhầm lẫn, phần dưới gọi tham số gate là $a$.

### 3.1 Đạo hàm đầu vào để xây dựng residual

Với cổng $R_P(a)=e^{-iaP/2}$, $P^2=I$:

$$
\frac{\partial f}{\partial a}
=\frac{f(a+\pi/2)-f(a-\pi/2)}{2}.
$$

Do cùng $x$ có thể đi vào nhiều góc, DQC dùng chain rule:

$$
\frac{df_\theta}{dx}
=\sum_j\varphi_j'(x)
\frac{f_{\theta,j,+}(x)-f_{\theta,j,-}(x)}{2}.
$$

Mỗi hạng chỉ dịch một cổng mã hóa. Đây đã là đạo hàm giải tích của mạch lý tưởng, không có sai số cắt cụt kiểu sai phân hữu hạn. [S1, Fig. 3; Sec. III A]

### 3.2 Đạo hàm theo tham số để tối ưu

Nếu residual có dạng $r=f'+f^2$, gradient loss cần cả:

$$
\partial_{\theta_k}r
=\partial_{\theta_k}f'+2f\partial_{\theta_k}f.
$$

Do đó, đánh giá một cải tiến derivative backend cần xét cả đạo hàm hỗn hợp theo đầu vào và tham số. Chỉ kiểm tra $f'(x)$ đúng chưa đủ để khẳng định gradient training đúng.

## 4. Ý tưởng của phương pháp Lagrange

S2 xét một generator Hermitian cố định $G$ với các trị riêng phân biệt $\lambda_k$. Eq. (4) viết chính xác:

$$
e^{-iaG}
=\sum_{k=0}^{n-1}e^{-ia\lambda_k}
\prod_{l\ne k}\frac{G-\lambda_l I}{\lambda_k-\lambda_l}.
$$

Đây là nội suy hàm của ma trận tại các trị riêng. Trong điều kiện lý tưởng và với phổ chính xác, nó là một đẳng thức; không phải xấp xỉ đa thức tùy ý của hàm expectation trên lưới đầu vào.

Từ biểu diễn đó, S2 xây dựng các hệ số của tổng expectation ở nhiều shift. Ý tưởng quan trọng là **có thể suy ra recipe từ cấu trúc phổ của generator**, thay vì bắt buộc phân rã một gate tổng quát thành nhiều gate đơn giản rồi lấy đạo hàm từng gate.

Nhưng cách suy dẫn khác không tự động tạo ra:

- Ít cấu hình mạch hơn trong mọi trường hợp.
- Ít shots hơn cho cùng độ chính xác.
- Gradient chính xác hơn một analytic recipe vốn đã đúng.
- Thời gian tối ưu ngắn hơn hoặc nghiệm phương trình tốt hơn.

## 5. Giữ nguyên kiến trúc của S1: lợi ích trực tiếp hạn chế

### 5.1 Các cổng quay đơn qubit

Encoding của S1 dùng $R_y$, ansatz HEA dùng các cổng quay $R_z$–$R_x$–$R_z$ và CNOT cố định. Generator của mỗi cổng quay có hai trị riêng. Trường hợp $n=2$ của S2 quay lại two-term parameter-shift quen thuộc. [S1, Fig. 5; S2, Sec. 3.2]

Vì vậy, nếu tiếp tục dịch từng góc riêng và giữ nguyên circuit, không có cơ sở kỳ vọng thay cách suy dẫn hệ số bằng Lagrange sẽ giảm từ hai evaluations xuống một evaluation cho một đạo hàm tổng quát của loại gate này.

Cần thống nhất quy ước:

| Quy ước unitary | Generator | Shift tiện dùng | Hệ số của hiệu hai expectation |
|---|---|---|---|
| $e^{-iaP/2}$ | $P/2$, gap bằng 1 | $\pi/2$ | $1/2$ |
| $e^{-iaP}$ | $P$, gap bằng 2 | $\pi/4$ | $1$ |

Hai công thức tương ứng hai cách định nghĩa tham số; không được dùng cùng một góc rồi tráo hệ số giữa chúng.

### 5.2 Evolution-enhanced map trong thí nghiệm gốc

Nếu $e^{-iH\tau}$ có $H,\tau$ cố định và không phụ thuộc $x$, thì khi tính $df/dx$ không cần lấy đạo hàm riêng của khối đó. Nó có thể được gộp vào phần unitary/observable hiệu dụng sau encoding. Vì thế, việc khối này chứa Hamiltonian nhiều trị riêng chưa tự tạo lợi ích cho generalized PSR của đạo hàm đầu vào. [S1, mô tả evolution-enhanced map và thí nghiệm Eq. (32)]

Lợi ích có thể xuất hiện nếu **tham số của khối general-generator trở thành đại lượng cần lấy đạo hàm**, ví dụ học $\tau$, hoặc thiết kế một encoding $e^{-i\varphi(x)G}$ với $G$ cố định và biết phổ.

## 6. Trường hợp áp dụng hữu ích: general-generator gates

Giả sử thay một khối encoding bằng:

$$
U_\varphi(x)=e^{-i\varphi(x)G}.
$$

Khi đó có thể tìm một recipe:

$$
\frac{df}{da}(a)
=\sum_{k=1}^m w_k\left[f(a+s_k)-f(a-s_k)\right],
$$

rồi dùng $df/dx=\varphi'(x)\,df/da$. Đây là hướng có thể phù hợp với native collective gates hoặc các gate có nhiều trị riêng.

Điều kiện để có lợi thực tế:

1. Generator cố định; phổ hoặc các gap cần thiết có thể biết/tính được với chi phí hợp lý.
2. Device thực thi được các giá trị shift của chính khối gate đó.
3. Số term và shot budget tốt hơn baseline phân rã phù hợp.
4. Hệ số ổn định về số học; recipe được kiểm chứng độc lập.

Nếu $G$ là ma trận tổng quát kích thước $2^N$, việc chéo hóa và xây dựng recipe có thể rất tốn kém. Nếu chính generator phụ thuộc vào $x$ và không giao hoán với đạo hàm của nó, không thể áp dụng trực tiếp quy tắc cho một $G$ cố định.

Đây là mở rộng thiết kế của DQC, không phải bằng chứng rằng các cấu hình gate gốc đã được cải thiện.

## 7. Gom các cổng encoding dùng chung biến: phân tích riêng cho Cheb/ChebT

Đặt $t=\arccos x$ trong trường hợp identity domain mapping. Vì các $Y_j$ trên qubit khác nhau giao hoán:

$$
\bigotimes_{j=1}^N R_{y,j}(2a_j t)
=e^{-itG},\qquad G=\sum_{j=1}^N a_jY_j.
$$

- Cheb: $a_j=1$.
- ChebT: $a_j=j$.

Đặt $F_\theta(t)$ là expectation sau encoding và phần mạch còn lại cố định theo $t$. Có thể lấy đạo hàm theo $t$ của cả khối rồi chuyển về $x$:

$$
f_\theta'(x)=t'(x)F_\theta'(t),
$$

$$
f_\theta''(x)
=t''(x)F_\theta'(t)+[t'(x)]^2F_\theta''(t).
$$

Với $t=\arccos x$:

$$
t'(x)=-\frac{1}{\sqrt{1-x^2}},\qquad
t''(x)=-\frac{x}{(1-x^2)^{3/2}}.
$$

Nếu có affine mapping trước encoding, phải áp thêm chain rule tương ứng.

### 7.1 Đạo hàm bậc một: gom gate không tự động rẻ hơn

Số frequency/gap phân biệt là yếu tố quan trọng. Các số dưới dùng toàn bộ tập gap của generator, không giả định observable hoặc trạng thái làm một số frequency biến mất; cũng không khẳng định đây là số evaluation tối thiểu cho mọi recipe.

| Encoding | Tập gap dương của $G$ | Số gap $R$ | Gate-wise first derivative | Recipe đối xứng dùng toàn bộ gap |
|---|---|---:|---:|---:|
| Cheb | $2,4,\ldots,2N$ | $N$ | $2N$ | $2N$ |
| ChebT | $2,4,\ldots,2S$, $S=N(N+1)/2$ | $S$ | $2N$ | $2S$ |

Với sáu qubit:

- Cheb: so sánh 12 với 12 evaluations cho riêng đạo hàm bậc một.
- ChebT: recipe theo toàn bộ tập gap có 42 evaluations, trong khi gate-wise có 12.

Đây là lý do không nên mặc định gom mọi gate rồi chuyển sang generalized recipe. Cần chọn chiến lược theo cấu trúc phổ, đại lượng cần tính và ngân sách đo.

### 7.2 Đạo hàm bậc hai: cơ hội rõ hơn

Gate-wise differentiation có các hạng Hessian chéo giữa nhiều góc, nên một implementation trực tiếp có thể cần số term bậc $O(N^2)$.

Với các generator Cheb/ChebT ở trên, expectation là một trigonometric polynomial hữu hạn theo $t$, có dạng:

$$
F(t)=\sum_{r=-R}^{R}c_r e^{2irt}.
$$

Một tập **$2R+1$ mẫu theo $t$** cho phép tái dựng hàm này và tính đồng thời $F$, $F'$, $F''$. Đây là cách khai thác finite-spectrum structure; cách kiểm tra dưới dùng Fourier reconstruction trực tiếp, **không phải đã triển khai/xác nhận Algorithm 1 của S2**.

Điều kiện quan trọng: biểu thức một generator ở trên áp dụng cho block product encoding giao hoán hiện tại. Không mặc định gom được các encoding layers bị xen bởi những khối không giao hoán, hoặc các khối khác cũng phụ thuộc $x$.

**Shift $t$ phải được thực hiện qua góc gate:** $\varphi_j\mapsto\varphi_j+2a_js$. Không dịch $x$ bởi cùng một hằng số; cũng không mặc định gọi lại `arccos(cos(t+s))`, vì thao tác đó có thể gấp nhánh và tạo mạch khác.

## 8. Kiểm tra số đã thực hiện

### 8.1 Thiết lập

- Backend statevector NumPy hiện có trong repo; không sửa source solver.
- $N=2,3,6$; Cheb và ChebT.
- HEA depth 2, chuỗi quay `YZ`, weighted-Z readout.
- Seed tham số 42, seed readout 7.
- Kiểm tra tại $x\in\{0.1,0.35,0.8\}$.
- So sánh đạo hàm gate-wise hiện tại với việc tái dựng toàn cục theo $t$ từ các mẫu exact/noiseless.
- Kiểm tra cả $f$, $f'$ và $f''$; không chạy training hay giả lập shot noise.

### 8.2 Kết quả

| Qubits | Map | Số term/cấu hình baseline cho $f,f',f''$ | Mẫu reconstruction dùng chung | Sai khác tuyệt đối lớn nhất của $f'$ | Sai khác tuyệt đối lớn nhất của $f''$ |
|---:|---|---:|---:|---:|---:|
| 2 | Cheb | 13 | 5 | $2.91\times10^{-16}$ | $8.88\times10^{-16}$ |
| 2 | ChebT | 13 | 7 | $1.94\times10^{-16}$ | $2.66\times10^{-15}$ |
| 3 | Cheb | 25 | 7 | $4.44\times10^{-16}$ | $2.36\times10^{-15}$ |
| 3 | ChebT | 25 | 13 | $3.33\times10^{-16}$ | $4.35\times10^{-14}$ |
| 6 | Cheb | 85 | 13 | $3.77\times10^{-15}$ | $2.36\times10^{-13}$ |
| 6 | ChebT | 85 | 43 | $9.55\times10^{-15}$ | $1.99\times10^{-12}$ |

**Diễn giải đúng phạm vi:**

- Các sai khác nhỏ xác nhận phép chuyển biến và reconstruction trong các mạch kiểm tra nhất quán với baseline, trong độ chính xác số học.
- Cột baseline là số entries mà `FeatureMap.prepare(order=2)` hiện tạo, bằng $2N^2+2N+1$. Nó chưa khử mọi trùng lặp mạch và không phải số tối ưu được chứng minh cho paper S1.
- Ví dụ các shift diagonal $+\pi$ và $-\pi$ của một Pauli rotation cho cùng expectation; khử trùng này đã giảm baseline sáu qubit từ 85 xuống 79 entries trước các tối ưu khác.
- Các số không tính floating-boundary anchor, gradient theo $\theta$, nhiều nhóm đo không giao hoán hoặc shots. Kiểm tra này dùng pinned readout để so cùng một hàm circuit thô.
- **85 → 13 không có nghĩa toàn bộ training nhanh hơn 85/13 lần.** Đó chỉ là so sánh recipe entries/mẫu trong phạm vi một bundle giá trị và hai bậc đạo hàm theo đầu vào.
- Các benchmark chính của S1 dùng phương trình bậc một. Lợi ích cho bundle có $f''$ là hướng mở rộng đáng thử, chưa phải cải thiện trực tiếp đã chứng minh cho Figs. 6–10.

### 8.3 Công thức đủ để tái hiện kiểm tra reconstruction

Với $L=2R+1$, chọn shifts $s_l=\pi l/L$ và các frequency $\omega_r=2r$, $r=-R,\ldots,R$. Các weights:

$$
w_l^{(k)}=\frac{1}{L}\sum_{r=-R}^R
(i\omega_r)^k e^{-i\omega_r s_l},\qquad k=1,2.
$$

Khi đó $F^{(k)}(t)=\sum_l w_l^{(k)}F(t+s_l)$. Phần ảo của weights triệt tiêu theo đối xứng, tới sai số làm tròn. Có thể tạo weights bằng:

```python
L = 2 * R + 1
shifts = np.pi * np.arange(L) / L
omega = 2 * np.arange(-R, R + 1)
fourier = np.exp(-1j * omega[:, None] * shifts[None, :]) / L
w1 = ((1j * omega) @ fourier).real
w2 = ((-omega**2) @ fourier).real
```

Phải đánh giá các gate ở góc $2a_j(t+s_l)$, rồi áp chain rule ở mục 7. Công thức này là kiểm chứng hữu hạn Fourier của ghi chú, không phải một đóng góp mới được gán cho S2.

## 9. Những điểm trong S2 cần kiểm tra trước khi hiện thực

### 9.1 Pha và hệ số trong four-term rule

Eq. (10) và Eq. (20) của S2 đặt một hệ số $-i$ trước tổng các hiệu expectation. Với observable Hermitian, expectation và đạo hàm đều thực. Dạng này chỉ nhất quán nếu định nghĩa các hệ số phức được quy định phù hợp. Tuy nhiên, hệ Eq. (12) được in với RHS thực $1/4$ và $1/2$ và không làm rõ sự chuyển quy ước tương ứng.

Kiểm tra độc lập theo quy ước $U(a)=e^{-iaG}$: nếu dùng real weights $w_k$ trong

$$
C'(a)=\sum_k w_k[C(a+s_k)-C(a-s_k)],
$$

thì với mỗi gap dương $\omega$ cần có:

$$
2\sum_k w_k\sin(\omega s_k)=\omega.
$$

Với phổ $\{-1,0,1\}$, các gap là 1 và 2, nên:

$$
\sum_k w_k\sin(s_k)=\frac12,\qquad
\sum_k w_k\sin(2s_k)=1.
$$

Đây khác hệ số in trong Eq. (12), nếu dùng cùng ý nghĩa real weights.

**Phản kiểm tra cụ thể:** chọn $G=\mathrm{diag}(-1,0,1)$, trạng thái chồng chập đều của hai eigenstates có eigenvalues 0 và 1, cùng observable nối hai trạng thái đó. Khi ấy $C(a)=\cos a$.

Với shifts $\pi/4,\pi/2$ và $a=0.37$:

| Cách tính | Kết quả |
|---|---:|
| Đạo hàm thật $-\sin(0.37)$ | $-0.361615431964962$ |
| Real weights theo điều kiện gap ở trên | $-0.361615431964962$ |
| Dùng RHS Eq. (12) làm real weights, bỏ $-i$ | $-0.180807715982481$ |
| Dùng cả Eq. (12) và $-i$ của Eq. (10) nguyên như in | $+0.180807715982481i$ |

Điều này chỉ ra **sự không nhất quán khi hiện thực nguyên văn theo cùng quy ước**, không bác bỏ đẳng thức nội suy Lagrange Eq. (4). Cần tác giả làm rõ hoặc chuẩn hóa lại coefficients trước khi dùng.

### 9.2 Algorithm 1: hệ thuần nhất và normalization

Algorithm 1 mô tả một ma trận vuông $T$ rồi giải $T R=0$, sau đó chuẩn hóa coefficients. Nếu $T$ full rank thì chỉ có $R=0$; normalization khi đó không xác định. Cần chỉ ra điều kiện chọn shifts, rank, nullspace không tầm thường và điều kiện normalization khác zero.

Một cách kiểm tra recipe rõ ràng là dựng $A_{rk}=2\sin(\omega_r s_k)$ và xác nhận $Aw=\omega$ trên **toàn bộ gap cần thiết**, dùng rank/condition-number checks. Đây là đề xuất kiểm chứng của ghi chú, không phải khẳng định hai thủ tục được viết trong S2 đã tương đương hoàn chỉnh.

### 9.3 Công thức đếm term trong Table 1

Table 1 in $m=\lfloor 2^n/4\rfloor-1$ cho phổ đối xứng, $n\ge4$. Đồng thời phần mô tả gắn $m$ với số phần tử khác zero trong một tam giác của $\Delta M$, có đường chéo bằng zero.

Nếu đọc đúng theo cách đếm đó, số vị trí không vượt $n(n-1)/2$; tại $n=7$, biểu thức in cho $m=31$ nhưng một tam giác ngoài đường chéo chỉ có 21 vị trí. Vì vậy cần kiểm tra lại định nghĩa hoặc lỗi trình bày của công thức này. Không nên dùng Table 1 để ước lượng measurement budget khi chưa làm rõ. Có thể có lỗi ký hiệu, nhưng ghi chú này không tự thay công thức thay cho tác giả.

### 9.4 Thiếu benchmark trong bản PDF được cung cấp

Sec. 5 và Sec. 6 chưa trình bày số liệu, setup hay kết quả so sánh. Chưa có bằng chứng trong bản PDF này về:

- DQC residual/solution accuracy sau training.
- Wall-clock speedup.
- Equal-shot-budget comparisons hoặc noise robustness.
- Chi phí tiền xử lý coefficients khi số trị riêng tăng.

S2 có dẫn [repository tác giả](https://github.com/vutuanhai237/LagrangeGPSR). Đánh giá này không phải audit toàn bộ repository đó và không dùng kết quả chưa đọc trong repository để lấp các phần trống của PDF.

### 9.5 Lưu ý tương tự ở S1

S1 Eq. (9) in hệ số $1/4$. Với quy ước $R_P(a)=e^{-iaP/2}$ và expectation dịch góc thông thường, kiểm tra một qubit cho hệ số $1/2$. Khi benchmark phải chuẩn hóa baseline trước; sửa một hệ số in chưa nhất quán không nên được báo cáo là một lợi thế thuật toán của phương pháp mới.

## 10. Vì sao ít cấu hình không đồng nghĩa ít shots?

Với estimator $\widehat D=\sum_l c_l\widehat F_l$ và các phép ước lượng độc lập:

$$
\mathrm{Var}(\widehat D)=\sum_l c_l^2\frac{\sigma_l^2}{S_l},
$$

trong đó $S_l$ là số shots cho cấu hình thứ $l$, $\sigma_l^2$ là phương sai một shot. Với tổng shots cố định và bỏ qua các ràng buộc nguyên, phân bổ tối ưu có dạng $S_l\propto|c_l|\sigma_l$.

Do đó cần theo dõi độ lớn coefficients và conditioning, không chỉ số term. Lagrange basis có các mẫu số chênh lệch eigenvalues; phổ gần suy biến hoặc lựa chọn shifts xấu có thể làm việc dựng recipe nhạy về số học. Ngoài ra, $t'(x)$, $t''(x)$ của inverse-trigonometric encoding có thể khuếch đại uncertainty gần đầu mút.

Generalized analytic rules không tự giải quyết gate noise, shot noise, barren plateaus hoặc sai số mô hình hàm thử.

## 11. Hessian và Fubini–Study metric có giúp training không?

S2 Sec. 4 trình bày higher-order derivatives và Fubini–Study metric. Những đại lượng này có thể phục vụ Newton-like hoặc quantum-natural-gradient optimizers, nhưng đó là **thay đổi chiến lược tối ưu**, khác với giảm chi phí $df/dx$ trong residual.

Cần thêm phép đo overlap/metric, xử lý metric suy biến, regularization và bài toán giải hệ cập nhật. Ví dụ metric Eq. (27) có phần tử zero; không thể mặc định đảo metric trực tiếp. Bản PDF chưa có benchmark training cho DQC để kết luận cách này tốt hơn Adam.

## 12. Hướng thử nghiệm đề xuất cho project

### Giai đoạn A: xác nhận toán học, chưa thay backend mặc định

1. Chuẩn hóa các quy ước $G$, góc, shift và coefficients.
2. Kiểm tra two-term trên $R_y$ bằng hàm lượng giác có đạo hàm đúng biết trước.
3. Kiểm tra four-term trên generator có phổ $\{-1,0,1\}$ và đối chiếu điều kiện gap.
4. Với multiple-term, kiểm tra residual của hệ recipe, rank và condition number.
5. So sánh $f'$, $f''$ và mixed derivatives với phép tính độc lập.

### Giai đoạn B: chọn recipe theo workload

- Giữ gate-wise PSR cho các góc Pauli độc lập và các bài chỉ cần đạo hàm bậc một.
- Thử shared-parameter reconstruction cho bundle $f,f',f''$ của Cheb/ChebT.
- Chỉ dùng general-generator recipe khi có một khối gate thực sự cần lấy đạo hàm và việc biết phổ là khả thi.
- Khử trùng lặp và cache các recipe/baseline evaluations trước khi so sánh.

### Giai đoạn C: benchmark end-to-end công bằng

| Cần đo | Cách so sánh |
|---|---|
| Đúng công thức | Sai số derivative và mixed derivative so với reference độc lập |
| Tài nguyên | Unique circuit settings sau dedup, tổng shots, nhóm observable, thời gian tiền xử lý |
| Hiệu năng | Wall-clock derivative/training time trên cùng backend |
| Chất lượng solver | Residual và nghiệm trên lưới validation riêng |
| Ổn định | Nhiều seeds, shot budgets và noise levels |
| Đóng góp riêng của Lagrange | So cả gate-wise PSR và generalized Fourier PSR, không chỉ so full reconstruction |

Với source hiện tại, phần nghiên cứu recipe nằm gần `src/dqc/feature_maps.py`; phần truyền đạo hàm/Jacobian nằm ở `src/dqc/model.py`. Không chỉ thay một hệ số trong `angles()`: các công thức chain rule và gradient loss cần được kiểm tra nhất quán.

## 13. Cách viết nhận định trong literature review

> Cách tiếp cận nội suy Lagrange cung cấp một hướng xây dựng generalized parameter-shift rules từ phổ của generator. Đối với kiến trúc DQC sử dụng các cổng quay Pauli độc lập, trường hợp hai trị riêng quy về quy tắc hai-shift hiện có, nên chưa thể suy ra cải thiện trực tiếp cho đạo hàm bậc một. Tiềm năng đáng khảo sát nằm ở các cổng native có generator tổng quát và việc tái sử dụng phép đánh giá của những encoding phụ thuộc một biến chung để tính đạo hàm bậc cao. Tuy nhiên, bản tài liệu được xem xét còn cần làm rõ một số hệ số và quy trình xác định coefficients; hiệu quả trên DQC phải được kiểm chứng với baseline generalized Fourier rules, cùng ngân sách đo và đánh giá chất lượng nghiệm.

Đoạn trên là nhận định tổng hợp của ghi chú; không phải kết luận thực nghiệm do S1 hoặc S2 báo cáo.

## 14. Kết luận sử dụng

**Đáng dùng làm hướng nghiên cứu tiếp, chưa nên thay nguyên văn derivative backend hiện tại bằng các công thức trong PDF.**

Ưu tiên rõ nhất cho một prototype là tính đồng thời $f,f',f''$ theo biến mã hóa chung và so với baseline đã tối ưu. Nếu mục tiêu chỉ là chạy lại các benchmark bậc một của S1 với các Pauli rotation hiện tại, lợi ích từ Lagrange generalized PSR chưa được xác lập.
