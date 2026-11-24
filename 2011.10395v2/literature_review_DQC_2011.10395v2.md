# Ghi chú literature review: Solving nonlinear differential equations with differentiable quantum circuits

## 1. Thông tin thư mục và phạm vi ghi chú

| Trường | Thông tin |
|---|---|
| Tác giả | Oleksandr Kyriienko, Annie E. Paine, Vincent E. Elfving |
| Tên bài | *Solving nonlinear differential equations with differentiable quantum circuits* |
| Công bố | *Physical Review A*, **103**, 052416 (2021) |
| DOI | [10.1103/PhysRevA.103.052416](https://doi.org/10.1103/PhysRevA.103.052416) |
| Bản đang đọc | [arXiv:2011.10395v2](https://arxiv.org/abs/2011.10395v2), cập nhật ngày 18/05/2021 |
| Bản đầu | arXiv v1, ngày 20/11/2020 |
| Nhóm phương pháp | Variational quantum algorithm; hybrid quantum-classical computing; giải phương trình vi phân bằng tối thiểu hóa residual |
| Từ khóa | Differentiable quantum circuits (DQCs), quantum feature maps, Chebyshev encoding, collocation, boundary handling, nonlinear differential equations |

**Phạm vi:** ghi chú một paper, không phải khảo sát đầy đủ hay cập nhật các công trình xuất hiện sau paper. Số phương trình, hình và trang bên dưới tham chiếu bản PDF arXiv v2 [S1]. Các nhận xét phê bình và hướng nghiên cứu đề xuất được ghi riêng với kết quả tác giả báo cáo. Không dùng kết quả từ notebook minh họa làm kết quả của paper.

## 2. Tóm tắt cốt lõi để ghi nhớ

Paper biểu diễn nghiệm của phương trình vi phân bằng giá trị kỳ vọng của một quantum circuit có tham số. Biến đầu vào được mã hóa thành góc quay qua **quantum feature map**; mạch biến phân điều chỉnh dạng hàm; các mạch được dịch góc cho phép tính đạo hàm theo biến đầu vào. Sau đó, một bộ tối ưu cổ điển cập nhật tham số để giảm sai số phương trình và thỏa mãn điều kiện đầu/biên.

Đóng góp đáng chú ý là kết hợp **biểu diễn hàm, đạo hàm mạch, Chebyshev feature maps và xử lý điều kiện biên** trong một quy trình giải phương trình vi phân. Bằng chứng thực nghiệm là mô phỏng số trên các bài toán chọn lọc; paper chưa chứng minh lợi thế tính toán đầu-cuối so với các phương pháp cổ điển mạnh. [S1, Sec. II–V]

## 3. Bài toán, động lực và khoảng trống được nhắm đến

### 3.1 Bài toán nghiên cứu

Tìm một hoặc nhiều hàm chưa biết thỏa mãn phương trình dạng tổng quát:

$$
F\bigl(x,u(x),u'(x),u''(x),\ldots\bigr)=0,
$$

cùng các điều kiện đầu/biên đã cho. Có thể có nhiều biến độc lập và nhiều phương trình; tuy nhiên, các minh họa chính của paper tập trung vào bài toán một biến, hệ ODE và dòng chảy nozzle ở trạng thái dừng. Không nên suy từ khung tổng quát rằng paper đã kiểm chứng trên PDE nhiều chiều quy mô lớn. [S1, Sec. II, Eq. (3); Sec. IV]

### 3.2 Động lực

- Phương trình phi tuyến, nghiệm dao động và hệ nhiều biến có thể đòi hỏi biểu diễn hàm phong phú và chi phí tính toán lớn.
- Các quantum solver dựa trên amplitude encoding có thể gặp chi phí chuẩn bị trạng thái, giả thiết oracle và khó khăn khi đọc toàn bộ nghiệm.
- Quantum circuit có tham số cung cấp một cách xây dựng hàm xấp xỉ và huấn luyện bằng vòng lặp quantum-classical.
- Tính đạo hàm trực tiếp từ cấu trúc mạch có thể tránh sai số cắt cụt do dùng sai phân hữu hạn để lấy đạo hàm của hàm thử. [S1, Sec. I–II]

### 3.3 Câu hỏi nghiên cứu có thể dùng khi viết review

> Có thể huấn luyện một hàm do quantum circuit biểu diễn, sử dụng đạo hàm mạch và các ràng buộc của phương trình, để xấp xỉ nghiệm của các phương trình vi phân phi tuyến và hệ phương trình hay không?

Đây là cách diễn đạt tổng hợp của người viết ghi chú, không phải câu trích nguyên văn. Paper chủ yếu cung cấp bằng chứng về tính khả thi của phương pháp và ảnh hưởng của các lựa chọn thiết kế.

## 4. Các khái niệm và công thức cần nắm

### 4.1 Phân biệt biến đầu vào và tham số học

| Ký hiệu | Ý nghĩa |
|---|---|
| $x$ | Biến độc lập: có thể là thời gian, tọa độ không gian hoặc tọa độ đã chuẩn hóa |
| $u(x)$ | Nghiệm thật cần tìm |
| $f_\theta(x)$ | Hàm thử do mạch biểu diễn sau khi xử lý điều kiện biên, nếu có |
| $\theta$ | Các tham số được cập nhật trong quá trình tối ưu |
| $\varphi_j(x)$ | Hàm ánh xạ từ đầu vào sang góc của cổng mã hóa thứ $j$ |
| $C$ | Observable dùng để đọc đầu ra của mạch |
| $\mathbb X=\{x_i\}_{i=1}^M$ | Các điểm collocation để đánh giá residual khi huấn luyện |

Ví dụ, trong nozzle, $x$ là vị trí dọc ống; $\rho(x)$, $T(x)$ và $V(x)$ là các hàm cần tìm. $x$ không phải giá trị nghiệm và không phải tham số được học.

### 4.2 Biểu diễn hàm bằng expectation value

Chuẩn bị trạng thái:

$$
|\psi_\theta(x)\rangle
=U_\theta U_\varphi(x)|0\cdots0\rangle.
$$

Đọc một hàm vô hướng:

$$
q_\theta(x)
=\langle\psi_\theta(x)|C|\psi_\theta(x)\rangle.
$$

Ở đây ký hiệu $q$ được dùng để phân biệt đầu ra thô với $f$ sau khi thêm offset hoặc áp điều kiện biên. Paper dùng expectation value làm biểu diễn hàm ở Eq. (1). Readout cũng có thể là tổng có trọng số của nhiều observable; các trọng số cổ điển có thể được tối ưu. [S1, Fig. 2; Sec. III C, Eq. (18)]

Nghiệm không được đọc bằng cách lấy từng amplitude của một trạng thái chứa toàn bộ lưới nghiệm. Với mỗi giá trị $x$ được hỏi, mạch cho một expectation value. Điều này vẫn đòi hỏi số lần đo phù hợp để đạt độ chính xác mong muốn trên phần cứng.

### 4.3 Tại sao cần quantum feature map?

Feature map đưa sự phụ thuộc vào $x$ vào mạch và quyết định họ đặc trưng dùng để biểu diễn nghiệm. Nếu mạch không có thành phần phụ thuộc vào $x$, đầu ra sẽ là một hằng số theo $x$.

Ví dụ một qubit: $R_y(x)|0\rangle$ có expectation $\langle Z\rangle=\cos x$. Các ánh xạ góc khác nhau tạo ra các kiểu phụ thuộc hàm khác nhau. Mạch biến phân trộn những đặc trưng đó để tìm dạng nghiệm thích hợp.

| Feature map | Cổng mã hóa tiêu biểu | Nguồn |
|---|---|---|
| Product, Prod | $R_{y,j}(\arcsin x)$ | Eq. (6) |
| Chebyshev, Cheb | $R_{y,j}(2\arccos x)$ | Eq. (14) |
| Chebyshev tower, ChebT | $R_{y,j}(2j\arccos x)$, với $j=1,\ldots,N$ | Eq. (15) |
| Evolution-enhanced | Bổ sung evolution unitary sau mã hóa; ví dụ $e^{-iH\tau}$ | Sec. III A; Fig. 3(c); Fig. 7 |

Chebyshev encoding dựa trên:

$$
T_n(x)=\cos(n\arccos x).
$$

Với $R_y(2n\arccos x)|0\rangle$, amplitude của $|0\rangle$ bằng $T_n(x)$, còn expectation của $Z$ bằng $T_{2n}(x)$. Cần phân biệt **amplitude** và **expectation**; đầu ra của mạch đã huấn luyện không đơn thuần là một dãy hệ số Chebyshev tự do. [S1, Sec. III A, Eqs. (11)–(15)]

### 4.4 Đạo hàm bằng parameter-shift và chain rule

Với cổng $R_P(\alpha)=e^{-i\alpha P/2}$, $P^2=I$, và $\alpha$ xuất hiện trong một cổng, expectation thỏa mãn:

$$
\frac{\partial q}{\partial\alpha}
=\frac{q(\alpha+\pi/2)-q(\alpha-\pi/2)}{2}.
$$

Nếu $x$ đi vào nhiều cổng thông qua các góc $\varphi_j(x)$:

$$
\frac{dq_\theta}{dx}
=\sum_j\frac{\varphi_j'(x)}{2}
\left[q_{\theta,j,+}(x)-q_{\theta,j,-}(x)\right].
$$

Mỗi hạng chỉ dịch góc của **một cổng mã hóa**; không dịch trực tiếp đầu vào $x$ và không dịch đồng thời mọi cổng. Biểu thức trên là dạng viết minh bạch chain rule với quy ước cổng vừa nêu. [S1, Sec. II–III A; Fig. 3; xem lưu ý công thức tại mục 10]

Cần phân biệt hai loại đạo hàm:

- $\partial f/\partial x$: xuất hiện trong phương trình vi phân.
- $\partial\mathcal L/\partial\theta$: dùng để cập nhật tham số của mạch.

Đạo hàm giải tích ở đây là đạo hàm của **hàm thử hiện tại**. Nó không có nghĩa nghiệm của phương trình đã được tìm chính xác, và cũng không loại bỏ sai số lấy mẫu khi đo trên phần cứng.

### 4.5 Loss và vai trò của tính phi tuyến

Với một phương trình, viết residual:

$$
r_\theta(x)=F\bigl(x,f_\theta(x),f_\theta'(x),\ldots\bigr).
$$

Một lựa chọn loss là:

$$
\mathcal L_{\mathrm{diff}}
=\frac{1}{M}\sum_{i=1}^M r_\theta(x_i)^2.
$$

Loss tổng có thể bao gồm:

$$
\mathcal L
=\mathcal L_{\mathrm{diff}}
+\mathcal L_{\mathrm{boundary}}
+\mathcal L_{\mathrm{reg}}.
$$

Các tích và hàm phi tuyến của đầu ra, ví dụ $f^2$, được tính trong bước hậu xử lý cổ điển. Không cần biến phép tiến hóa lượng tử thành một phép biến đổi phi tuyến. Các giá trị nghiệm ở điểm regularization có thể được dùng để định hướng quá trình học; vì vậy không nên mô tả mọi thí nghiệm là hoàn toàn không sử dụng dữ liệu nghiệm. [S1, Sec. III D, F]

### 4.6 Điều kiện đầu/biên

| Cách xử lý | Biểu diễn tiêu biểu cho $f(x_0)=u_0$ | Ý nghĩa |
|---|---|---|
| Pinned | $f=q_\theta$, thêm penalty tại $x_0$ | Điều kiện biên và phương trình cùng được tối ưu trong loss |
| Floating | $f(x)=u_0+q_\theta(x)-q_\theta(x_0)$ | Ép điều kiện đầu bằng cấu trúc hàm trong mô hình không nhiễu |
| Optimized | $f(x)=c+q_\theta(x)$, học thêm $c$ | Có offset cổ điển nhưng vẫn cần ràng buộc biên |

Floating boundary giảm cạnh tranh giữa loss phương trình và loss điều kiện đầu. Tuy nhiên, offset phải được cập nhật khi $\theta$ thay đổi; gradient theo $\theta$ phải bao gồm cả hạng $q_\theta(x_0)$. Công thức một điểm này không tự động ép được mọi dạng điều kiện biên nhiều điểm. [S1, Sec. III E, Eqs. (23)–(27)]

### 4.7 Nhiều hàm và quy trình huấn luyện

Một hệ phương trình có thể dùng chung register với nhiều observable hoặc dùng các mạch/register riêng cho từng hàm. Trong trường hợp thứ hai, các tham số vẫn được tối ưu cùng nhau qua loss chứa các phương trình ghép cặp. [S1, Sec. III G]

Quy trình chính: chọn feature map, ansatz, readout và boundary handler → đánh giá hàm cùng đạo hàm tại các điểm collocation → tính loss cổ điển → lấy gradient và cập nhật tham số → lặp → đánh giá nghiệm tại các điểm quan tâm. Paper dùng Adam trong các thí nghiệm. [S1, Fig. 1; Sec. III H]

## 5. Đóng góp của paper

| Đóng góp | Bằng chứng và vị trí trong paper | Cách diễn đạt phù hợp trong review |
|---|---|---|
| Khung DQC giải phương trình vi phân | Expectation-valued functions, derivative circuits, residual optimization | Đề xuất một quy trình giải phương trình bằng tối ưu biến phân quantum-classical |
| Chebyshev feature maps | Sec. III A, Eqs. (11)–(15); Fig. 6 | Thiết kế và khảo sát mã hóa có liên hệ với cơ sở Chebyshev |
| Kết hợp xử lý điều kiện biên, regularization, nhiều đầu ra | Sec. III E–G; Fig. 8 | Phân tích các lựa chọn thiết kế ảnh hưởng đến huấn luyện |
| Minh họa trên bài toán ngày càng phức tạp | Sec. IV, Figs. 6–10 | Cung cấp bằng chứng mô phỏng về tính khả thi trên các ví dụ được chọn |

Không nên gán tính mới cho từng thành phần riêng lẻ như Adam hay parameter-shift. Điểm đáng chú ý nằm ở cách tổ chức và kết hợp chúng cho bài toán phương trình vi phân. Cũng không nên gọi paper là công trình đầu tiên theo một nghĩa rộng nếu chưa kiểm tra đầy đủ tài liệu trước đó.

## 6. Thiết kế thí nghiệm và các kết quả cần ghi lại

### 6.1 Môi trường và thước đo

Tác giả sử dụng Yao.jl trong Julia và full-state quantum simulation ở thiết lập không nhiễu. Các minh họa chính dùng register sáu qubit. Đây là kết quả mô phỏng, không phải đo đạc trên quantum hardware. [S1, Sec. IV, phần thí nghiệm Eq. (30)]

Các chỉ số cần phân biệt:

- $L_F$: full loss, gồm các thành phần được dùng trong huấn luyện.
- $L_Q$: MSE giữa nghiệm DQC và nghiệm tham chiếu; đây là thước đo chất lượng nghiệm, không mặc nhiên là loss được tối ưu.
- $L_D$: paper mô tả là phần loss loại regularization. Khi dùng floating boundary và không có penalty biên riêng, nó tương ứng phần residual phương trình. Khi so sánh các thí nghiệm, cần kiểm tra các thành phần loss cụ thể. [S1, Sec. IV, phần định nghĩa metrics gần Fig. 6]

### 6.2 Bảng tổng hợp thí nghiệm

| Thí nghiệm | Cấu hình/chủ đích chính | Kết quả tác giả báo cáo | Nguồn |
|---|---|---|---|
| Hàm dao động tắt dần | Eq. (30); $\lambda=8,20$, $\kappa=0.1$, $u_0=1$; 20 điểm trong miền đến $x=0.9$; 6 qubit, depth 5, floating boundary; so sánh Prod/Cheb/ChebT | Với $\lambda=8$, hai Chebyshev map tiến gần nghiệm tốt hơn Prod; với $\lambda=20$, ChebT tiếp tục biểu diễn được nghiệm tốt hơn hai lựa chọn còn lại trong thiết lập khảo sát | Fig. 6(a,b,d,e), pp. 11–13 |
| Độ sâu ansatz | Cùng họ bài toán, khảo sát depth 3, 6, 12, 24 | Mạch sâu hơn cải thiện kết quả trong ví dụ này, đồng thời tăng số tham số và chi phí mỗi vòng lặp | Fig. 6(c,f) |
| ODE phi tuyến dao động nhanh | Eq. (32); $u_0=0.75$; 100 điểm; ChebT rồi evolution-enhanced; evolution time $\tau=2$; hai giai đoạn, mỗi giai đoạn 200 Adam iterations | Giai đoạn bổ sung evolution-enhanced map cải thiện đường nghiệm và metrics so với kết quả giai đoạn đầu | Fig. 7, p. 13 |
| Hệ hai phương trình ghép cặp | Eqs. (33)–(34); $\lambda_1=5$, $\lambda_2=3$; $(u_{1,0},u_{2,0})=(0.5,0)$; ChebT, depth 5; so sánh ba boundary handler | Floating boundary cho $L_Q$ xấp xỉ $10^{-5}$ trong kết quả được báo cáo và tiến gần nghiệm nhanh hơn hai lựa chọn còn lại | Fig. 8, pp. 14–15 |
| Nozzle flow | Ba hàm $\rho,T,V$; Chebyshev map, 6 qubit, depth 6; floating boundary; huấn luyện hai giai đoạn có regularization | DQC tạo profile gần nghiệm tham chiếu và thể hiện quá trình tăng vận tốc, giảm mật độ/nhiệt độ theo nhánh nghiệm mong muốn | Figs. 9–10, pp. 15–17 |

**Cách sử dụng kết quả:** con số $10^{-5}$ là giá trị gần đúng được tác giả nêu cho thí nghiệm cụ thể, không phải bảo đảm sai số cho toàn phương pháp. Không có một bảng benchmark thống nhất chứng minh tốc độ hoặc độ chính xác vượt trội so với mọi solver cổ điển.

### 6.3 Bài toán phi tuyến Eq. (32)

$$
u'-4u+6u^2-\sin(50x)-u\cos(25x)+\frac{1}{2}=0,
\qquad u(0)=0.75.
$$

Đây là ví dụ phù hợp để dẫn khi bàn về khả năng biểu diễn nghiệm có thành phần dao động nhanh và phi tuyến. Evolution-enhanced map dùng một Ising-type Hamiltonian trong thí nghiệm. Tuy nhiên, mô hình tăng cường được huấn luyện thêm sau mô hình ban đầu; khi đánh giá quan hệ nhân quả, cần chú ý rằng cả kiến trúc lẫn tổng ngân sách huấn luyện đều thay đổi. [S1, Sec. IV; nhận xét cuối là phân tích phê bình]

### 6.4 Chi tiết nozzle cần giữ khi trích dẫn

Đây là bài toán dòng chảy **nén được, không nhớt, quasi-1D**, với các phương trình bảo toàn đã được rút gọn và xét trạng thái dừng. Không phải mô phỏng Navier–Stokes ba chiều tổng quát hay dòng rối.

Hình học và điều kiện đầu được nêu:

$$
A(x)=1+4.95(2x-1)^2,\qquad \gamma=1.4,
$$

$$
\rho(0)=1,\qquad T(0)=1,\qquad V(0)=0.1.
$$

Dạng phương trình đã giải tường minh cho đạo hàm có mẫu số $T-V^2$, gây khó khăn gần điều kiện sonic. Thiết kế huấn luyện của paper gồm:

1. Giai đoạn 1: 20 điểm upstream thuộc vùng $(0,0.4)$; 200 Adam iterations; learning rate 0.01.
2. Giai đoạn 2: 40 điểm chia giữa $(0,0.4)$ và $(0.6,0.9)$; 600 iterations; learning rate 0.005; khởi tạo từ giai đoạn 1.
3. Dùng 20 điểm regularization upstream từ nghiệm giai đoạn 1 và 5 điểm downstream để định hướng nhánh supersonic; giảm regularization quanh iteration 150 của giai đoạn 2.
4. Không lấy collocation points trong khoảng $(0.4,0.6)$ quanh throat ở giai đoạn 2. Đường nghiệm trong khoảng này được sinh bởi biểu diễn hàm toàn cục, không phải do residual được ép trực tiếp tại đó. [S1, pp. 16–17; Fig. 10]

Khi viết review, nên giữ các chi tiết về vùng bị loại và dữ liệu định hướng. Việc một dạng explicit ODE gặp kỳ dị không đủ để kết luận toàn bộ phương pháp CFD cổ điển đều thất bại.

## 7. Vị trí của paper trong literature

Các đối chiếu dưới đây là **so sánh cơ chế**, không phải bảng xếp hạng hiệu năng thực nghiệm.

| Nhóm phương pháp | Điểm đối chiếu với DQC |
|---|---|
| Sai phân hữu hạn / phương pháp giải số theo lưới | DQC huấn luyện một hàm toàn cục; đạo hàm hàm thử lấy từ mạch. Tuy nhiên vẫn dùng điểm collocation để tối ưu residual |
| Classical spectral methods | Cùng ý tưởng xấp xỉ nghiệm qua các đặc trưng/cơ sở toàn cục; DQC dùng feature map và ansatz để tham số hóa họ hàm |
| Physics-informed neural networks | Cùng nguyên tắc dùng hàm có tham số và residual phương trình; DQC thay bộ xấp xỉ cổ điển bằng quantum circuit |
| HHL và các solver dựa trên amplitude encoding | Khác cách mã hóa/đọc nghiệm và các giả thiết tính toán; không thể suy ra ưu thế thời gian chỉ từ việc đổi cách biểu diễn |
| Các variational quantum methods cho bài toán phi tuyến | Cần so sánh riêng cách mã hóa nghiệm, xử lý phi tuyến, lấy đạo hàm, điều kiện biên và chi phí đo |

Từ khóa phân loại thích hợp: **quantum variational differential-equation solver**, **expectation-value function representation**, **analytic circuit differentiation**, **residual-based learning**. Có thể mô tả là có quan hệ với physics-informed learning, nhưng không nên đồng nhất nó với neural operator học ánh xạ từ cả một họ phương trình sang nghiệm. [S1, Sec. I–III; phần phân loại là tổng hợp của người viết]

## 8. Đánh giá phê bình: điểm mạnh, giới hạn và độ mạnh bằng chứng

### 8.1 Điểm mạnh

- Liên kết rõ biểu diễn hàm, tính đạo hàm và residual optimization trong một khung thống nhất.
- Chebyshev encoding có động lực từ xấp xỉ hàm/spectral representation; có thí nghiệm thay đổi feature map và độ sâu mạch.
- Xử lý điều kiện biên là một phần thiết kế được khảo sát, thay vì chỉ thêm penalty cố định.
- Minh họa cả ODE đơn, hệ ghép cặp và ứng dụng vật lý rút gọn. [S1, Sec. III–IV]

### 8.2 Những gì chưa được chứng minh

| Vấn đề | Giới hạn của bằng chứng | Hệ quả khi viết literature review |
|---|---|---|
| Quantum advantage | Chưa chứng minh speedup đầu-cuối so với baseline cổ điển mạnh | Dùng “potential” hoặc “hướng tiếp cận”; tránh “đã giải nhanh hơn” |
| Noise robustness | Các minh họa không nhiễu; Sec. V nêu chưa nghiên cứu tác động noise lên hiệu năng thuật toán | Đạo hàm giải tích không tương đương miễn nhiễm shot noise/gate noise |
| Scalability | Các minh họa dùng register nhỏ; chưa xác nhận hiệu quả trên PDE nhiều chiều lớn | Không suy diễn từ kích thước Hilbert space sang thời gian giải |
| Trainability | Loss phi lồi, độ sâu, ansatz và feature map ảnh hưởng tối ưu | Chưa có bảo đảm hội tụ toàn cục cho các bài toán tổng quát |
| Độ chính xác ngoài lưới | Hàm có thể được đánh giá ở điểm mới nhưng residual được huấn luyện tại hữu hạn điểm | Khả năng truy vấn liên tục không phải chứng nhận sai số toàn miền |
| Chi phí thực thi | Nhiều điểm, mạch dịch góc và phép đo cần được tính trong ngân sách | Một iteration không phải đơn vị đủ để so chi phí quantum-classical |
| Nozzle | Có regularization, định hướng nhánh và bỏ vùng throat khỏi collocation | Không mô tả là giải tự động toàn miền chỉ từ phương trình và inlet |

Một phân tích tài nguyên sơ bộ, **không phải định lý complexity của paper**: với $M$ điểm và $G$ cổng mã hóa phụ thuộc đầu vào, tính hàm cùng đạo hàm bậc một theo cách dịch từng cổng có thể cần cỡ $M(1+2G)$ cấu hình mạch cho một readout setting, trước tối ưu tái sử dụng. Gradient theo tham số, điều kiện biên, các nhóm observable và số shots làm tăng ngân sách. Phân tích lợi thế cần tính toàn bộ các khoản này.

### 8.3 Mức kết luận hợp lý

**Được hỗ trợ:** phương pháp có thể tạo nghiệm xấp xỉ hữu ích cho các ví dụ mô phỏng đã chọn; feature map, boundary handler và lịch huấn luyện ảnh hưởng kết quả.

**Chưa được hỗ trợ:** mọi hệ phi tuyến đều được giải hiệu quả; quantum speedup đã được thực nghiệm chứng minh; noise không ảnh hưởng; DQC đã thay thế solver CFD tổng quát.

## 9. Research gaps và hướng nghiên cứu có thể rút ra

Đây là các hướng đề xuất từ việc đọc phê bình, không phải kết quả đã hoàn thành trong paper và cũng không khẳng định chúng còn mới so với toàn bộ literature hiện nay.

1. **Benchmark công bằng:** so với solver ODE/PDE cổ điển phù hợp, spectral methods và PINNs; thống nhất tolerance, miền kiểm tra, thời gian và ngân sách tối ưu.
2. **Chi phí đo và noise:** khảo sát shots, gate noise, qubit count, circuit depth và error mitigation trên cùng một mục tiêu sai số nghiệm.
3. **Tách expressivity khỏi optimization budget:** so sánh feature maps với nhiều random seeds, cùng số cập nhật hoặc cùng ngân sách thời gian/measurement.
4. **Sampling và chứng nhận sai số:** adaptive collocation; kiểm tra residual trên lưới độc lập; phân tích sai số trong vùng chưa được huấn luyện.
5. **Ràng buộc vật lý:** positivity, bảo toàn khối lượng/năng lượng và nhiều điều kiện biên; đánh giá xem ràng buộc bằng cấu trúc hàm có ổn định hơn penalty hay không.
6. **Bài toán khó hơn:** stiffness, nghiệm có vùng biến thiên nhanh, nhiều chiều hoặc nhiều nhánh; nêu rõ giả thiết và điều kiện chọn nhánh.
7. **Nozzle:** lượng hóa ảnh hưởng của vùng throat bị bỏ, dữ liệu downstream và lịch regularization; kiểm tra tính nhất quán với các phương trình bảo toàn trên toàn miền.

## 10. Các lưu ý toán học khi đọc hoặc tái hiện

Các mục sau là kiểm tra công thức theo quy ước đã nêu; không thay thế cho một erratum chính thức.

- **Eq. (31):** với phương trình thuần nhất $u'+\lambda[\kappa+\tan(\lambda x)]u=0$, nghiệm thỏa $u(0)=u_0$ là $u(x)=u_0e^{-\lambda\kappa x}\cos(\lambda x)$ trên các khoảng hệ số xác định. Hằng số tích phân là nhân tử, không phải hằng số cộng như biểu thức in trong Eq. (31).
- **Các cực của $\tan(\lambda x)$:** nghiệm dạng đóng có thể có phần kéo dài trơn qua điểm mà hệ số ODE không xác định. Khi tái hiện cần làm rõ xử lý những điểm này và việc có thay đổi dạng/weight của residual hay không.
- **Hệ số parameter-shift:** Eq. (9) in hệ số $1/4$. Với $R_P(\alpha)=e^{-i\alpha P/2}$ và expectation dịch góc thông thường, kiểm tra một qubit $\langle Z\rangle=\cos\alpha$ cho hệ số $1/2$. Cần đối chiếu quy ước trước khi hiện thực.
- **Đầu mút Chebyshev:** đạo hàm góc $\arccos x$ kỳ dị ở $x=\pm1$, nhưng đạo hàm của đa thức $T_n$ không tự nó phân kỳ; ví dụ $T_n'(1)=n^2$. Tránh diễn giải câu trong paper thành “mọi đạo hàm Chebyshev đều phân kỳ”.
- **Chuẩn hóa miền:** khi dùng tọa độ mã hóa $z=z(x)$, phải nhân thêm $dz/dx$ khi chuyển đạo hàm về tọa độ vật lý.

Các lưu ý này liên quan trực tiếp đến tái lập; trong phần tổng quan chính chỉ cần nêu ngắn gọn nếu chúng ảnh hưởng nhận định hoặc kết quả so sánh.

## 11. Đoạn tổng hợp có thể chỉnh sửa để đưa vào literature review

> Kyriienko, Paine và Elfving (2021) đề xuất một phương pháp quantum-classical để giải phương trình vi phân bằng differentiable quantum circuits. Nghiệm được biểu diễn qua expectation values của mạch có tham số, trong khi quantum feature maps đưa biến độc lập vào các góc quay. Đạo hàm của hàm thử được xây dựng từ các mạch dịch góc và kết hợp trong loss dựa trên residual phương trình. Công trình giới thiệu Chebyshev feature maps, khảo sát các cách xử lý điều kiện biên và minh họa trên ODE, hệ phương trình ghép cặp cùng bài toán nozzle quasi-1D ở trạng thái dừng. Kết quả mô phỏng cho thấy chất lượng nghiệm phụ thuộc vào khả năng biểu diễn của feature map, cấu trúc mạch và chiến lược huấn luyện. Tuy vậy, các thí nghiệm không nhiễu và chưa xác lập lợi thế tính toán đầu-cuối so với các solver cổ điển mạnh; ví dụ nozzle còn sử dụng regularization và lựa chọn miền collocation để hỗ trợ tìm nhánh nghiệm mong muốn. [S1]

Đoạn trên là diễn giải tổng hợp, không phải trích dẫn nguyên văn. Khi dùng trong luận văn/bài báo, thay `[S1]` bằng citation key theo hệ quản lý tài liệu của bạn.

## 12. Một dòng cho literature review matrix

| Citation | Mục tiêu | Phương pháp | Bằng chứng | Kết quả chính | Hạn chế | Liên hệ nghiên cứu |
|---|---|---|---|---|---|---|
| Kyriienko et al., 2021 | Giải DE phi tuyến/hệ DE bằng variational quantum circuits | Expectation-valued trial function; Chebyshev maps; parameter-shift; residual loss; boundary handling | Noiseless statevector simulation; ODE, coupled ODE, steady quasi-1D nozzle | Các thiết kế feature map và boundary handler thích hợp cải thiện nghiệm trong ví dụ khảo sát | Chưa có end-to-end speedup; noise/scaling chưa được xác nhận; nozzle có guidance và vùng collocation bị loại | Nền tảng phương pháp để đối chiếu encoding, gradients, ràng buộc biên và chi phí huấn luyện |

## 13. Nguồn, vị trí tra cứu và trích dẫn

### Nguồn chính

- **[S1]** Kyriienko, O., Paine, A. E., & Elfving, V. E. (2021). *Solving nonlinear differential equations with differentiable quantum circuits*. [arXiv:2011.10395v2](https://arxiv.org/abs/2011.10395v2). [PDF v2](https://arxiv.org/pdf/2011.10395v2).
- **[S2]** Bản công bố trên *Physical Review A*, 103, 052416: [DOI](https://doi.org/10.1103/PhysRevA.103.052416). Thông tin thư mục được đối chiếu với hồ sơ arXiv; số hình/phương trình trong ghi chú dựa vào S1.

| Nội dung cần tra lại | Vị trí trong S1 |
|---|---|
| Động lực và liên hệ với công trình trước | Sec. I |
| Định nghĩa hàm và ý tưởng solver | Sec. II, Eqs. (1)–(4), Figs. 1–2 |
| Feature maps và đạo hàm | Sec. III A, Eqs. (5)–(15), Figs. 3–4 |
| Ansatz và readout | Sec. III B–C, Fig. 5, Eqs. (16)–(18) |
| Loss, boundary, regularization, nhiều hàm | Sec. III D–G, Eqs. (19)–(29) |
| Scalar ODE và hệ ghép cặp | Sec. IV, Eqs. (30)–(36), Figs. 6–8 |
| Nozzle và lịch huấn luyện | Sec. IV, Eqs. (40)–(50), Figs. 9–10, pp. 15–17 |
| Giới hạn, noise, khả năng quantum advantage | Sec. V |

### Những nhánh tài liệu nên đọc tiếp

Các mục dưới là đầu mối từ danh mục tài liệu của S1, chưa phải một review độc lập về từng công trình:

- Quantum circuit learning và quantum feature spaces: refs. [73], [153]–[155] trong S1.
- Variational quantum methods cho bài toán phi tuyến: Lubasch et al. (2020), ref. [141].
- Quantum computing cho dòng chảy Navier–Stokes: Gaitan (2020), ref. [142].
- Physics-informed neural networks: Raissi, Perdikaris và Karniadakis (2019), ref. [145].
- Trainability/barren plateaus: McClean et al. (2018), ref. [172], cùng các hướng giảm thiểu được bàn tại Sec. V.

### BibTeX

```bibtex
@article{Kyriienko2021DQC,
  author        = {Kyriienko, Oleksandr and Paine, Annie E. and Elfving, Vincent E.},
  title         = {Solving nonlinear differential equations with differentiable quantum circuits},
  journal       = {Physical Review A},
  volume        = {103},
  pages         = {052416},
  year          = {2021},
  doi           = {10.1103/PhysRevA.103.052416},
  eprint        = {2011.10395},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## 14. Checklist trước khi sử dụng trong bài viết

- [ ] Đã chọn chính xác năm công bố 2021, đồng thời phân biệt bản arXiv đầu năm 2020 nếu cần lịch sử phát triển.
- [ ] Đã diễn đạt rõ nghiệm là expectation-valued function và phân biệt $x$ với $\theta$.
- [ ] Đã phân biệt kết quả mô phỏng không nhiễu với thực nghiệm phần cứng.
- [ ] Đã phân biệt loss huấn luyện với sai số so với nghiệm tham chiếu.
- [ ] Không biến tiềm năng biểu diễn thành tuyên bố quantum speedup đã được chứng minh.
- [ ] Nếu dẫn nozzle, đã nêu mô hình quasi-1D, trạng thái dừng, regularization và vùng throat không có collocation points.
- [ ] Nếu dẫn số liệu, đã ghi đúng bài toán, cấu hình, metric và hình nguồn.
- [ ] Nếu nêu research gap, đã kiểm tra các công trình sau 2021 trước khi khẳng định tính mới.
