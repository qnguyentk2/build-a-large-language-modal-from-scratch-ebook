# build-a-large-language-modal-from-scratch-ebook

<img width="1625" height="966" alt="image" src="https://github.com/user-attachments/assets/0fad6a8e-1d15-4106-a28a-fa03f687f075" />

## Chapter 1 - Understanding large language models

LLM không suy nghĩ → nó mô phỏng kết quả của tư duy thông qua thống kê ngôn ngữ.
LLM giúp máy tính tạo ra câu trả lời bằng cách dự đoán và sinh các từ (token) sao cho phù hợp với ngữ cảnh ngôn ngữ mà con người thường dùng; về bản chất, nó không suy nghĩ hay có ý thức như con người khi nói ra câu đó.

Nhờ những tiến bộ trong học sâu (deep learning), LLM được huấn luyện trên lượng dữ liệu văn bản khổng lồ. Việc huấn luyện quy mô lớn giúp mô hình nắm bắt ngữ cảnh và sắc thái ngôn ngữ tốt hơn, từ đó cải thiện mạnh hiệu suất trên nhiều bài toán NLP như dịch thuật, phân tích cảm xúc và trả lời câu hỏi.

**Không phải kiến trúc “thông minh hơn” → mà là kiến trúc + dữ liệu + scale tạo ra khả năng mới.**

```
More data + bigger models + longer training
→ better language representations
→ emergent behaviors (reasoning, summarization, QA)
```

> Emergent behaviors là những khả năng “mọc ra” ngoài dự kiến khi mô hình được scale đủ lớn (nhiều tham số, nhiều dữ liệu, huấn luyện lâu), chứ không phải do con người chủ động code vào.
>> Ví dụ đời thường (rất dễ hiểu)
>> 
>> 🐜 Một con kiến → không thông minh
>> 
>> 🐜🐜🐜 Cả đàn kiến → biết tìm đường, xây tổ, phân công
>> 
>> 👉 Không con kiến nào “biết” chiến lược,
>> 
>> 👉 nhưng hành vi tập thể tự xuất hiện → emergent behavior.

LLM sử dụng một kiến trúc gọi là Transformer, cho phép mô hình tập trung sự chú ý một cách chọn lọc vào các phần khác nhau của dữ liệu đầu vào khi đưa ra dự đoán. Nhờ đó, chúng đặc biệt hiệu quả trong việc xử lý những sắc thái và độ phức tạp của ngôn ngữ con người.
Kiến trúc Transformer giúp LLM ‘chú ý đúng chỗ’ trong văn bản đầu vào, nên mô hình có thể hiểu và xử lý tốt hơn các mối quan hệ và sắc thái tinh tế của ngôn ngữ.


### Keyword:
1. [Deep neural network models (DNN models):](https://chatgpt.com/g/g-p-696e03d1cfd481918a4ca9cdc44a493c-build-a-large-language-model-from-scratch/c/696e03d8-ba1c-8332-a092-3f3c2e82bdb3) 
Deep Neural Network là một hệ thống gồm nhiều tầng toán học nối tiếp nhau, học cách ánh xạ input → output bằng cách tự điều chỉnh trọng số thông qua dữ liệu, thay vì viết rule thủ công.

👉 Trọng số (weights) không “tự nhiên mà có” — nó được khởi tạo ngẫu nhiên, rồi được học dần từ dữ liệu.

2. [Selective attention] (https://chatgpt.com/g/g-p-696e03d1cfd481918a4ca9cdc44a493c-build-a-large-language-model-from-scratch/c/696e0f12-05f8-832b-a099-1a7f7ac94294)
   Selective attention trong kiến trúc Transformer là cơ chế cho phép mô hình chọn lọc những phần thông tin quan trọng nhất để tập trung, thay vì xử lý mọi thứ một cách đồng đều.


   **Khi đọc một chuỗi (câu, token, vector):**

    Transformer không “nhìn đều” tất cả token.

    Ở mỗi token, mô hình quyết định token nào đáng chú ý hơn (liên quan hơn) để tổng hợp thông tin.

    Việc “chọn lọc” này diễn ra tự động, thông qua trọng số attention được học trong quá trình train.


   **Ví dụ**
   “Con mèo nằm trên tấm thảm vì nó rất ấm.”

   Khi xử lý từ “nó”:

    Attention sẽ tập trung mạnh vào “tấm thảm”,

    Ít chú ý hơn tới “con”, “nằm”, “vì”, …

    → Transformer chọn lọc ngữ cảnh có ý nghĩa.
