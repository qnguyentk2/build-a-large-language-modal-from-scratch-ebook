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


### Vậy LLM là gì? 
**LLM (mô hình ngôn ngữ lớn)** là một mạng nơ-ron được thiết kế để hiểu, tạo sinh và phản hồi văn bản giống con người. Các mô hình này là những mạng nơ-ron sâu, được huấn luyện trên khối lượng dữ liệu văn bản khổng lồ — đôi khi bao phủ một phần rất lớn của toàn bộ văn bản công khai hiện có trên internet.

Chữ ‘lớn’ trong ‘mô hình ngôn ngữ lớn’ vừa đề cập đến kích thước của mô hình xét theo số lượng tham số, vừa đề cập đến bộ dữ liệu cực kỳ lớn mà nó được huấn luyện trên đó. Những mô hình như vậy thường có hàng chục, thậm chí hàng trăm tỷ tham số — tức các trọng số có thể điều chỉnh trong mạng — và chúng được tối ưu trong quá trình huấn luyện để dự đoán từ tiếp theo trong một chuỗi.

Việc dự đoán từ tiếp theo là hợp lý vì nó tận dụng bản chất tuần tự vốn có của ngôn ngữ để huấn luyện mô hình học cách nắm bắt ngữ cảnh, cấu trúc và các mối quan hệ bên trong văn bản. Đây là một nhiệm vụ rất đơn giản, vì vậy nhiều nhà nghiên cứu cảm thấy bất ngờ khi nó có thể tạo ra những mô hình mạnh đến vậy. Trong các chương sau, chúng ta sẽ thảo luận và triển khai quy trình huấn luyện dự đoán từ tiếp theo theo từng bước một.

LLM sử dụng một kiến trúc gọi là Transformer, cho phép mô hình tập trung sự chú ý một cách chọn lọc vào các phần khác nhau của dữ liệu đầu vào khi đưa ra dự đoán. Nhờ đó, chúng đặc biệt hiệu quả trong việc xử lý những sắc thái và độ phức tạp của ngôn ngữ con người.
Kiến trúc Transformer giúp LLM ‘chú ý đúng chỗ’ trong văn bản đầu vào, nên mô hình có thể hiểu và xử lý tốt hơn các mối quan hệ và sắc thái tinh tế của ngôn ngữ.

Vì các mô hình ngôn ngữ lớn (LLM) có khả năng tạo sinh văn bản, nên chúng thường được xem là một dạng trí tuệ nhân tạo tạo sinh (generative artificial intelligence), hay thường được viết tắt là generative AI hoặc GenAI

<img width="1349" height="481" alt="image" src="https://github.com/user-attachments/assets/25ef83d5-ff6e-4942-ab1c-5c7ea528b56e" />

>1.1, AI encompasses the broader field of creating machines that can perform tasks requiring human-like intelligence, including understanding language, recognizing patterns, and making decisions, and includes subfields like machine learning and deep learning.

### sơ đồ phân cấp:

```
Artificial Intelligence (AI)
│
├── Machine Learning (ML)
│   │
│   ├── Deep Learning (DL)
│   │   │
│   │   ├── Neural Networks
│   │   │   │
│   │   │   └── Large Language Models (LLMs)
│   │   │       (Transformer-based models)

```

***1️⃣ Artificial Intelligence (AI)***
Mục tiêu lớn: tạo máy móc có hành vi giống trí tuệ con người   
Bao gồm:

     - hiểu ngôn ngữ
     
     - nhận diện mẫu
     
     - ra quyết định

     👉 AI = khái niệm bao trùm

***2️⃣ Machine Learning (ML)***

     - Một nhánh của AI
   
     - Máy không cần code rule cứng
   
     - Học từ dữ liệu → tìm quy luật
   
     👉 “Learn from data, not from rules”


***3️⃣ Deep Learning (DL)***

   Một nhánh của ML
   
     - Dựa trên mạng nơ-ron nhiều tầng
   
     - Scale tốt khi:
   
        - dữ liệu lớn
   
        - compute mạnh
   
     👉 DL = ML + neural networks + scale


***4️⃣ Large Language Models (LLMs)***

   Một ứng dụng cụ thể của Deep Learning
   
        Tập trung vào ngôn ngữ
   
        Thường dùng:
   
             kiến trúc Transformer
   
             bài toán next-word prediction
   
        👉 LLM = Deep Neural Network + Transformer + Massive Text Data



### Keyword:
1. [Deep neural network models (DNN models):](https://chatgpt.com/g/g-p-696e03d1cfd481918a4ca9cdc44a493c-build-a-large-language-model-from-scratch/c/696e03d8-ba1c-8332-a092-3f3c2e82bdb3) 
Deep Neural Network là một hệ thống gồm nhiều tầng toán học nối tiếp nhau, học cách ánh xạ input → output bằng cách tự điều chỉnh trọng số thông qua dữ liệu, thay vì viết rule thủ công.

👉 Trọng số (weights) không “tự nhiên mà có” — nó được khởi tạo ngẫu nhiên, rồi được học dần từ dữ liệu.

2. [Selective attention](https://chatgpt.com/g/g-p-696e03d1cfd481918a4ca9cdc44a493c-build-a-large-language-model-from-scratch/c/696e0f12-05f8-832b-a099-1a7f7ac94294)
   Selective attention trong kiến trúc Transformer là cơ chế cho phép mô hình chọn lọc những phần thông tin quan trọng nhất để tập trung, thay vì xử lý mọi thứ một cách đồng đều.


   **Khi đọc một chuỗi (câu, token, vector):**

    Transformer không “nhìn đều” tất cả token.

    Ở mỗi token, mô hình quyết định token nào đáng chú ý hơn (liên quan hơn) để tổng hợp thông tin.

    Việc “chọn lọc” này diễn ra tự động, thông qua trọng số attention được học trong quá trình train.


> **Ví dụ**
> “Con mèo nằm trên tấm thảm vì nó rất ấm.”
>
> 
> Khi xử lý từ “nó”:
>
> 
> Attention sẽ tập trung mạnh vào “tấm thảm”,
> 
> Ít chú ý hơn tới “con”, “nằm”, “vì”, …
> 
> → Transformer chọn lọc ngữ cảnh có ý nghĩa.
