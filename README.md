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


### Keyword:
1. [Deep neural network models (DNN models):](https://chatgpt.com/g/g-p-696e03d1cfd481918a4ca9cdc44a493c-build-a-large-language-model-from-scratch/c/696e03d8-ba1c-8332-a092-3f3c2e82bdb3) 
Deep Neural Network là một hệ thống gồm nhiều tầng toán học nối tiếp nhau, học cách ánh xạ input → output bằng cách tự điều chỉnh trọng số thông qua dữ liệu, thay vì viết rule thủ công.

👉 Trọng số (weights) không “tự nhiên mà có” — nó được khởi tạo ngẫu nhiên, rồi được học dần từ dữ liệu.
