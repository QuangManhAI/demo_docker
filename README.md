## Báo cáo cho nhiệm vụ: 
### *Sử dụng Docker. Xử lý dữ liệu, Huấn luyện mô hình với kiến trúc GRU. Viết Dockerfile, tạo Docker Image và Docker Container. Triển khai container local hoặc ngrok để demo.*
#### 1. Khái quát xử lí dữ liệu và huấn luyện mô hình.
* Dữ liệu hành khách theo tháng được tiền xử lí:
    * Chuẩn hóa đặc trưng số (*Passenger Count, Year, Month*).
    * Mã hóa embedding cho đặc trưng phân loại (*Airline, Region, Activity, Terminal*).
    * Chuỗi thời gian được xây dựng dưới dạng *sliding window* (12 tháng đầu vào dự đoán tháng 13).
    * Mô hình GRU nhiều lớp được thiết kế, kết hợp embedding và đặc trưng số.
    * Huấn luyện mô hình với hàm mất mát MSE, đánh giá bằng các chỉ số RMSE, MAE, $R^2$.
    * Kết quả:
        ```
      Train Loss: 0.0262
      Val Loss: 0.0504
      RMSE: 13089.48 
      MAE: 4995.36 
      R²: 0.9498
        ```
    * Mô hình có kết quả khá tốt đặc biệt là $R^2$ (có nghĩa là mô hình giải thích được **95% phương sai** của dữ liệu, tức là mức độ khớp rất cao)
#### 2. Kiến thức về FastAPI
* FastAPI là một framework web hiện đại cho python, dùng để xây dựng API. Em chọn FastAPI vì nó thân thiện với triển khai mô hình, tốc độ cao, dễ sử dụng, hỗ trợ cho mô hình tốt như em đang huấn luyện.
* Đặc điểm chính: 
    * FastAPI được xây dựng trên Starlette và Uvicorn, tốc độ gần bằng Node.js và Go.
    * Cú pháp rõ ràng, khai bóa check point như viết hàm python.
    * Khi chạy FastAPi tự sinh Swagger UI và ReDoc.
    * Hỗ trợ async/ await bất đồng bộ cho hệ thống xử lí nhiều yêu cầu.
* Ứng dụng:
    * Xây dựng dịch vụ web nhỏ gọn.
    * Triển khai mô hình ML/DL dưới dạng API.
    * Có thể tích hợp vào hệ thống Micoservices.
* Quy trình:
    * Cài đặt: 
        ```
        pip install fastapi uvicorn
        ```
    * Code trong app.py:
        ```
        class PassengerInput(BaseModel):
            x_num: list
            airline: list
            activity: list
            region: list
            terminal: list

        @app.post("/predict")
        # reshape các input thành tensor
        def predict(data: PassengerInput):
            x_num = np.array(data.x_num, dtype=np.float32).reshape(1, 12, 4)
            x_airline = np.array(data.airline, dtype=np.int64).reshape(1, 12)
            x_region = np.array(data.region, dtype=np.int64).reshape(1, 12)
            x_activity = np.array(data.activity, dtype=np.int64).reshape(1, 12)
            x_terminal = np.array(data.terminal, dtype=np.int64).reshape(1, 12)

            # gọi model để dự báo (inference)
            with torch.no_grad():
                pred = model(
                    torch.tensor(x_num).to(device),
                    torch.tensor(x_airline).to(device),
                    torch.tensor(x_region).to(device),
                    torch.tensor(x_activity).to(device),
                    torch.tensor(x_terminal).to(device),
                )
                # dùng std và mean để đưa về kết quả gốc
                pred_real = scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1,1))[0][0]

                return {"prediction": float(pred_real)}
        ```
    * Chạy:
        ```
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```
#### 3. Kiến thức về Docker
* Docker là công cụ giúp **đóng gói** ứng dụng, môi trường, thư viện, hệ điều hành... vào một cái hộp gọi là container.
* Container như một Virtual Machine nhỏ, nhẹ. Vì nó dùng chung OS, nhân kernel chung với hệ điều hành.
* Lợi ích: Dễ dàng chạy trên nhiều môi trường, hệ điều hành khác nhau. Vì hộp đã đóng gói toàn bộ những thứ cần thiết. Không lo bị thiếu hay xung đột. Chạy khắp nơi với FastAPI + model AI chẳng hạn chỉ cần build và run container. Nhẹ, nhanh ít tốn tài nguyên.
* Hiểu về các khái niệm chính và sự liên tưởng chính.
    * Dockerfile: Như là một kịch bản hay bản thiết kế để tạo ra docker image. Chứa các chỉ dẫn như base image nào, cần copy file code nào, cần thư nào để chạy...
        ```
        # python 3.12 đúng như môi trường
        FROM python:3.12-slim

        # work directory 
        WORKDIR /app

        # copy các file trong thư mục vào
        COPY requirements.txt .
        COPY app.py .
        COPY gru_passenger_best.pt .
        COPY scaler_X.pkl .
        COPY scaler_y.pkl .

        # lệnh cần chạy cài thư viện cần thiết
        RUN pip install --no-cache-dir -r requirements.txt
        # expose cổng 8000
        EXPOSE 8000
        # lệnh để chạy cuối 
        CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        ```

    * Docker image: Là khuôn mẫu tạo từ bản thiết kế dockerfile.
    Khi build image là lúc ra xây container từ trong ra ngoài. mang mọi thứ vào và đóng container mẫu này lại. 
    * Docker container: là thể hiện của image (Một image có thể tạo nhiều thể hiện và chạy).
    * Docker Hub: là chợ chứa các image có sẵn (Như python:3.12-slim).
* Quy trình:
    * Viết code, huấn luyện mô hình, chuẩn bị các file liên quan ,viết Dockerfile.
    * Build image:
        ```
        sudo docker build -t passenger-gru .
        ```
    * Chạy container từ image:
        ```
        sudo docker run -d -p 8000:8000 passenger-gru
        ```
#### 4. Sử dụng Ngrok
* Ngrok là công cụ tạo hầm từ máy tính cá nhân ra trang web. 
* Hiểu đơn giản là chạy server API trên local sau đó nhờ Ngrok sẽ cung cấp một link để người khác có thể truy cập.
* Dùng để demo mà không cần server như GCP, AWS, Azure...
* Chỉ dùng để demo vì máy cá nhân phải mở thì Ngrok mới có nguồn hầm.
* Quy trình:
    * Cài trên linux và chạy với token được cấp trên trang chủ Ngrok:
        ```
        sudo snap install ngrok
        ngrok config add-authtoken <thay token vào>
        ```
    * Chạy với FastAPI:
        ```
        uvicorn main:app --reload --port 8000
        ```
    * Mở terminal khác và chạy:
        ```
        ngrok http 8000
        ```
    * Sẽ thấy:
        ```
        Forwarding    https://ffeb5020e2fa.ngrok-free.app -> http://localhost:8000
        ```

* Có thể thử với `sample.json`:

    ```
        curl -X POST "https://ffeb5020e2fa.ngrok-free.app predict" \
        -H "Content-Type: application/json" \
        -d @sample.json
    ``` 
