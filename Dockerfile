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