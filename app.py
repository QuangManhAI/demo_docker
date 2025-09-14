# gọi các thư viện cần thiết
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

# gọi GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dựng lại mạng với đúng cấu hình lúc train để áp model đã lưu, đúng tham số đã huấn luyện.
class PassengerGRU(nn.Module):
    def __init__(self, num_airlines, num_regions, num_activities, num_terminals,
                 input_dim_numeric, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()

        self.airline_embed = nn.Embedding(num_airlines, 32)
        self.region_embed = nn.Embedding(num_regions, 16)
        self.activity_embed = nn.Embedding(num_activities, 8)
        self.terminal_embed = nn.Embedding(num_terminals, 4)
        
        self.numeric_fc = nn.Linear(input_dim_numeric, 32)
        
        total_dim = 32 + 32 + 16 + 8 + 4
        
        self.gru = nn.GRU(total_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x_num, x_airline, x_region, x_activity, x_terminal):
        num_feat = self.numeric_fc(x_num)
        airline_feat = self.airline_embed(x_airline)
        region_feat = self.region_embed(x_region)
        activity_feat = self.activity_embed(x_activity)
        terminal_feat = self.terminal_embed(x_terminal)

        x = torch.cat([num_feat, airline_feat, region_feat,
                       activity_feat, terminal_feat], dim=-1)
        
        out, _ = self.gru(x)  
        
        context = out[:, -1, :]  
        return self.fc_out(context).squeeze(-1)

# gọi lại dựng cấu hình
model = PassengerGRU(
    num_airlines=77,
    num_regions=9,
    num_activities=3,
    num_terminals=5,
    input_dim_numeric=4,
    hidden_size=256,
    num_layers=3,
).to(device)

# áp trọng số, tham số đã huấn luyện lên model đã gọi
model.load_state_dict(torch.load("gru_passenger_best.pt", map_location=device))
model.eval()

# tải cả mean và std
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# code API với FastAPI
app = FastAPI()

# Input là một json gôm các trường được quy định dưới dạng danh sách
# x_num: danh sách đặc trưng số đã chuẩn hóa
# các trường còn lại là các đặc trưng được mã hóa category.
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