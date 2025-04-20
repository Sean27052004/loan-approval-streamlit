import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load mô hình và preprocessor
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("📊 Dự đoán phê duyệt khoản vay")

st.header("🔍 Nhập thông tin khách hàng")

# Flag để gán dữ liệu mẫu chỉ một lần
if "sample_data" not in st.session_state:
    st.session_state.sample_data = None

# Sidebar: chọn mẫu dữ liệu
st.sidebar.title("🎯 Chọn mẫu dữ liệu")
if st.sidebar.button("📋 Mẫu 1"):
    st.session_state.sample_data = "sample_1"
if st.sidebar.button("📋 Mẫu 2"):
    st.session_state.sample_data = "sample_2"
if st.sidebar.button("📋 Mẫu 3"):
    st.session_state.sample_data = "sample_3"
if st.sidebar.button("📋 Mẫu 4"):
    st.session_state.sample_data = "sample_4"
if st.sidebar.button("📋 Mẫu 5"):
    st.session_state.sample_data = "sample_5"
if st.sidebar.button("📋 Mẫu 6"):
    st.session_state.sample_data = "sample_6"
if st.sidebar.button("📋 Mẫu 7"):
    st.session_state.sample_data = "sample_7"
    
# Gán dữ liệu mẫu
if st.session_state.sample_data == "sample_1":
    st.session_state.credit_policy = 1
    st.session_state.purpose = "credit_card"
    st.session_state.interest_rate_percent = 10.5
    st.session_state.installment = 300.0
    st.session_state.annual_inc = 120000.0
    st.session_state.dti = 15.0
    st.session_state.fico = 750
    st.session_state.days_with_cr_line = 4000.0
    st.session_state.revol_bal = 15000.0
    st.session_state.revol_util = 30.0
    st.session_state.inq_last_6mths = 1
    st.session_state.delinq_2yrs = 0
    st.session_state.pub_rec = 0
    st.session_state.sample_data = None

elif st.session_state.sample_data == "sample_2":
    st.session_state.credit_policy = 1
    st.session_state.purpose = "debt_consolidation"
    st.session_state.interest_rate_percent = 18.0
    st.session_state.installment = 500.0
    st.session_state.annual_inc = 60000.0
    st.session_state.dti = 25.0
    st.session_state.fico = 680
    st.session_state.days_with_cr_line = 2500.0
    st.session_state.revol_bal = 8000.0
    st.session_state.revol_util = 65.0
    st.session_state.inq_last_6mths = 2
    st.session_state.delinq_2yrs = 1
    st.session_state.pub_rec = 0
    st.session_state.sample_data = None

elif st.session_state.sample_data == "sample_3":
    st.session_state.credit_policy = 0
    st.session_state.purpose = "small_business"
    st.session_state.interest_rate_percent = 29.5
    st.session_state.installment = 850.0
    st.session_state.annual_inc = 30000.0
    st.session_state.dti = 45.0
    st.session_state.fico = 600
    st.session_state.days_with_cr_line = 1200.0
    st.session_state.revol_bal = 20000.0
    st.session_state.revol_util = 120.0
    st.session_state.inq_last_6mths = 4
    st.session_state.delinq_2yrs = 2
    st.session_state.pub_rec = 1
    st.session_state.sample_data = None

elif st.session_state.sample_data == "sample_4":
    st.session_state.credit_policy = 1
    st.session_state.purpose = "home_improvement"
    st.session_state.interest_rate_percent = 14.0
    st.session_state.installment = 450.0
    st.session_state.annual_inc = 80000.0
    st.session_state.dti = 20.0
    st.session_state.fico = 720
    st.session_state.days_with_cr_line = 3000.0
    st.session_state.revol_bal = 10000.0
    st.session_state.revol_util = 40.0
    st.session_state.inq_last_6mths = 2
    st.session_state.delinq_2yrs = 0
    st.session_state.pub_rec = 0
    st.session_state.sample_data = None

elif st.session_state.sample_data == "sample_5":
    st.session_state.credit_policy = 0
    st.session_state.purpose = "educational"
    st.session_state.interest_rate_percent = 24.5
    st.session_state.installment = 700.0
    st.session_state.annual_inc = 40000.0
    st.session_state.dti = 38.0
    st.session_state.fico = 630
    st.session_state.days_with_cr_line = 1800.0
    st.session_state.revol_bal = 16000.0
    st.session_state.revol_util = 95.0
    st.session_state.inq_last_6mths = 3
    st.session_state.delinq_2yrs = 1
    st.session_state.pub_rec = 1
    st.session_state.sample_data = None

elif st.session_state.sample_data == "sample_6":
    st.session_state.credit_policy = 1
    st.session_state.purpose = "credit_card"
    st.session_state.interest_rate_percent = 8.0
    st.session_state.installment = 200.0
    st.session_state.annual_inc = 150000.0
    st.session_state.dti = 10.0
    st.session_state.fico = 800
    st.session_state.days_with_cr_line = 5000.0
    st.session_state.revol_bal = 5000.0
    st.session_state.revol_util = 20.0
    st.session_state.inq_last_6mths = 0
    st.session_state.delinq_2yrs = 0
    st.session_state.pub_rec = 0
    st.session_state.sample_data = None

elif st.session_state.sample_data == "sample_7":
    st.session_state.credit_policy = 1
    st.session_state.purpose = "debt_consolidation"
    st.session_state.interest_rate_percent = 11.5
    st.session_state.installment = 350.0
    st.session_state.annual_inc = 100000.0
    st.session_state.dti = 18.0
    st.session_state.fico = 740
    st.session_state.days_with_cr_line = 3500.0
    st.session_state.revol_bal = 9000.0
    st.session_state.revol_util = 30.0
    st.session_state.inq_last_6mths = 1
    st.session_state.delinq_2yrs = 0
    st.session_state.pub_rec = 0
    st.session_state.sample_data = None

# Bắt đầu lấy input từ user hoặc session_state

credit_policy = st.selectbox(
    "Có tuân thủ chính sách tín dụng?",
    options=[1, 0],
    format_func=lambda x: "✅ Tuân thủ" if x == 1 else "❌ Không tuân thủ",
    key="credit_policy"
)

purpose = st.selectbox(
    "Mục đích vay",
    options=[
        'credit_card', 'debt_consolidation', 'educational',
        'home_improvement', 'major_purchase', 'small_business'
    ],
    format_func=lambda x: x.replace("_", " ").capitalize(),
    key="purpose"
)

interest_rate_percent = st.number_input(
    "Lãi suất (%/năm)", min_value=1.0, max_value=50.0, step=0.5,
    value=st.session_state.get("interest_rate_percent", 10.0),
    key="interest_rate_percent"
)
int_rate = interest_rate_percent / 100

installment = st.number_input(
    "Khoản trả góp hàng tháng", min_value=1.0, step=1.0,
    value=st.session_state.get("installment", 300.0),
    key="installment"
)

annual_inc = st.number_input(
    "Thu nhập hàng năm (USD)", min_value=1000.0, max_value=500000.0, step=1000.0,
    value=st.session_state.get("annual_inc", 50000.0),
    key="annual_inc"
)

log_annual_inc = np.log(annual_inc)
st.write("📉 Log(Thu nhập):", log_annual_inc)

dti = st.number_input(
    "Tỷ lệ nợ trên thu nhập (DTI)", min_value=0.0, max_value=60.0, step=0.1,
    value=st.session_state.get("dti", 20.0),
    key="dti"
)

fico = st.slider(
    "Điểm FICO", 300, 850,
    value=st.session_state.get("fico", 700),
    key="fico"
)

days_with_cr_line = st.number_input(
    "Số ngày có lịch sử tín dụng", min_value=1.0,
    value=st.session_state.get("days_with_cr_line", 2000.0),
    key="days_with_cr_line"
)

revol_bal = st.number_input(
    "Số dư nợ quay vòng (revol_bal)", min_value=0.0,
    value=st.session_state.get("revol_bal", 5000.0),
    key="revol_bal"
)

revol_util = st.number_input(
    "Tỷ lệ sử dụng tín dụng (revol_util %)", min_value=0.0, max_value=150.0,
    value=st.session_state.get("revol_util", 50.0),
    key="revol_util"
)

inq_last_6mths = st.number_input(
    "Số lần hỏi tín dụng 6 tháng", min_value=0, max_value=10, step=1,
    value=st.session_state.get("inq_last_6mths", 1),
    key="inq_last_6mths"
)

delinq_2yrs = st.number_input(
    "Số lần trễ hạn 2 năm", min_value=0, max_value=10, step=1,
    value=st.session_state.get("delinq_2yrs", 0),
    key="delinq_2yrs"
)

pub_rec = st.number_input(
    "Số bản ghi công khai", min_value=0, max_value=10, step=1,
    value=st.session_state.get("pub_rec", 0),
    key="pub_rec"
)
# Nút submit
if st.button("🚀 Dự đoán kết quả"):
    input_dict = {
        'credit.policy': [credit_policy],
        'int.rate': [int_rate],
        'installment': [installment],
        'log.annual.inc': [log_annual_inc],
        'dti': [dti],
        'fico': [fico],
        'days.with.cr.line': [days_with_cr_line],
        'revol.bal': [revol_bal],
        'revol.util': [revol_util],
        'inq.last.6mths': [inq_last_6mths],
        'delinq.2yrs': [delinq_2yrs],
        'pub.rec': [pub_rec],
        'purpose_credit_card': [1 if purpose == 'credit_card' else 0],
        'purpose_debt_consolidation': [1 if purpose == 'debt_consolidation' else 0],
        'purpose_educational': [1 if purpose == 'educational' else 0],
        'purpose_home_improvement': [1 if purpose == 'home_improvement' else 0],
        'purpose_major_purchase': [1 if purpose == 'major_purchase' else 0],
        'purpose_small_business': [1 if purpose == 'small_business' else 0],
    }

    new_df = pd.DataFrame(input_dict)

    # Feature engineering
    new_df["installment_income_ratio"] = new_df["installment"] / np.exp(new_df["log.annual.inc"])
    new_df["fico_squared"] = new_df["fico"] ** 2
    new_df["dti_squared"] = new_df["dti"] ** 2
    new_df["total_debt"] = new_df["revol.bal"] + new_df["installment"] * 12
    new_df["int.rate_dti"] = new_df["int.rate"] * new_df["dti"]
    new_df["log.annual.inc_installment"] = new_df["log.annual.inc"] * new_df["installment"]

    # Tiền xử lý
    X_final = preprocessor.transform(new_df)

    # Dự đoán
    prob = model.predict_proba(X_final)[0, 1]
    approved = prob < 0.2

    st.subheader("🎯 Kết quả:")
    st.write(f"Xác suất vỡ nợ: **{prob:.2%}**")
    if approved:
        st.success("✅ Khoản vay được PHÊ DUYỆT")
    else:
        st.error("❌ Khoản vay KHÔNG được phê duyệt")
    # Phân nhóm tín dụng
    if prob < 0.05:
        credit_score = "A - Rất tốt"
    elif prob < 0.15:
        credit_score = "B - Tốt"
    elif prob < 0.30:
        credit_score = "C - Trung bình"
    elif prob < 0.50:
        credit_score = "D - Rủi ro"
    else:
        credit_score = "E - Rất rủi ro"

    st.info(f"🏷️ Nhóm tín dụng nội bộ: **{credit_score}**")
