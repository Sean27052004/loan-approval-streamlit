import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load mô hình và preprocessor
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("📊 Dự đoán phê duyệt khoản vay")

st.header("🔍 Nhập thông tin khách hàng")

# Các thông tin đầu vào
credit_policy = st.selectbox("Có tuân thủ chính sách tín dụng?", [0, 1], key="credit_policy")
purpose = st.selectbox("Mục đích vay", [
    'credit_card', 'debt_consolidation', 'educational',
    'home_improvement', 'major_purchase', 'small_business'
], key="purpose")
int_rate = st.number_input("Lãi suất (int.rate)", min_value=0.0, step=0.01)
installment = st.number_input("Khoản trả góp hàng tháng", min_value=0.0, step=1.0)
log_annual_inc = st.number_input("log(Thu nhập hàng năm)", min_value=0.0, step=0.1)
dti = st.number_input("Tỷ lệ nợ trên thu nhập (dti)", min_value=0.0, step=0.1)
fico = st.slider("Điểm FICO", 300, 850, step=1)
days_with_cr_line = st.number_input("Số ngày có lịch sử tín dụng", min_value=0.0)
revol_bal = st.number_input("Số dư nợ quay vòng (revol_bal)", min_value=0.0)
revol_util = st.number_input("Tỷ lệ sử dụng tín dụng (revol_util)", min_value=0.0, max_value=100.0)
inq_last_6mths = st.number_input("Số lần hỏi tín dụng 6 tháng", min_value=0, step=1)
delinq_2yrs = st.number_input("Số lần trễ hạn 2 năm", min_value=0, step=1)
pub_rec = st.number_input("Số bản ghi công khai", min_value=0, step=1)

# Nút submit
if st.button("🚀 Dự đoán kết quả"):
    # Lấy giá trị mục đích vay từ dropdown
    purpose = st.selectbox("Mục đích vay", [
        'credit_card', 'debt_consolidation', 'educational',
        'home_improvement', 'major_purchase', 'small_business'
    ])

    # Chuyển thành DataFrame
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

    # Tạo feature phụ
    new_df["installment_income_ratio"] = new_df["installment"] / np.exp(new_df["log.annual.inc"])
    new_df["fico_squared"] = new_df["fico"] ** 2
    new_df["dti_squared"] = new_df["dti"] ** 2
    new_df["total_debt"] = new_df["revol.bal"] + new_df["installment"] * 12  # ĐÃ SỬA CÔNG THỨC
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
