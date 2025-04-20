import streamlit as st
import numpy as np

# Flag để gán dữ liệu mẫu chỉ một lần
if "sample_data" not in st.session_state:
    st.session_state.sample_data = None

# Sidebar: chọn mẫu dữ liệu
st.sidebar.title("🎯 Chọn dữ liệu mẫu")
if st.sidebar.button("📋 Mẫu: Khách tốt"):
    st.session_state.sample_data = "good"
if st.sidebar.button("📋 Mẫu: Trung bình"):
    st.session_state.sample_data = "average"
if st.sidebar.button("📋 Mẫu: Rủi ro cao"):
    st.session_state.sample_data = "risky"

# Gán dữ liệu mẫu
if st.session_state.sample_data == "good":
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

elif st.session_state.sample_data == "average":
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

elif st.session_state.sample_data == "risky":
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

# Bắt đầu lấy input từ user hoặc session_state
st.title("📊 Dự đoán phê duyệt khoản vay")
st.header("🔍 Nhập thông tin khách hàng")

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
