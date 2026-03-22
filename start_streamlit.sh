PORT=8513
TP_DIR="TP2"
cd $TP_DIR
streamlit run ./app.py --server.port $PORT --server.address 0.0.0.0