import os
import json
import time
import math
import random
import threading
import logging
from collections import defaultdict, deque
from flask import Flask, jsonify
from flask_cors import CORS
import requests
import numpy as np

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Model Persistence Paths ---
MODEL_STATE_FILE = "model_state_super_vip.json"

# --- Helper Functions ---
def _get_history_chars(history_list):
    """Hàm trợ giúp để lấy danh sách ký tự 't'/'x' từ danh sách dict."""
    return ['t' if item['ket_qua'] == 'Tài' else 'x' for item in history_list]

# --- Model Persistence Functions ---
def save_model_state(app):
    state = {
        'pattern_stats': dict(app.pattern_stats),
        'consecutive_misses': app.consecutive_misses,
        'prediction_performance_stats': app.prediction_performance_stats # Lưu trạng thái hiệu suất dự đoán
    }
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(state, f)
    logging.info("Model state saved.")

def load_model_state(app):
    try:
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
            app.pattern_stats = defaultdict(lambda: {"tai": 0, "xiu": 0, "total": 0}, state.get('pattern_stats', {}))
            app.consecutive_misses = state.get('consecutive_misses', 0)
            app.prediction_performance_stats = state.get('prediction_performance_stats', {"correct": 0, "incorrect": 0, "total": 0}) # Load hiệu suất dự đoán
        logging.info("Model state loaded.")
    except FileNotFoundError:
        logging.warning("No saved model state found. Initializing new models.")
        _initialize_new_state(app)
    except Exception as e:
        logging.error(f"Error loading model state: {e}. Initializing new models.")
        _initialize_new_state(app)

def _initialize_new_state(app):
    app.pattern_stats = defaultdict(lambda: {"tai": 0, "xiu": 0, "total": 0})
    app.consecutive_misses = 0
    app.prediction_performance_stats = {"correct": 0, "incorrect": 0, "total": 0} # Khởi tạo hiệu suất dự đoán

# --- Core Logic Functions ---

def update_pattern_stats(app, history_chars_full, actual_result_char):
    """
    Cập nhật thống kê cho TẤT CẢ các pattern con có thể tạo ra từ lịch sử.
    Mỗi pattern con có độ dài từ 2 đến MAX_PATTERN_LENGTH.
    """
    MAX_PATTERN_LENGTH = 15 # Giới hạn chiều dài pattern để tránh quá tải
    
    if len(history_chars_full) < 2:
        return

    history_chars_for_pattern = history_chars_full[:-1] # Lịch sử (N-1 phiên)

    for length in range(2, min(len(history_chars_for_pattern) + 1, MAX_PATTERN_LENGTH + 1)):
        pattern_sub_string = "".join(history_chars_for_pattern[-length:])
        
        stats = app.pattern_stats[pattern_sub_string]
        stats['total'] += 1
        if actual_result_char == 't':
            stats['tai'] += 1
        else:
            stats['xiu'] += 1

def get_pattern_based_prediction(app, history_chars, window_size=100):
    """
    Phân tích các pattern trong 'window_size' phiên gần nhất để đưa ra dự đoán.
    Ưu tiên pattern dài nhất, có độ chính xác cao nhất và đủ số lần xuất hiện.
    """
    if len(history_chars) < 2:
        return {
            "name": "Không đủ lịch sử", "length": 0, "history_str": "",
            "tai_count": 0, "xiu_count": 0, "total_count": 0,
            "pred_match": None, "pred_match_percent": 0
        }

    recent_history = history_chars[-window_size:]
    MAX_PATTERN_LENGTH = 15 
    
    best_pattern_info = {
        "name": "Xu hướng chung",
        "length": 0,
        "history_str": "",
        "tai_count": 0,
        "xiu_count": 0,
        "total_count": 0,
        "pred_match": None,
        "pred_match_percent": 0.0
    }
    
    best_score = -1
    MIN_OCCURRENCES = 5

    for length in range(min(len(recent_history), MAX_PATTERN_LENGTH), 1, -1):
        current_pattern_str = "".join(recent_history[-length:])
        
        if current_pattern_str in app.pattern_stats:
            stats = app.pattern_stats[current_pattern_str]
            if stats['total'] >= MIN_OCCURRENCES:
                accuracy_tai = (stats['tai'] / stats['total']) if stats['total'] > 0 else 0
                accuracy_xiu = (stats['xiu'] / stats['total']) if stats['total'] > 0 else 0
                
                current_pred_match = 't' if accuracy_tai >= accuracy_xiu else 'x'
                current_pred_percent = max(accuracy_tai, current_pred_match) # Fixed: current_pred_match is char, should be accuracy_xiu
                
                # Corrected: current_pred_percent should be max(accuracy_tai, accuracy_xiu)
                current_pred_percent = max(accuracy_tai, accuracy_xiu)

                current_score = current_pred_percent * math.sqrt(stats['total']) * (length / MAX_PATTERN_LENGTH)

                if current_score > best_score:
                    best_score = current_score
                    best_pattern_info = {
                        "name": f"Pattern '{current_pattern_str}'",
                        "length": length,
                        "history_str": current_pattern_str,
                        "tai_count": stats['tai'],
                        "xiu_count": stats['xiu'],
                        "total_count": stats['total'],
                        "pred_match": 'Tài' if current_pred_match == 't' else 'Xỉu',
                        "pred_match_percent": round(current_pred_percent * 100, 2)
                    }
    
    if best_pattern_info["pred_match"] is None:
        tai_count_overall = recent_history.count('t')
        xiu_count_overall = recent_history.count('x')
        total_overall = tai_count_overall + xiu_count_overall

        percent_tai_overall = (tai_count_overall / total_overall * 100) if total_overall > 0 else 50
        percent_xiu_overall = (xiu_count_overall / total_overall * 100) if total_overall > 0 else 50

        best_pattern_info["name"] = "Xu hướng chung"
        best_pattern_info["pred_match"] = 'Tài' if percent_tai_overall >= percent_xiu_overall else 'Xỉu'
        best_pattern_info["pred_match_percent"] = max(percent_tai_overall, percent_xiu_overall)
        best_pattern_info["tai_count"] = tai_count_overall
        best_pattern_info["xiu_count"] = xiu_count_overall
        best_pattern_info["total_count"] = total_overall

    return best_pattern_info

def apply_meta_logic(app, prediction, confidence, history_chars):
    """
    Áp dụng logic cấp cao để điều chỉnh dự đoán cuối cùng.
    Ví dụ: Logic "bẻ cầu" khi cầu quá dài và "Đang trật 3 lần -> Auto đảo ngược".
    """
    final_prediction, final_confidence, reason = prediction, confidence, ""

    # Logic 1: Bẻ cầu khi cầu bệt quá dài (Anti-Streak)
    streak_char = history_chars[-1] if history_chars else None
    streak_len = 0
    if streak_char:
        for x in reversed(history_chars):
            if x == streak_char: streak_len += 1
            else: break
    
    if streak_len >= 9 and prediction.lower()[0] == streak_char:
        final_prediction = 'Xỉu' if streak_char == 't' else 'Tài'
        final_confidence = 78.0
        reason = f"Bẻ cầu bệt siêu dài ({streak_len})"
        logging.warning(f"META-LOGIC: Activated Anti-Streak. Streak of {streak_len} detected. Forcing prediction to {final_prediction}.")
    elif streak_len >= 7 and prediction.lower()[0] == streak_char:
        final_confidence = max(50.0, confidence - 15)
        reason = f"Cầu bệt dài ({streak_len}), giảm độ tin cậy"
        logging.info(f"META-LOGIC: Long streak of {streak_len} detected. Reducing confidence.")
    
    # Logic 2: "Đang trật 3 lần -> Auto đảo ngược"
    if app.consecutive_misses >= 3:
        initial_pred = final_prediction 
        final_prediction = 'Xỉu' if final_prediction == 'Tài' else 'Tài'
        final_confidence = 90.0
        reason = f"Đang trật {app.consecutive_misses} lần -> Auto đảo ngược từ {initial_pred} sang {final_prediction}"
        logging.warning(f"META-LOGIC: Activated Auto-Reverse. Missed {app.consecutive_misses} times. Forcing prediction to {final_prediction}.")

    return final_prediction, final_confidence, reason

def predict_super_vip(app, history_chars):
    """
    Thực hiện dự đoán siêu VIP dựa trên pattern và xu hướng trong 100 phiên gần nhất.
    """
    if len(history_chars) < 20: # Cần một lượng lịch sử đủ lớn để phân tích pattern
        return {
            "id": "heheviptool", # Thêm ID công cụ
            "Phien_moi": app.history[-1]['phien'] + 1 if app.history else 0,
            "pattern_length": 0,
            "pattern": "",
            "matches": [],
            "pattern_tai": 0,
            "pattern_xiu": 0,
            "pattern_percent_tai": 0,
            "pattern_percent_xiu": 0,
            "phan_tram_tai": 0,
            "phan_tram_xiu": 0,
            "tong_tai": 0.0,
            "tong_xiu": 0.0,
            "du_doan": "Chờ dữ liệu",
            "ly_do": "Chưa có đủ dữ liệu lịch sử",
            "phien_du_doan": app.history[-1]['phien'] + 1 if app.history else 0
        }

    pattern_analysis = get_pattern_based_prediction(app, history_chars)

    raw_prediction = pattern_analysis['pred_match']
    raw_confidence = pattern_analysis['pred_match_percent']

    WEIGHT_PATTERN = 0.7
    WEIGHT_OVERALL = 0.3

    score_tai = 0.0
    score_xiu = 0.0

    if raw_prediction == 'Tài':
        score_tai += raw_confidence * WEIGHT_PATTERN
        score_xiu += (100 - raw_confidence) * WEIGHT_PATTERN
    else:
        score_xiu += raw_confidence * WEIGHT_PATTERN
        score_tai += (100 - raw_confidence) * WEIGHT_PATTERN
    
    phan_tram_tai_overall, phan_tram_xiu_overall = get_overall_balance(history_chars, window_size=100)
    score_tai += phan_tram_tai_overall * WEIGHT_OVERALL
    score_xiu += phan_tram_xiu_overall * WEIGHT_OVERALL

    final_prediction_raw = 'Tài' if score_tai >= score_xiu else 'Xỉu'
    total_score_raw = score_tai + score_xiu
    final_confidence_raw = (max(score_tai, score_xiu) / total_score_raw) * 100 if total_score_raw > 0 else 50.0
    main_reason = pattern_analysis['name']

    final_prediction, final_confidence, meta_reason = apply_meta_logic(app, final_prediction_raw, final_confidence_raw, history_chars)

    response_data = {
        "id": "heheviptool", # Thêm ID công cụ
        "Phien_moi": app.history[-1]['phien'] + 1 if app.history else 0,
        "pattern_length": pattern_analysis['length'],
        "pattern": pattern_analysis['history_str'],
        "matches": [pattern_analysis['pred_match'].lower()[0]] if pattern_analysis['pred_match'] else [],
        "pattern_tai": pattern_analysis['tai_count'],
        "pattern_xiu": pattern_analysis['xiu_count'],
        "pattern_percent_tai": round(pattern_analysis['pred_match_percent'] if pattern_analysis['pred_match'] == 'Tài' else 100 - pattern_analysis['pred_match_percent'], 2),
        "pattern_percent_xiu": round(pattern_analysis['pred_match_percent'] if pattern_analysis['pred_match'] == 'Xỉu' else 100 - pattern_analysis['pred_match_percent'], 2),
        "phan_tram_tai": round(phan_tram_tai_overall, 2),
        "phan_tram_xiu": round(phan_tram_xiu_overall, 2),
        "tong_tai": round(score_tai, 2),
        "tong_xiu": round(score_xiu, 2),
        "du_doan": final_prediction,
        "ly_do": meta_reason if meta_reason else main_reason,
        "phien_du_doan": app.history[-1]['phien'] + 1 if app.history else 0
    }
    
    app.last_prediction = {
        'session': response_data['phien_du_doan'],
        'prediction': final_prediction,
    }

    return response_data

def get_overall_balance(history_chars, window_size):
    """Tính tỷ lệ Tài/Xỉu trong cửa sổ lịch sử gần nhất."""
    recent_history = history_chars[-window_size:]
    tai_count = recent_history.count('t')
    xiu_count = recent_history.count('x')
    total = tai_count + xiu_count
    
    percent_tai = (tai_count / total * 100) if total > 0 else 50
    percent_xiu = (xiu_count / total * 100) if total > 0 else 50
    
    return percent_tai, percent_xiu

# --- Flask App Factory ---
def create_app():
    app = Flask(__name__)
    CORS(app)

    # --- Khởi tạo State ---
    app.lock = threading.Lock()
    app.MAX_HISTORY_LEN = 1000

    app.history = deque(maxlen=app.MAX_HISTORY_LEN)
    app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN)
    
    app.pattern_stats = defaultdict(lambda: {"tai": 0, "xiu": 0, "total": 0})
    app.last_prediction = None
    app.consecutive_misses = 0 
    app.prediction_performance_stats = {"correct": 0, "incorrect": 0, "total": 0} # Biến mới để lưu hiệu suất

    load_model_state(app)

    app.API_URL = os.getenv("API_URL", "https://apihitclub.up.railway.app/api/taixiu")

    def fetch_data_from_api():
        while True:
            try:
                response = requests.get(app.API_URL, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                ket_qua = data.get("Ket_qua")
                phien = data.get("Phien")

                if ket_qua == "X\u1ec9u":
                    ket_qua = "Xỉu"
                elif ket_qua == "T\u00e0i":
                    ket_qua = "Tài"
                
                if ket_qua not in ["Tài", "Xỉu"] or phien is None:
                    logging.warning(f"Invalid data received from API: {data}")
                    time.sleep(5)
                    continue
                
                with app.lock:
                    if not app.session_ids or phien > app.session_ids[-1]:
                        prev_history_chars_for_training = _get_history_chars(list(app.history))
                        actual_result_char = 't' if ket_qua == 'Tài' else 'x'
                        
                        app.session_ids.append(phien)
                        app.history.append({'ket_qua': ket_qua, 'phien': phien})
                        logging.info(f"New result from API for session {phien}: {ket_qua}")

                        # --- Bước học Online (Online Learning) và Cập nhật hiệu suất ---
                        if app.last_prediction and app.last_prediction['session'] == phien:
                            
                            app.prediction_performance_stats["total"] += 1
                            if app.last_prediction['prediction'] == ket_qua:
                                app.consecutive_misses = 0 # Reset nếu dự đoán đúng
                                app.prediction_performance_stats["correct"] += 1
                                logging.info(f"Prediction Correct! Resetting consecutive misses.")
                            else:
                                app.consecutive_misses += 1
                                app.prediction_performance_stats["incorrect"] += 1
                                logging.warning(f"Prediction Missed! Consecutive misses: {app.consecutive_misses}")

                            # Cập nhật thống kê pattern
                            update_pattern_stats(app, prev_history_chars_for_training + [actual_result_char], actual_result_char)
                            
                            save_model_state(app)
                            logging.info(f"Learned from session {phien}. Prediction was {app.last_prediction['prediction']}, actual was {ket_qua}.")
                            
                        app.last_prediction = None # Reset last_prediction để chờ phiên tiếp theo
                    else:
                        logging.debug(f"Session {phien} already processed or older. Current latest: {app.session_ids[-1]}")
                        
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data from API: {e}")
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from API response: {response.text}")
            except Exception as e:
                logging.error(f"Unhandled error in API data fetching thread: {e}")
            
            time.sleep(10)

    # --- API Endpoints ---
    @app.route("/api/taixiu", methods=["GET"])
    def get_taixiu_prediction():
        with app.lock:
            history_copy = list(app.history)
        
        history_chars_for_prediction = _get_history_chars(history_copy)
        
        prediction_output = predict_super_vip(app, history_chars_for_prediction)
        
        return jsonify(prediction_output)

    @app.route("/api/history", methods=["GET"])
    def get_history_api():
        with app.lock:
            hist_copy = list(app.history)
        return jsonify({"history": hist_copy, "length": len(hist_copy)})

    @app.route("/api/performance", methods=["GET"])
    def get_performance():
        with app.lock:
            MIN_OCCURRENCES_FOR_DISPLAY = 10 
            seen_patterns = {k: v for k, v in app.pattern_stats.items() if v['total'] >= MIN_OCCURRENCES_FOR_DISPLAY} 
            sorted_patterns = sorted(
                seen_patterns.items(), 
                key=lambda item: (item[1]['total'], max((item[1]['tai'] / item[1]['total']), (item[1]['xiu'] / item[1]['total'])) if item[1]['total'] > 0 else 0),
                reverse=True
            )
            pattern_result = {}
            for p_key, data in sorted_patterns[:50]:
                accuracy_tai = round(data["tai"] / data["total"] * 100, 2) if data["total"] > 0 else 0
                accuracy_xiu = round(data["xiu"] / data["total"] * 100, 2) if data["total"] > 0 else 0
                pattern_result[p_key] = { 
                    "total": data["total"], 
                    "tai_count": data["tai"], 
                    "xiu_count": data["xiu"], 
                    "accuracy_tai_percent": accuracy_tai,
                    "accuracy_xiu_percent": accuracy_xiu,
                    "best_accuracy_percent": max(accuracy_tai, accuracy_xiu)
                }
            
        return jsonify({
            "id": "heheviptool", # Thêm ID công cụ
            "pattern_performance_detailed": pattern_result, 
            "consecutive_misses_current": app.consecutive_misses,
            "overall_prediction_performance": app.prediction_performance_stats # Thêm hiệu suất tổng thể
        })

    # Khởi động luồng lấy dữ liệu API
    api_fetch_thread = threading.Thread(target=fetch_data_from_api, daemon=True)
    api_fetch_thread.start()
    return app

# --- Thực thi chính ---
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Flask app ready. Serving on http://0.0.0.0:{port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=8)
