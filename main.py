import os
import json
import time
import math
import random
import threading
import logging
from collections import defaultdict, deque
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Helper Functions ---
def _get_history_strings(history_list):
    """Hàm trợ giúp để lấy danh sách chuỗi 'Tài'/'Xỉu' từ danh sách dict."""
    return [item['ket_qua'] for item in history_list]

# 1. Định nghĩa các Patterns
def define_patterns():
    """
    Định nghĩa một bộ sưu tập lớn các patterns từ đơn giản đến siêu phức tạp.
    Mỗi pattern là một hàm lambda nhận lịch sử (dạng chuỗi) và trả về True nếu khớp.
    """
    patterns = {
        # --- Cầu Bệt (Streaks) ---
        "Bệt": lambda h: len(h) >= 3 and h[-1] == h[-2] == h[-3],
        "Bệt siêu dài": lambda h: len(h) >= 5 and all(x == h[-1] for x in h[-5:]),
        "Bệt gãy nhẹ": lambda h: len(h) >= 4 and h[-1] != h[-2] and h[-2] == h[-3] == h[-4],
        "Bệt gãy sâu": lambda h: len(h) >= 5 and h[-1] != h[-2] and all(x == h[-2] for x in h[-5:-1]),
        "Bệt xen kẽ ngắn": lambda h: len(h) >= 4 and h[-4:-2] == [h[-4]]*2 and h[-2:] == [h[-2]]*2 and h[-4] != h[-2],
        "Bệt ngược": lambda h: len(h) >= 4 and h[-1] == h[-2] and h[-3] == h[-4] and h[-1] != h[-3],
        "Xỉu kép": lambda h: len(h) >= 2 and h[-1] == 'Xỉu' and h[-2] == 'Xỉu',
        "Tài kép": lambda h: len(h) >= 2 and h[-1] == 'Tài' and h[-2] == 'Tài',
        "Ngẫu nhiên bệt": lambda h: len(h) > 8 and 0.4 < (h[-8:].count('Tài') / 8) < 0.6 and h[-1] == h[-2],

        # --- Cầu Đảo (Alternating) ---
        "Đảo 1-1": lambda h: len(h) >= 4 and h[-1] != h[-2] and h[-2] != h[-3] and h[-3] != h[-4],
        "Xen kẽ dài": lambda h: len(h) >= 5 and all(h[i] != h[i+1] for i in range(-5, -1)),
        "Xen kẽ": lambda h: len(h) >= 3 and h[-1] != h[-2] and h[-2] != h[-3],
        "Xỉu lắc": lambda h: len(h) >= 4 and h[-4:] == ['Xỉu', 'Tài', 'Xỉu', 'Tài'],
        "Tài lắc": lambda h: len(h) >= 4 and h[-4:] == ['Tài', 'Xỉu', 'Tài', 'Xỉu'],
        
        # --- Cầu theo nhịp (Rhythmic) ---
        "Kép 2-2": lambda h: len(h) >= 4 and h[-4:] == [h[-4], h[-4], h[-2], h[-2]] and h[-4] != h[-2],
        "Nhịp 3-3": lambda h: len(h) >= 6 and all(x == h[-6] for x in h[-6:-3]) and all(x == h[-3] for x in h[-3:]),
        "Nhịp 4-4": lambda h: len(h) >= 8 and h[-8:-4] == [h[-8]]*4 and h[-4:] == [h[-4]]*4 and h[-8] != h[-4],
        "Lặp 2-1": lambda h: len(h) >= 3 and h[-3:-1] == [h[-3], h[-3]] and h[-1] != h[-3],
        "Lặp 3-2": lambda h: len(h) >= 5 and h[-5:-2] == [h[-5]]*3 and h[-2:] == [h[-2]]*2 and h[-5] != h[-2],
        "Cầu 3-1": lambda h: len(h) >= 4 and all(x == h[-4] for x in h[-4:-1]) and h[-1] != h[-4],
        "Cầu 4-1": lambda h: len(h) >= 5 and h[-5:-1] == [h[-5]]*4 and h[-1] != h[-5],
        "Cầu 1-2-1": lambda h: len(h) >= 4 and h[-4] != h[-3] and h[-3]==h[-2] and h[-2] != h[-1] and h[-4]==h[-1],
        "Cầu 2-1-2": lambda h: len(h) >= 5 and h[-5:-3] == [h[-5]]*2 and h[-2] != h[-5] and h[-1] == h[-5],
        "Cầu 3-1-2": lambda h: len(h) >= 6 and h[-6:-3]==[h[-6]]*3 and h[-3]!=h[-2] and h[-2:]==[h[-2]]*2 and len(set(h[-6:])) == 2,
        "Cầu 1-2-3": lambda h: len(h) >= 6 and h[-6:-5]==[h[-6]] and h[-5:-3]==[h[-5]]*2 and h[-3:]==[h[-3]]*3 and len(set(h[-6:])) == 2,
        "Dài ngắn đảo": lambda h: len(h) >= 5 and h[-5:-2] == [h[-5]] * 3 and h[-2] != h[-1] and h[-2] != h[-5],

        # --- Cầu Chu Kỳ & Đối Xứng (Cyclic & Symmetric) ---
        "Chu kỳ 2": lambda h: len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4],
        "Chu kỳ 3": lambda h: len(h) >= 6 and h[-1] == h[-4] and h[-2] == h[-5] and h[-3] == h[-6],
        "Chu kỳ 4": lambda h: len(h) >= 8 and h[-8:-4] == h[-4:],
        "Đối xứng (Gương)": lambda h: len(h) >= 5 and h[-1] == h[-5] and h[-2] == h[-4],
        "Bán đối xứng": lambda h: len(h) >= 5 and h[-1] == h[-4] and h[-2] == h[-5],
        "Ngược chu kỳ": lambda h: len(h) >= 4 and h[-1] == h[-4] and h[-2] == h[-3] and h[-1] != h[-2],
        "Chu kỳ biến đổi": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-4], h[-5], h[-4], h[-5]],
        "Cầu linh hoạt": lambda h: len(h) >= 6 and h[-1]==h[-3]==h[-5] and h[-2]==h[-4]==h[-6],
        "Chu kỳ tăng": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-6], h[-5], h[-6], h[-5]] and h[-6] != h[-5],
        "Chu kỳ giảm": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-6], h[-5], h[-5], h[-4], h[-4]] and len(set(h[-6:])) == 3,
        "Cầu lặp": lambda h: len(h) >= 6 and h[-6:-3] == h[-3:],
        "Gãy ngang": lambda h: len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4] and h[-1] != h[-2],

        # --- Cầu Phức Tạp & Tổng Hợp ---
        "Gập ghềnh": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-5], h[-3], h[-3], h[-5]],
        "Bậc thang": lambda h: len(h) >= 3 and h[-3:] == [h[-3], h[-3], h[-1]] and h[-3] != h[-1],
        "Cầu đôi": lambda h: len(h) >= 4 and h[-1] == h[-2] and h[-3] != h[-4] and h[-3] != h[-1],
        "Đối ngược": lambda h: len(h) >= 4 and h[-1] == ('Xỉu' if h[-2]=='Tài' else 'Tài') and h[-3] == ('Xỉu' if h[-4]=='Tài' else 'Tài'),
        "Cầu gập": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-4], h[-4], h[-2], h[-2]],
        "Phối hợp 1": lambda h: len(h) >= 5 and h[-1] == h[-2] and h[-3] != h[-4],
        "Phối hợp 2": lambda h: len(h) >= 4 and h[-4:] == ['Tài', 'Tài', 'Xỉu', 'Tài'],
        "Phối hợp 3": lambda h: len(h) >= 4 and h[-4:] == ['Xỉu', 'Xỉu', 'Tài', 'Xỉu'],
        "Chẵn lẻ lặp": lambda h: len(h) >= 4 and len(set(h[-4:-2])) == 1 and len(set(h[-2:])) == 1 and h[-1] != h[-3],
        "Cầu dài ngẫu": lambda h: len(h) >= 7 and all(x == h[-7] for x in h[-7:-3]) and len(set(h[-3:])) > 1,
        
        # --- Cầu Dựa Trên Phân Bố (Statistical) ---
        "Ngẫu nhiên": lambda h: len(h) > 10 and 0.4 < (h[-10:].count('Tài') / 10) < 0.6,
        "Đa dạng": lambda h: len(h) >= 5 and len(set(h[-5:])) == 2,
        "Phân cụm": lambda h: len(h) >= 6 and (all(x == 'Tài' for x in h[-6:-3]) or all(x == 'Xỉu' for x in h[-6:-3])),
        "Lệch ngẫu nhiên": lambda h: len(h) > 10 and (h[-10:].count('Tài') / 10 > 0.7 or h[-10:].count('Xỉu') / 10 > 0.7),

        # --- Siêu Cầu (Super Patterns) ---
        "Cầu Tiến 1-1-2-2": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-4], h[-4], h[-2], h[-2]] and len(set(h[-6:])) == 2,
        "Cầu Lùi 3-2-1": lambda h: len(h) >= 6 and h[-6:-3]==[h[-6]]*3 and h[-3:-1]==[h[-3]]*2 and h[-1]!=h[-3] and len(set(h[-6:])) == 2,
        "Cầu Sandwich": lambda h: len(h) >= 5 and h[-1] == h[-5] and h[-2] == h[-3] == h[-4] and h[-1] != h[-2],
        "Cầu Thang máy": lambda h: len(h) >= 7 and h[-7:] == [h[-7],h[-7],h[-5],h[-5],h[-3],h[-3],h[-1]] and len(set(h[-7:]))==4, # T-T-X-X-T-T-X
        "Cầu Sóng vỗ": lambda h: len(h) >= 8 and h[-8:] == [h[-8],h[-8],h[-6],h[-8],h[-8],h[-6],h[-8],h[-8]],
    }
    return patterns

# 2. Các hàm cập nhật và huấn luyện mô hình
def update_transition_matrix(app, prev_result, current_result):
    if not prev_result: return
    prev_idx = 0 if prev_result == 'Tài' else 1
    curr_idx = 0 if current_result == 'Tài' else 1
    app.transition_counts[prev_idx][curr_idx] += 1
    total_transitions = sum(app.transition_counts[prev_idx])
    alpha = 1 # Laplace smoothing để tránh xác suất bằng 0
    num_outcomes = 2
    app.transition_matrix[prev_idx][0] = (app.transition_counts[prev_idx][0] + alpha) / (total_transitions + alpha * num_outcomes)
    app.transition_matrix[prev_idx][1] = (app.transition_counts[prev_idx][1] + alpha) / (total_transitions + alpha * num_outcomes)

def update_pattern_accuracy(app, predicted_pattern_name, prediction, actual_result):
    if not predicted_pattern_name: return
    stats = app.pattern_accuracy[predicted_pattern_name]
    stats['total'] += 1
    if prediction == actual_result:
        stats['success'] += 1

def train_logistic_regression(app, features, actual_result):
    y = 1.0 if actual_result == 'Tài' else 0.0
    z = app.logistic_bias + sum(w * f for w, f in zip(app.logistic_weights, features))
    try:
        p = 1.0 / (1.0 + math.exp(-z))
    except OverflowError: # Xử lý trường hợp z quá lớn hoặc quá nhỏ gây tràn số
        p = 0.0 if z < 0 else 1.0
        
    error = y - p
    app.logistic_bias += app.learning_rate * error
    for i in range(len(app.logistic_weights)):
        gradient = error * features[i]
        regularization_term = app.regularization * app.logistic_weights[i]
        app.logistic_weights[i] += app.learning_rate * (gradient - regularization_term)

def update_model_weights(app):
    """Cập nhật trọng số của các mô hình trong ensemble dựa trên hiệu suất."""
    total_accuracy_score = 0
    accuracies_raw = {}
    
    # Tính toán độ chính xác và trọng số thô
    for name, perf in app.model_performance.items():
        # Chỉ cập nhật nếu có đủ dữ liệu, nếu không giữ trọng số mặc định ban đầu
        if perf['total'] > 5: 
            accuracy = perf['success'] / perf['total']
            accuracies_raw[name] = accuracy
            total_accuracy_score += accuracy
        else:
            # Nếu chưa đủ dữ liệu, gán trọng số mặc định ban đầu để chúng có cơ hội được "học"
            accuracies_raw[name] = app.default_model_weights[name] * 2 # Nhân đôi để ưu tiên khởi tạo
            total_accuracy_score += accuracies_raw[name]

    if total_accuracy_score > 0:
        for name in app.model_weights:
            app.model_weights[name] = accuracies_raw.get(name, 0) / total_accuracy_score
    else: # Trường hợp không có dữ liệu học
        app.model_weights = app.default_model_weights.copy()
        
    # Chuẩn hóa lại để tổng bằng 1 (đảm bảo)
    sum_weights = sum(app.model_weights.values())
    if sum_weights > 0:
        for name in app.model_weights:
            app.model_weights[name] /= sum_weights
    logging.info(f"Updated model weights: {app.model_weights}")


# 3. Các hàm dự đoán cốt lõi
def detect_pattern(app, history_str):
    detected_patterns = []
    if len(history_str) < 2: return None
    
    # Tính tổng số lần xuất hiện của tất cả các pattern để chuẩn hóa recency_score
    total_occurrences = max(1, sum(s['total'] for s in app.pattern_accuracy.values()))

    for name, func in app.patterns.items():
        try:
            if func(history_str):
                stats = app.pattern_accuracy[name]
                # Độ chính xác: nếu chưa đủ dữ liệu (total < 10), gán độ chính xác mặc định (ví dụ 0.55)
                accuracy = (stats['success'] / stats['total']) if stats['total'] > 10 else 0.55 
                # Điểm gần đây: tần suất xuất hiện của pattern
                recency_score = stats['total'] / total_occurrences
                
                # Trọng số kết hợp độ chính xác lịch sử (70%) và tần suất xuất hiện (30%)
                weight = 0.7 * accuracy + 0.3 * recency_score
                detected_patterns.append({'name': name, 'weight': weight})
        except IndexError:
            continue
    if not detected_patterns:
        return None
    # Trả về pattern có trọng số cao nhất
    return max(detected_patterns, key=lambda x: x['weight'])

def predict_with_pattern(app, history_str, detected_pattern_info):
    if not detected_pattern_info or len(history_str) < 2:
        return 'Tài', 0.5 # Dự đoán mặc định và độ tin cậy thấp nếu không có pattern

    name = detected_pattern_info['name']
    last = history_str[-1]
    prev = history_str[-2]
    anti_last = 'Xỉu' if last == 'Tài' else 'Tài' # Ngược lại của kết quả cuối cùng

    # Logic dự đoán chi tiết hơn dựa trên loại pattern
    if any(p in name for p in ["Bệt", "kép", "2-2", "3-3", "4-4", "Nhịp", "Sóng vỗ", "Cầu 3-1", "Cầu 4-1", "Lặp"]):
        prediction = last # Theo cầu
    elif any(p in name for p in ["Đảo 1-1", "Xen kẽ", "lắc", "Đối ngược", "gãy", "Bậc thang", "Dài ngắn đảo"]):
        prediction = anti_last # Bẻ cầu
    elif any(p in name for p in ["Chu kỳ 2", "Gãy ngang", "Chu kỳ tăng", "Chu kỳ giảm"]):
        prediction = prev # Quay về kết quả trước đó
    elif 'Chu kỳ 3' in name:
        prediction = history_str[-3]
    elif 'Chu kỳ 4' in name:
        prediction = history_str[-4]
    elif name == "Cầu 2-1-2":
        prediction = history_str[-5] # Kết quả của phiên T-T-X-X-T-T-X
    elif name == "Cầu 1-2-1":
        prediction = anti_last # Nếu T-XX-T, dự đoán Xỉu
    elif name == "Đối xứng (Gương)":
        prediction = history_str[-3] # Dự đoán phần tử tiếp theo trong chuỗi đối xứng
    elif name == "Cầu lặp":
        prediction = history_str[-3]
    elif name == "Cầu Sandwich":
        prediction = anti_last # Nếu T-XXX-T, dự đoán Xỉu
    elif name == "Cầu Thang máy":
        prediction = history_str[-3] # Nếu T-T-X-X-T-T-X, dự đoán Tài
    else: # Mặc định cho các cầu phức tạp khác là bẻ cầu
        prediction = anti_last
        
    return prediction, detected_pattern_info['weight']

def get_logistic_features(history_str):
    if not history_str: return [0.0] * 6 # Đảm bảo trả về list đủ kích thước

    # Feature 1: Current streak length (độ dài cầu hiện tại)
    current_streak = 0
    if len(history_str) > 0:
        last = history_str[-1]
        current_streak = 1
        for i in range(len(history_str) - 2, -1, -1):
            if history_str[i] == last: current_streak += 1
            else: break
    
    # Feature 2: Previous streak length (độ dài cầu trước đó)
    previous_streak_len = 0
    if len(history_str) > current_streak:
        prev_streak_start_idx = len(history_str) - current_streak - 1
        if prev_streak_start_idx >= 0:
            prev_streak_val = history_str[prev_streak_start_idx]
            previous_streak_len = 1
            for i in range(prev_streak_start_idx - 1, -1, -1):
                if history_str[i] == prev_streak_val: previous_streak_len += 1
                else: break

    # Feature 3 & 4: Balance (Tài-Xỉu) short-term and long-term (tỷ lệ Tài/Xỉu trong quá khứ gần và xa)
    recent_history = history_str[-20:] # Lịch sử 20 phiên gần nhất
    balance_short = (recent_history.count('Tài') - recent_history.count('Xỉu')) / max(1, len(recent_history))
    
    long_history = history_str[-100:] # Lịch sử 100 phiên gần nhất
    balance_long = (long_history.count('Tài') - long_history.count('Xỉu')) / max(1, len(long_history))
    
    # Feature 5: Volatility (tần suất thay đổi giữa Tài và Xỉu)
    changes = sum(1 for i in range(len(recent_history)-1) if recent_history[i] != recent_history[i+1])
    volatility = changes / max(1, len(recent_history) - 1) if len(recent_history) > 1 else 0.0

    # Feature 6: Alternation count in last 10 results (số lần luân phiên trong 10 phiên gần nhất)
    last_10 = history_str[-10:]
    alternations = sum(1 for i in range(len(last_10) - 1) if last_10[i] != last_10[i+1])
    
    return [float(current_streak), float(previous_streak_len), balance_short, balance_long, volatility, float(alternations)]

def apply_meta_logic(prediction, confidence, history_str):
    """
    Áp dụng logic cấp cao để điều chỉnh dự đoán cuối cùng.
    Ví dụ: Logic "bẻ cầu" khi cầu quá dài.
    """
    final_prediction, final_confidence, reason = prediction, confidence, ""

    # Logic 1: Bẻ cầu khi cầu bệt quá dài (Anti-Streak)
    streak_len = 0
    if len(history_str) > 0: # Check if history_str is not empty
        last = history_str[-1]
        for x in reversed(history_str):
            if x == last: streak_len += 1
            else: break
    
    if streak_len >= 9 and prediction == history_str[-1]:
        final_prediction = 'Xỉu' if history_str[-1] == 'Tài' else 'Tài'
        final_confidence = 78.0 # Gán một độ tin cậy khá cao cho việc bẻ cầu
        reason = f"Bẻ cầu bệt siêu dài ({streak_len})"
        logging.warning(f"META-LOGIC: Activated Anti-Streak. Streak of {streak_len} detected. Forcing prediction to {final_prediction}.")
    elif streak_len >= 7 and prediction == history_str[-1]:
        final_confidence = max(50.0, confidence - 15) # Giảm độ tin cậy
        reason = f"Cầu bệt dài ({streak_len}), giảm độ tin cậy"
        logging.info(f"META-LOGIC: Long streak of {streak_len} detected. Reducing confidence.")
        
    return final_prediction, final_confidence, reason


def predict_advanced(app, history_str):
    """Hàm điều phối dự đoán nâng cao, kết hợp nhiều mô hình với trọng số động."""
    if len(history_str) < 5: # Yêu cầu tối thiểu 5 phiên lịch sử để bắt đầu dự đoán
        return "Chờ dữ liệu", "Phân tích", 50.0, {}

    last_result = history_str[-1]

    # --- Model 1: Pattern Matching ---
    detected_pattern_info = detect_pattern(app, history_str)
    patt_pred, patt_conf = predict_with_pattern(app, history_str, detected_pattern_info)
    # Scale confidence to be from 0 to 1
    patt_conf_scaled = patt_conf # pattern weight is already a confidence score

    # --- Model 2: Markov Chain ---
    last_result_idx = 0 if last_result == 'Tài' else 1
    prob_tai_markov = app.transition_matrix[last_result_idx][0]
    markov_pred = 'Tài' if prob_tai_markov >= 0.5 else 'Xỉu'
    markov_conf_scaled = max(prob_tai_markov, 1 - prob_tai_markov)

    # --- Model 3: Logistic Regression ---
    features = get_logistic_features(history_str)
    z = app.logistic_bias + sum(w * f for w, f in zip(app.logistic_weights, features))
    try:
        prob_tai_logistic = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        prob_tai_logistic = 0.0 if z < 0 else 1.0
        
    logistic_pred = 'Tài' if prob_tai_logistic >= 0.5 else 'Xỉu'
    logistic_conf_scaled = max(prob_tai_logistic, 1 - prob_tai_logistic)
    
    # Lưu lại dự đoán của từng mô hình để học
    individual_predictions = {
        'pattern': patt_pred,
        'markov': markov_pred,
        'logistic': logistic_pred
    }

    # --- Ensemble Prediction (Kết hợp các mô hình với trọng số động) ---
    # Sử dụng confidence đã được scale (0-1)
    predictions_with_weights = {
        'pattern': {'pred': patt_pred, 'conf': patt_conf_scaled, 'weight': app.model_weights['pattern']},
        'markov': {'pred': markov_pred, 'conf': markov_conf_scaled, 'weight': app.model_weights['markov']},
        'logistic': {'pred': logistic_pred, 'conf': logistic_conf_scaled, 'weight': app.model_weights['logistic']},
    }
    
    tai_score, xiu_score = 0.0, 0.0
    for model_info in predictions_with_weights.values():
        score = model_info['conf'] * model_info['weight']
        if model_info['pred'] == 'Tài': tai_score += score
        else: xiu_score += score

    final_prediction = 'Tài' if tai_score > xiu_score else 'Xỉu'
    total_score = tai_score + xiu_score
    # Chuyển đổi về phần trăm (0-100)
    final_confidence = (max(tai_score, xiu_score) / total_score * 100) if total_score > 0 else 50.0
    
    # Tăng độ tin cậy nếu pattern mạnh nhất trùng với dự đoán cuối cùng
    if detected_pattern_info and detected_pattern_info['weight'] > 0.6 and patt_pred == final_prediction:
        final_confidence = min(98.0, final_confidence + (patt_conf_scaled * 10)) # Thêm một phần nhỏ từ độ tin cậy của pattern

    # Áp dụng logic meta cuối cùng
    final_prediction, final_confidence, meta_reason = apply_meta_logic(final_prediction, final_confidence, history_str)

    used_pattern_name = detected_pattern_info['name'] if detected_pattern_info else "Ensemble"
    if meta_reason:
        used_pattern_name = meta_reason

    return final_prediction, used_pattern_name, final_confidence, individual_predictions

# --- Flask App Factory ---
def create_app():
    app = Flask(__name__)
    CORS(app)

    # --- Khởi tạo State ---
    app.lock = threading.Lock() # Lock để bảo vệ dữ liệu dùng chung giữa các luồng
    app.MAX_HISTORY_LEN = 200 # Số phiên lịch sử tối đa lưu trữ
    
    app.history = deque(maxlen=app.MAX_HISTORY_LEN) # Lưu kết quả và phiên
    app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN) # Lưu id phiên để kiểm tra trùng lặp
    app.last_fetched_session = None # Phiên cuối cùng đã được fetch từ API

    # State cho các thuật toán
    app.patterns = define_patterns() # Các pattern được định nghĩa
    app.transition_matrix = [[0.5, 0.5], [0.5, 0.5]] # Ma trận chuyển đổi Markov
    app.transition_counts = [[0, 0], [0, 0]] # Số lần chuyển đổi cho Markov
    app.logistic_weights = [0.0] * 6 # Trọng số cho Logistic Regression (tương ứng với 6 features)
    app.logistic_bias = 0.0 # Bias cho Logistic Regression
    app.learning_rate = 0.01 # Tốc độ học cho Logistic Regression
    app.regularization = 0.01 # Hệ số điều chuẩn cho Logistic Regression
    
    # State cho ensemble model động
    app.default_model_weights = {'pattern': 0.5, 'markov': 0.2, 'logistic': 0.3} # Trọng số mặc định ban đầu
    app.model_weights = app.default_model_weights.copy() # Trọng số hiện tại của các mô hình
    app.model_performance = {name: {"success": 0, "total": 0} for name in app.model_weights} # Hiệu suất từng mô hình
    
    app.overall_performance = {"success": 0, "total": 0} # Hiệu suất tổng thể của API dự đoán

    app.last_prediction = None # Lưu thông tin dự đoán cuối cùng để phục vụ việc học
    app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0}) # Hiệu suất của từng pattern

    # --- Cấu hình API endpoint mới ---
    app.TAIXIUMD5_API_URL = "https://apib52.up.railway.app/api/taixiumd5"
    logging.info(f"External TaiXiu API URL: {app.TAIXIUMD5_API_URL}")

    def fetch_data_from_api():
        """Luồng chạy ngầm để lấy dữ liệu lịch sử từ API định kỳ."""
        while True:
            try:
                response = requests.get(app.TAIXIUMD5_API_URL, timeout=10) # Thêm timeout để tránh treo
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                
                # --- ĐÃ SỬA LỖI ĐỊNH DẠNG DỮ LIỆU TỪ API BÊN NGOÀI ---
                latest_result = None
                if isinstance(data, list) and data:
                    latest_result = data[0] # Lấy phần tử đầu tiên nếu là list
                    logging.debug("External API response is a list, taking the first element.")
                elif isinstance(data, dict):
                    latest_result = data # Sử dụng trực tiếp data nếu nó là một dict
                    logging.debug("External API response is a dictionary, using it directly.")
                else:
                    logging.warning(f"External API response is not a list or dict as expected: {data}. Skipping.")
                    time.sleep(2)
                    continue
                # --- KẾT THÚC SỬA LỖI ---

                if latest_result:
                    phien = latest_result.get("Phien")
                    ket_qua = latest_result.get("Ket_qua")

                    if phien is None or ket_qua not in ["Tài", "Xỉu"]:
                        logging.warning(f"Invalid 'Phien' or 'Ket_qua' in data from external API: {latest_result}. Skipping.")
                        time.sleep(2)
                        continue

                    with app.lock: # Đảm bảo an toàn luồng khi cập nhật app.history
                        # Chỉ thêm dữ liệu mới nếu phiên chưa tồn tại hoặc là phiên mới nhất
                        if not app.session_ids or phien > app.session_ids[-1]:
                            app.session_ids.append(phien)
                            app.history.append({'ket_qua': ket_qua, 'phien': phien})
                            app.last_fetched_session = phien # Cập nhật phiên cuối cùng đã fetch
                            logging.info(f"Fetched new result for session {phien}: {ket_qua}. History length: {len(app.history)}")
                        elif phien == app.session_ids[-1]:
                            logging.debug(f"Session {phien} already in history, no new data to add.")
                        else:
                            # Xử lý trường hợp nhận được phiên cũ hơn phiên cuối cùng trong lịch sử (có thể do API trả về không sắp xếp)
                            logging.warning(f"Fetched older session {phien} (current latest: {app.session_ids[-1]}). Skipping addition.")
                
            except requests.exceptions.Timeout:
                logging.error("External API request timed out while fetching historical data.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data from external API: {e}")
            except (json.JSONDecodeError, TypeError) as e:
                logging.error(f"Error decoding external API response or invalid format: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
            except Exception as e:
                logging.error(f"Unexpected error in fetch_data_from_api: {e}", exc_info=True) # exc_info=True để in traceback
            
            time.sleep(2) # Poll API every 2 seconds

    # --- API Endpoints ---
    # KHÔNG CÓ KIỂM TRA API KEY TRÊN TẤT CẢ CÁC ENDPOINT NÀY NỮA
    @app.route("/api/taixiumd5", methods=["GET"])
    def get_taixiu_prediction():
        with app.lock:
            # Kiểm tra xem có đủ dữ liệu lịch sử để dự đoán không
            if len(app.history) < 2:
                if not app.last_fetched_session:
                    return jsonify({"error": "Đang chờ lấy dữ liệu lịch sử từ API. Vui lòng thử lại sau vài giây."}), 503
                else:
                    return jsonify({"error": "Chưa có đủ dữ liệu lịch sử để dự đoán.", "current_history_length": len(app.history)}), 503
            
            # Tạo bản sao lịch sử để thao tác mà không cần giữ khóa
            history_copy = list(app.history)
            last_prediction_copy = app.last_prediction
        
        # --- Học Online (Online Learning) ---
        if last_prediction_copy and history_copy and \
           last_prediction_copy['session'] == history_copy[-1]['phien'] + 1 and \
           not last_prediction_copy.get('learned', False):
            
            actual_result_of_learned_session = history_copy[-1]['ket_qua']
            history_at_prediction_time = _get_history_strings(history_copy[:-1]) 
            
            with app.lock: # Khóa để cập nhật state của các mô hình
                train_logistic_regression(app, last_prediction_copy['features'], actual_result_of_learned_session)
                
                if len(history_at_prediction_time) > 0:
                     update_transition_matrix(app, history_at_prediction_time[-1], actual_result_of_learned_session)
                
                update_pattern_accuracy(app, last_prediction_copy['pattern'], last_prediction_copy['prediction'], actual_result_of_learned_session)
                
                for model_name, model_pred in last_prediction_copy['individual_predictions'].items():
                    app.model_performance[model_name]['total'] += 1
                    if model_pred == actual_result_of_learned_session:
                        app.model_performance[model_name]['success'] += 1
                
                app.overall_performance['total'] += 1
                if last_prediction_copy['prediction'] == actual_result_of_learned_session:
                    app.overall_performance['success'] += 1

                update_model_weights(app)
                app.last_prediction['learned'] = True 

            logging.info(f"Learned from session {history_copy[-1]['phien']}. Predicted: {last_prediction_copy['prediction']}, Actual: {actual_result_of_learned_session}. Pattern: {last_prediction_copy['pattern']}")
        
        # --- Dự đoán cho phiên tiếp theo (Prediction) ---
        history_str_for_prediction = _get_history_strings(history_copy)
        prediction_str, pattern_str, confidence, individual_preds = predict_advanced(app, history_str_for_prediction)
        
        # Lưu lại thông tin dự đoán hiện tại để học ở lần tiếp theo (khi có kết quả thực tế)
        with app.lock:
            current_session = history_copy[-1]['phien']
            app.last_prediction = {
                'session': current_session + 1, # Phiên tiếp theo mà chúng ta đang dự đoán
                'prediction': prediction_str,
                'pattern': pattern_str,
                'features': get_logistic_features(history_str_for_prediction), # Tính lại features cho phiên này
                'individual_predictions': individual_preds,
                'learned': False # Đánh dấu là chưa học cho dự đoán này
            }
            current_result = history_copy[-1]['ket_qua']
        
        # Tinh chỉnh hiển thị độ tin cậy và dự đoán
        prediction_display = prediction_str
        final_confidence_display = round(confidence, 1)

        # Nếu độ tin cậy thấp và không phải là do logic bẻ cầu, hiển thị "Đang phân tích"
        if confidence < 70.0 and "Bẻ cầu" not in pattern_str: # Ngưỡng 70% để hiển thị dự đoán rõ ràng
            prediction_display = "Đang phân tích"
            
        return jsonify({
            "current_session": current_session,
            "current_result": current_result,
            "next_session": current_session + 1,
            "prediction": prediction_display,
            "confidence_percent": final_confidence_display,
            "suggested_pattern": pattern_str,
        })

    @app.route("/api/history", methods=["GET"])
    def get_history_api():
        with app.lock:
            hist_copy = list(app.history)
        return jsonify({"history": hist_copy, "length": len(hist_copy)})

    @app.route("/api/performance", methods=["GET"])
    def get_performance():
        with app.lock:
            # Sắp xếp pattern theo tổng số lần xuất hiện và độ chính xác
            seen_patterns = {k: v for k, v in app.pattern_accuracy.items() if v['total'] > 0}
            sorted_patterns = sorted(
                seen_patterns.items(), 
                key=lambda item: (item[1]['total'], (item[1]['success'] / item[1]['total'] if item[1]['total'] > 0 else 0)),
                reverse=True
            )
            pattern_result = {}
            for p_type, data in sorted_patterns[:30]: # Lấy 30 pattern hàng đầu có dữ liệu
                accuracy = round(data["success"] / data["total"] * 100, 2) if data["total"] > 0 else 0
                pattern_result[p_type] = { "total": data["total"], "success": data["success"], "accuracy_percent": accuracy }
            
            # Lấy hiệu suất của các mô hình con
            model_perf_result = {}
            for name, perf in app.model_performance.items():
                 accuracy = round(perf["success"] / perf["total"] * 100, 2) if perf["total"] > 0 else 0
                 model_perf_result[name] = {**perf, "accuracy_percent": accuracy}

            # Lấy hiệu suất tổng thể của API dự đoán
            overall_total = app.overall_performance['total']
            overall_success = app.overall_performance['success']
            overall_accuracy_percent = round(overall_success / overall_total * 100, 2) if overall_total > 0 else 0


        return jsonify({
            "pattern_performance": pattern_result,
            "model_performance": model_perf_result,
            "model_weights": app.model_weights,
            "overall_prediction_performance": { # Thêm hiệu suất tổng thể
                "total_predictions": overall_total,
                "correct_predictions": overall_success,
                "accuracy_percent": overall_accuracy_percent
            }
        })

    # Khởi tạo và chạy luồng lấy dữ liệu API định kỳ
    api_fetch_thread = threading.Thread(target=fetch_data_from_api, daemon=True)
    api_fetch_thread.start()
    logging.info("Background API fetching thread started.")
    
    return app

# --- Thực thi chính ---
app = create_app()

if __name__ == "__main__":
    # Render sẽ tự đặt biến môi trường PORT. 
    # Nếu chạy local, nó sẽ sử dụng 8080 làm mặc định.
    port = int(os.getenv("PORT", 8080)) 
    logging.info(f"Flask app is starting. Serving on http://0.0.0.0:{port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=8)
