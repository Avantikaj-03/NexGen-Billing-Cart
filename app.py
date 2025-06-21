from flask import Flask, render_template, Response, request, url_for, jsonify, send_file, redirect
import sqlite3
from ultralytics import YOLO
import cv2
import time
import uuid
from collections import Counter
import qrcode
from io import BytesIO

import requests
import numpy as np
from collections import Counter


app = Flask(__name__)

# Load YOLOv8 mode
# Load pretrained and custom models
model_pretrained = YOLO('yolov8n.pt')
model_custom = YOLO(r'best.pt')

# Model switcher: can be 'pretrained' or 'custom'
current_model_type = 'pretrained'

# Custom model label mapping
custom_label_map = {
    "Chaat Masala": "Everest Chaat Masala 50 g",
    "Pav Bhaji Masala": "Everest Pav Bhaji Masala 100 g",
    "Parachute Coconut Oil": "Parachute Coconut Oil 600 ml - Bottle"
    "MAGGI Pichkoo - Rich Tomato Ketchup 80 g Pouch"
    "Nescafe Classic Instant Coffee 24 g"
    "Maggi 2-Minute Masala Instant Noodles 420 g"
   
}


detected_items = {}  # {"item": quantity}
current_frame_detections = []
pending_items = {}  # {"item": first_detected_time}
AUTO_ADD_DELAY = 0.0

prices = {
    "apple": 5,
    "orange": 6,
    "bottle": 20,
    "banana": 3,
    "box": 15,
    "Everest Chaat Masala 50 g": 35,
    "Everest Pav Bhaji Masala 100 g": 50,
    "MAGGI Pichkoo - Rich Tomato Ketchup 80 g Pouch": 20,
    "Parachute Coconut Oil 600 ml - Bottle": 111,
    "Nescafe Classic Instant Coffee 24 g":125,
    "Maggi 2-Minute Masala Instant Noodles 420 g":68,
}


UPI_ID = "9890900842@ibl"
UPI_NAME = "NexGen Billing Cart"
ADMIN_PIN = "1234"

def init_db():
    conn = sqlite3.connect('store.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions 
                 (id TEXT PRIMARY KEY, items TEXT, total REAL, payment_mode TEXT, timestamp TEXT, payment_details TEXT, mobile_number TEXT)''')
    conn.commit()
    conn.close()

def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    print("Camera opened successfully")
    return cap





# ip = "192.168.201.27"

# camera_url = f"http://{ip}:8550/video_feed"


def gen_frames():
    global current_frame_detections, pending_items, detected_items

    try:
        stream = requests.get(camera_url, stream=True)
        bytes_buffer = b""

        for chunk in stream.iter_content(chunk_size=1024):
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]

                frame_data = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

                if frame is None:
                    continue  # Skip invalid frame

                # Run both models
                results_pre = model_pretrained(frame, imgsz=320)
                results_cust = model_custom(frame, imgsz=160)

                current_frame_detections = []
                temp_detections = []
                current_time = time.time()

                # Pretrained model detections
                for r in results_pre:
                    for box in r.boxes:
                        label = model_pretrained.names[int(box.cls)]
                        if label in prices:
                            temp_detections.append(label)
                            current_frame_detections.append(label)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} (₹{prices[label]})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Custom model detections
                for r in results_cust:
                    for box in r.boxes:
                        raw_label = model_custom.names[int(box.cls)]
                        label = custom_label_map.get(raw_label, raw_label)
                        if label in prices:
                            temp_detections.append(label)
                            current_frame_detections.append(label)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, f"{label} (₹{prices[label]})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Update detection memory
                detection_counts = Counter(temp_detections)
                for item, count in detection_counts.items():
                    if item not in detected_items and item not in pending_items:
                        pending_items[item] = current_time
                    elif item in pending_items and (current_time - pending_items[item]) >= AUTO_ADD_DELAY:
                        detected_items[item] = count
                        del pending_items[item]

                pending_items = {item: t for item, t in pending_items.items() if item in current_frame_detections}

                # Encode and stream the processed frame
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ret:
                    continue
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print("Error fetching or decoding stream from Raspberry Pi:", e)


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    total = sum(prices[item] * qty for item, qty in detected_items.items())
    return render_template('home.html', items=detected_items, total=total, prices=prices)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/items')
def get_items():
    total = sum(prices[item] * qty for item, qty in detected_items.items())
    return jsonify({"items": detected_items, "total": total, "prices": prices, "current": current_frame_detections})

@app.route('/remove/<item>', methods=['POST'])
def remove_item(item):
    if item in detected_items:
        detected_items[item] -= 1
        if detected_items[item] <= 0:
            del detected_items[item]
    return jsonify({"success": True, "items": detected_items})

@app.route('/clear', methods=['POST'])
def clear_cart():
    detected_items.clear()
    pending_items.clear()
    return jsonify({"success": True, "items": detected_items})

@app.route('/generate_upi_qr')
def generate_upi_qr():
    total = sum(prices[item] * qty for item, qty in detected_items.items())
    upi_url = f"upi://pay?pa={UPI_ID}&pn={UPI_NAME}&am={total}&cu=INR"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(upi_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

@app.route('/checkout', methods=['GET', 'POST'])
def checkout():
    total = sum(prices[item] * qty for item, qty in detected_items.items())
    if request.method == 'POST':
        if not detected_items:
            return render_template('checkout.html', total=total, items=detected_items, prices=prices, error="Cart is empty.")
        payment_mode = request.form['payment_mode']
        mobile_number = request.form.get('mobile_number', '').strip() or None
        trans_id = str(uuid.uuid4())
        items_str = ','.join(f"{item} x{qty}" for item, qty in detected_items.items())
        
        payment_details = ""
        return_amount = 0
        if payment_mode == "Cash":
            cash_tendered = float(request.form.get('cash_tendered', '0'))
            return_amount = round(cash_tendered - total, 2) if cash_tendered > total else 0
            payment_details = f"Cash Tendered: ₹{cash_tendered}"
        elif payment_mode == "Card":
            card_last4 = request.form.get('card_last4', 'XXXX')
            payment_details = f"Card Ending: {card_last4}"
        elif payment_mode == "UPI":
            upi_trans_id = request.form.get('upi_trans_id', 'N/A')
            payment_details = f"UPI Transaction ID: {upi_trans_id}"

        conn = sqlite3.connect('store.db')
        c = conn.cursor()
        c.execute("INSERT INTO transactions (id, items, total, payment_mode, timestamp, payment_details, mobile_number) VALUES (?, ?, ?, ?, datetime('now'), ?, ?)",
                  (trans_id, items_str, total, payment_mode, payment_details, mobile_number))
        conn.commit()
        conn.close()
        
        detected_items.clear()
        pending_items.clear()
        print(f"Checkout completed. Total: ₹{total}, Payment: {payment_mode}, Mobile: {mobile_number or 'None'}")
        return render_template('checkout.html',
                       trans_id=trans_id,
                       total=total,
                       payment_mode=payment_mode,
                       payment_details=payment_details,
                       mobile_number=mobile_number,
                       items_str=items_str,
                       prices=prices,
                       return_amount=return_amount)

    
    if not detected_items:
        return render_template('checkout.html', total=total, items=detected_items, prices=prices, error="Cart is empty.")
    return render_template('checkout.html', total=total, items=detected_items, prices=prices, error=None)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        pin = request.form.get('pin')
        if pin == ADMIN_PIN:
            conn = sqlite3.connect('store.db')
            c = conn.cursor()
            c.execute("SELECT * FROM transactions ORDER BY timestamp DESC")
            transactions = c.fetchall()
            conn.close()
            return render_template('admin.html', transactions=transactions, prices=prices, 
                                 detected_items=detected_items, current_detections=current_frame_detections)
        return render_template('admin_login.html', error="Invalid PIN")
    return render_template('admin_login.html', error=None)

@app.route('/update_price', methods=['POST'])
def update_price():
    item = request.form['item']
    new_price = float(request.form['price'])
    prices[item] = new_price
    return jsonify({"success": True, "prices": prices})

@app.route('/submit_review', methods=['POST'])
def submit_review():
    customer_name = request.form.get('customer_name', 'Anonymous')
    rating = int(request.form.get('rating', 0))
    review = request.form.get('review', '')

    conn = sqlite3.connect('store.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_name TEXT,
                    rating INTEGER,
                    review TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    c.execute("INSERT INTO reviews (customer_name, rating, review) VALUES (?, ?, ?)",
              (customer_name, rating, review))
    conn.commit()
    conn.close()
    return redirect(url_for('welcome'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)