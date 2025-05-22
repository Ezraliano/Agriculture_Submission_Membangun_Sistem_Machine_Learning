from prometheus_client import start_http_server, Histogram, Counter
import threading
from flask import Flask

# Definisikan metrik
REQUEST_LATENCY = Histogram('request_latency_ms', 'Latency of incoming requests')
REQUEST_COUNT = Counter('request_count_total', 'Total number of HTTP requests')

# Dummy server Flask untuk trigger metrik
app = Flask(__name__)

@app.route("/trigger")
def trigger():
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        # Simulasi proses
        time.sleep(0.1)
    return "Request recorded"

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)

    # Jalankan Flask server
    app.run(host="0.0.0.0", port=5001)