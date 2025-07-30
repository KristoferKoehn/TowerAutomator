import sys
import threading
import queue
import easyocr
import os

# Optional: Limit GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class OCRRequest:
    def __init__(self, image):
        self.image = image
        self.event = threading.Event()
        self.result = None

class OCRPoolSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, num_workers=2):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_pool(num_workers)
            return cls._instance

    def _init_pool(self, num_workers):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.shutdown_event = threading.Event()

        self.workers = []

        print("Initializing thread pool:")
        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

            # Progress bar
            bar_length = 40
            filled_length = int(bar_length * (i + 1) / num_workers)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            percent = int((i + 1) / num_workers * 100)
            sys.stdout.write(f'\r|{bar}| {percent}%')
            sys.stdout.flush()

            # Optional: slow it down for visibility (remove or adjust in production)
            # time.sleep(0.05)

        print("\nPool initialized.")

    def _worker_loop(self):
        reader = easyocr.Reader(['en'], gpu=True)
        while not self.shutdown_event.is_set():
            try:
                request = self.task_queue.get(timeout=0.1)
                try:
                    result = reader.readtext(request.image, height_ths=1.0, y_ths=1.0)
                    request.result = result
                except Exception as e:
                    print("OCR error:", e)
                    request.result = []
                finally:
                    request.event.set()  # signal result is ready
            except queue.Empty:
                continue

    def submit_and_wait(self, image, timeout=None):
        """Submit image and block until OCR result is ready (or timeout)."""
        request = OCRRequest(image)
        self.task_queue.put(request)
        if not request.event.wait(timeout):
            raise TimeoutError("OCR processing timed out.")
        return request.result

    def stop(self):
        self.shutdown_event.set()
        for t in self.workers:
            t.join()

# Singleton accessor
ocr_pool = OCRPoolSingleton(num_workers=5)