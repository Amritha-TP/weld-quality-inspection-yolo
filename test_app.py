import unittest
import json
import os
import io
import cv2
import numpy as np
from PIL import Image

import config
from utils.detector import WeldDetector
from app import app


class TestWeldInspectionApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Running Weld Quality Inspection Automated Test Suite ---")
        cls.detector = WeldDetector(model_path=config.MODEL_PATH)
        cls.client = app.test_client()

    def test_01_model_loading_and_classes(self):
        """Verify YOLO model loads properly and contains expected class names."""
        self.assertTrue(self.detector.loaded, "Model should be successfully loaded.")
        print(f"Loaded model path: {self.detector.model_path}")
        print(f"Device: {self.detector.device.upper()}")
        print(f"Class Names: {self.detector.class_names}")
        self.assertIn(0, self.detector.class_names)
        self.assertIn(1, self.detector.class_names)

    def test_02_health_endpoint(self):
        """Test GET /health API endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        print(f"Health Check Response: {data}")

    def test_03_synthetic_blank_image_detection(self):
        """Test inference on a blank image (should return NO WELD DETECTED)."""
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.predict(blank_img, conf_threshold=0.25)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['overall_result'], 'NO WELD DETECTED')
        self.assertEqual(result['detection_count'], 0)
        self.assertTrue(result['processed_image'].startswith('data:image/jpeg;base64,'))
        print(f"Blank Image Test Result: {result['overall_result']}")

    def test_04_dataset_image_detection(self):
        """Test inference on a real dataset image if available."""
        test_img_path = "datasets/train/images/0e538f39-1d30436c-c23_jpg.rf.9af91763a613bf193d03c39400da2ec6.jpg"
        if not os.path.exists(test_img_path):
            # Find any image in datasets/
            for root, _, files in os.walk("datasets"):
                for f in files:
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        test_img_path = os.path.join(root, f)
                        break
                if os.path.exists(test_img_path):
                    break

        if os.path.exists(test_img_path):
            print(f"Testing real sample image: {test_img_path}")
            result = self.detector.predict(test_img_path, conf_threshold=0.25)
            self.assertTrue(result['success'])
            print(f"Sample Image Overall Result: {result['overall_result']}")
            print(f"Detections Count: {result['detection_count']}")
            for det in result['detections']:
                print(f" -> Det: {det['class_name']} ({det['confidence_percent']}%) quality={det['quality']} bbox={det['bbox']}")
            
            # Also test POST /detect API
            with open(test_img_path, 'rb') as img_f:
                data = {
                    'file': (img_f, 'test_weld.jpg'),
                    'confidence': '0.25'
                }
                response = self.client.post('/detect', data=data, content_type='multipart/form-data')
                self.assertEqual(response.status_code, 200)
                api_res = json.loads(response.data)
                self.assertTrue(api_res['success'])
                self.assertIn('processed_image', api_res)
                print(f"POST /detect API Response overall_result: {api_res['overall_result']}")

    def test_05_detect_frame_api(self):
        """Test POST /detect_frame endpoint with Base64 payload."""
        blank_img = np.zeros((320, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_img)
        b64_str = "data:image/jpeg;base64," + base64_encode(buffer)

        response = self.client.post('/detect_frame', json={'image': b64_str, 'confidence': 0.25})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['overall_result'], 'NO WELD DETECTED')
        print(f"POST /detect_frame API Response: {data['overall_result']}")


def base64_encode(buffer):
    import base64
    return base64.b64encode(buffer).decode('utf-8')


if __name__ == '__main__':
    unittest.main()
