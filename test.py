import os
import torch
import timm  # تأكد من تثبيت مكتبة timm
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# السماح باستخدام فئة EfficientNet من مكتبة timm
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

# Base class for emotion recognition models
class EmotiEffLibRecognizerBase:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        model_name = os.path.basename(model_path)

        # تخصيص فئات العواطف بناءً على اسم النموذج
        if "_7" in model_name:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Disgust",
                2: "Fear",
                3: "Happiness",
                4: "Neutral",
                5: "Sadness",
                6: "Surprise",
            }
        else:
            self.idx_to_emotion_class = {
                0: "Anger",
                1: "Contempt",
                2: "Disgust",
                3: "Fear",
                4: "Happiness",
                5: "Neutral",
                6: "Sadness",
                7: "Surprise",
            }

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_size = 224  # حجم الصورة الافتراضي
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        تحميل النموذج من المسار المحدد
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        
        # محاولة تحميل النموذج باستخدام safe_globals لضمان تحميل الفئة EfficientNet
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()  # وضع النموذج في وضع التقييم
        return model

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess an image before passing it to the model.
        """
        test_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        return test_transforms(Image.fromarray(img))

    def classify_emotions(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify emotions based on the features.
        """
        # استخراج الاحتمالات باستخدام المخرجات مباشرة من النموذج
        with torch.no_grad():  # تعطيل حساب التدرجات أثناء التقييم
            outputs = self.model(features)
        
        # تحويل المخرجات إلى احتمالات باستخدام softmax
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs)
        
        # الحصول على التصنيف الأعلى
        _, preds = torch.max(probs, 1)

        # إرجاع العاطفة المتوقعة والاحتمال المقابل
        emotion = self.idx_to_emotion_class[preds.item()]
        confidence = probs[0, preds].item()

        return emotion, confidence

    def extract_features(self, face_img: np.ndarray) -> torch.Tensor:
        """
        استخراج السمات من صورة الوجه.
        """
        img_tensor = self._preprocess(face_img)
        img_tensor.unsqueeze_(0)  # إضافة بُعد الدفعة
        return img_tensor  # إرجاع الصورة بدون تغييرها

# Function to test images in a folder using a specified model
def test_images_in_folder(folder_path: str, model_path: str, model_name: str) -> None:
    """
    اختبار التعرف على العواطف في الصور من المجلد باستخدام نموذج معين.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return
    
    # قائمة الصور في المجلد
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in folder '{folder_path}'!")
        return
    
    # إنشاء مثيل من المعرف
    recognizer = EmotiEffLibRecognizerBase(model_path=model_path)

    print(f"\nTesting with model: {model_name}")
    
    # معالجة كل صورة
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {image_file}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # تحويل من BGR إلى RGB
        features = recognizer.extract_features(img)  # استخراج السمات من الصورة
        emotion, confidence = recognizer.classify_emotions(features)  # تصنيف العواطف

        # عرض النتيجة بشكل منظم
        print(f"  Image: {image_file}")
        print(f"    Predicted Emotion: {emotion}")
        print(f"    Confidence: {confidence * 100:.2f}%\n")

# Example usage
folder_path = "images"  # Path to the folder with images

# Model paths for "enet_b2_7" and "enet_b2_8"
model_paths = {
    "enet_b2_7": "model/enet_b2_7.pt",
}

# Test using "enet_b2_7"
test_images_in_folder(folder_path, model_paths["enet_b2_7"], "enet_b2_7")

