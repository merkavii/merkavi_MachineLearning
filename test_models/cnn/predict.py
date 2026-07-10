# predict.py
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


# فقط یک بار مدل را لود می‌کنیم (خارج از تابع)
print("Loading model...")
MODEL = load_model(r"D:/machine_learning/deep_learning/cats_dogs_model.keras")
print("Model loaded!")


def predict_image(img_path: str = None, img_array = None) -> dict:
    """
    پیش‌بینی می‌کند که تصویر گربه است یا سگ
    
    پارامترها:
        img_path: مسیر فایل تصویر (اختیاری)
        img_array: آرایه numpy تصویر که آماده شده (اختیاری)
    
    فقط یکی از دو پارامتر بالا باید داده شود.
    
    خروجی: دیکشنری با کلیدهای result و confidence
    """
    if img_path is None and img_array is None:
        raise ValueError("حداقل یکی از img_path یا img_array باید مقدار داشته باشد")

    # اگر مسیر دادیم → تصویر را لود و آماده می‌کنیم
    if img_path is not None:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

    # اگر قبلاً آرایه آماده شده بود، مستقیم استفاده می‌کنیم
    # نکته: فرض می‌کنیم img_array شکل درستی دارد (1,64,64,3)

    # اگر موقع آموزش نرمال‌سازی کردی، اینجا هم انجام بده
    img_array = img_array / 255.0     # ← اگر لازم است فعال کن

    prediction = MODEL.predict(img_array, verbose=0)

    # خروجی مدل (فرض: تک خروجی sigmoid)
    prob_dog = float(prediction[0][0])
    prob_cat = 1 - prob_dog

    if prob_dog > 0.5:
        label = "Dog"
        confidence = prob_dog
    else:
        label = "Cat"
        confidence = prob_cat

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "is_dog": prob_dog > 0.5,
        "raw_prob_dog": round(prob_dog * 100, 2)
    }


# فقط برای تست مستقیم (وقتی فایل را اجرا می‌کنی)
if __name__ == "__main__":
    while True:
        path = input("Enter image path (or type exit): ")
        if path.lower() == "exit":
            break

        try:
            result = predict_image(img_path=path)
            emoji = "🐶" if result["is_dog"] else "🐱"
            print(f"{emoji} {result['label']} - اطمینان: {result['confidence']}%")
        except Exception as e:
            print("خطا:", e)