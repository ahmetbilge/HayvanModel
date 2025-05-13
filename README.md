# 🐾 Hayvan Görüntü Sınıflandırıcı (CNN + Gradio)

Bu proje, derin öğrenme (CNN) mimarisi kullanılarak eğitilmiş bir modelle hayvan resimlerini sınıflandıran interaktif bir web arayüzü sunar. Uygulama, **Keras (TensorFlow)** kütüphanesi ile oluşturulmuş bir **Convolutional Neural Network (CNN)** modelini kullanır ve son kullanıcıya **Gradio** üzerinden kullanıcı dostu bir deneyim sağlar.

## 📚 Kullanılan Veri Seti

Bu proje için kullanılan eğitim verisi:

🔗 [Animals-10 Dataset (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

Bu veri setinde 10 farklı hayvan kategorisine ait toplam 28.000'den fazla etiketlenmiş görüntü bulunmaktadır:

- Cat 🐱
- Dog 🐶
- Horse 🐴
- Elephant 🐘
- Butterfly 🦋
- Chicken 🐔
- Cow 🐮
- Sheep 🐑
- Spider 🕷️
- Squirrel 🐿️

## 🧠 Kullanılan Yöntem: CNN (Convolutional Neural Network)

Model mimarisi:
- Convolutional katmanlar (özellik çıkarımı)
- MaxPooling katmanları (boyut azaltma)
- Dense (Tam Bağlantılı) katmanlar
- Dropout (overfitting’i önlemek için)
- Aktivasyon Fonksiyonları: ReLU ve Softmax

Modelin eğitimi Keras kullanılarak yapılmış, `.h5` formatında kaydedilmiştir.

## 🌐 Arayüz: Gradio

Kullanıcılar, tarayıcı üzerinden kolayca:
- Bir görüntü yükleyebilir
- “Tahmin Et” butonuyla sınıf tahmini alabilir
- Olasılık dağılımını grafiksel olarak görebilir

## 🛠️ Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install tensorflow gradio numpy pillow matplotlib opencv-python

## 🛠️ modeli çalıştırmak için 
```bash
(Dosya yoluna gelip) "python app.py" Komutu çalıştırlmalıdır

