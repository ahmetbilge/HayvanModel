"""
Görüntü Sınıflandırıcı Web Arayüzü

Bu modül, eğitilmiş bir model kullanarak görüntü sınıflandırma yapan
bir web arayüzü sunar. Gradio kütüphanesi kullanılarak oluşturulmuştur.

Kullanım:
    python app.py --model_dir "model_klasörü"
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import json
import argparse
import time
from PIL import Image
import gradio as gr

import matplotlib.pyplot as plt
import cv2


def parse_arguments():
    """
    Komut satırı argümanlarını ayrıştırır.

    Returns:
        argparse.Namespace: Ayrıştırılmış argümanlar
    """
    parser = argparse.ArgumentParser(description='Görüntü sınıflandırma web arayüzü')
    parser.add_argument('--model_dir', type=str, default='model', help='Model klasörü')
    parser.add_argument('--image_size', type=int, default=100, help='Görüntü boyutu')
    return parser.parse_args()


def preprocess_image(image, image_size):
    """
    Görüntüyü model için hazırlar.

    Args:
        image: Giriş görüntüsü (PIL Image, numpy dizisi veya dosya yolu)
        image_size: Yeniden boyutlandırma boyutu

    Returns:
        numpy.ndarray: İşlenmiş görüntü (batch olarak)
    """
    # Eğer dosya yoluysa, görüntüyü yükle
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Eğer numpy dizisiyse PIL görüntüsüne dönüştür
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8')).convert("RGB")

    # Görüntüyü yeniden boyutlandır
    image = image.resize((image_size, image_size))

    # Numpy dizisine dönüştür ve normalize et
    img_array = np.array(image) / 255.0

    # Batch boyutu ekle
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def apply_gradcam(model, img_array, last_conv_layer_name="conv2d_4"):
    """
    Grad-CAM görselleştirmesi uygular.

    Args:
        model: Eğitilmiş model
        img_array: İşlenmiş görüntü
        last_conv_layer_name: Son konvolüsyon katmanının adı

    Returns:
        numpy.ndarray: Grad-CAM görselleştirmesi
    """
    # Son konvolüsyon katmanı çıktısını almak için model oluştur
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradient tape ile gradient hesapla
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    # Feature map için gradientleri hesapla
    grads = tape.gradient(top_class_channel, conv_outputs)

    # Global average pooling uygula
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Feature map ile ağırlıkları çarp
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Heatmap'i normalize et
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Heatmap'i yeniden boyutlandır ve renklendir
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Orijinal görüntü ile heatmap'i birleştir
    original_img = (img_array[0] * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return superimposed_img


def classify_image(image, model, label_map, image_size):
    """
    Görüntüyü sınıflandırır.

    Args:
        image: Giriş görüntüsü
        model: Eğitilmiş model
        label_map: Etiket haritası
        image_size: Görüntü boyutu

    Returns:
        tuple: (sınıf_adı, olasılıklar_sözlüğü, açıklama)
    """
    start_time = time.time()

    # Görüntü işleme
    try:
        processed_image = preprocess_image(image, image_size)
    except Exception as e:
        return None, {}, f"Görüntü işlenirken hata oluştu: {str(e)}"

    # Tahmin yap
    prediction = model.predict(processed_image)[0]

    # Sonuçları yorumla
    predicted_index = np.argmax(prediction)
    predicted_probability = float(prediction[predicted_index])

    # Etiket haritasını tersine çevir (index -> sınıf adı)
    reverse_map = {int(v): k for k, v in label_map.items()}
    predicted_class = reverse_map[predicted_index]

    # Tüm sınıflar için olasılıkları hazırla
    probabilities = {reverse_map[i]: float(prediction[i]) for i in range(len(prediction))}

    # En yüksek 3 tahmini al
    top_3_indices = np.argsort(prediction)[-3:][::-1]
    top_3_classes = [reverse_map[i] for i in top_3_indices]
    top_3_probs = [float(prediction[i]) for i in top_3_indices]

    # Açıklama metni oluştur
    elapsed_time = time.time() - start_time
    description = f"Tahmin: {predicted_class} ({predicted_probability:.2%})\n"
    description += f"İşlem süresi: {elapsed_time:.2f} saniye\n\n"
    description += "En olası 3 sınıf:\n"

    for cls, prob in zip(top_3_classes, top_3_probs):
        description += f"- {cls}: {prob:.2%}\n"

    return predicted_class, probabilities, description


def create_interface(model, label_map, image_size):
    """
    Gradio arayüzü oluşturur.

    Args:
        model: Eğitilmiş model
        label_map: Etiket haritası
        image_size: Görüntü boyutu

    Returns:
        gradio.Interface: Oluşturulan arayüz
    """

    def predict(image):
        if image is None:
            return "Lütfen bir görüntü yükleyin.", None, None

        class_name, probs, description = classify_image(image, model, label_map, image_size)

        # Sınıflandırma başarısızsa
        if class_name is None:
            return description, None, None

        # Olasılık grafiği oluştur
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]  # En yüksek 5 sınıf
        labels = [item[0] for item in sorted_probs]
        values = [item[1] for item in sorted_probs]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color='skyblue')
        plt.title('En Olası Sınıflar')
        plt.xlabel('Sınıf')
        plt.ylabel('Olasılık')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()

        # Geçici dosya olarak kaydet
        plot_path = "temp_plot.png"
        plt.savefig(plot_path)
        plt.close()

        return description, class_name, plot_path

    with gr.Blocks(title="Görüntü Sınıflandırıcı") as interface:
        gr.Markdown("""
        # Yapay Zeka Destekli Görüntü Sınıflandırıcı

        Bu uygulama, yüklediğiniz görüntüyü analiz ederek hangi kategoriye ait olduğunu tahmin eder.
        Desteklenen sınıflar: {}.

        Bir görüntü yükleyin ve "Tahmin Et" butonuna tıklayın.
        """.format(", ".join(label_map.keys())))

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Görüntü Yükle")
                predict_btn = gr.Button("Tahmin Et", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Tahmin Sonucu", lines=6)
                output_class = gr.Textbox(label="Tahmin Edilen Sınıf")
                output_plot = gr.Image(label="Olasılık Dağılımı")

        predict_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_text, output_class, output_plot]
        )

        gr.Examples(
            examples=[
                # Örnek görüntüler eklenerek kullanıcıların test etmesi sağlanabilir
                # Proje ödevinize uygun örnek görüntüler ekleyebilirsiniz
            ],
            inputs=input_image
        )

        gr.Markdown("""
        ### Nasıl Çalışır?
        1. Görüntü yükleyin (desteklenen formatlar: jpg, png, bmp)
        2. "Tahmin Et" butonuna tıklayın
        3. Model görüntüyü analiz edecek ve en olası sınıfı gösterecektir

        ### Proje Hakkında
        Bu proje, PEAKUP-Bulut Bilişim ve Yapay Zeka Teknolojileri Dersi kapsamında geliştirilmiştir.
        Model, eğitim sırasında kullanılan veri setindeki sınıfları tanımak üzere eğitilmiştir.
        """)

    return interface


def main():
    """
    Ana işlev: Komut satırı argümanlarını alır ve web arayüzünü başlatır.
    """
    args = parse_arguments()

    # Model ve etiket haritasını yükle
    model_path = os.path.join(args.model_dir, "model.h5")
    label_map_path = os.path.join(args.model_dir, "label_map.json")

    try:
        model = load_model(model_path)
        print(f"Model başarıyla yüklendi: {model_path}")

        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        print(f"Etiket haritası başarıyla yüklendi: {label_map_path}")

    except Exception as e:
        print(f"Model veya etiket haritası yüklenirken hata oluştu: {str(e)}")
        return

    # Web arayüzü oluştur ve başlat
    interface = create_interface(model, label_map, args.image_size)
    interface.launch(share=True)


if __name__ == "__main__":
    try:
        import tensorflow as tf

        main()
    except Exception as e:
        print(f"Uygulama başlatılırken bir hata oluştu: {str(e)}")